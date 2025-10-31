#!/usr/bin/env python3
import os, math
import numpy as np
import casadi as ca
import h5py
import subprocess
import time

# ✨ 导入 Acados ✨
from acados_template import get_tera
get_tera(tera_version='0.0.34', force_download=True)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# ===== vehicle dynamic model (保持不变) =====
L = 0.6 #wheel base
R_EGO = 0.5 #radius 
DT = 0.1 #integral step length
N  = 20  #prediction horizon
K_NEAR = 2 #consider nearest 2 obstacles
D_SAFE = 0.1

Y_MIN, Y_MAX = -2, 2 
V_MIN, V_MAX = 0.0, 3.0
A_MIN, A_MAX = -3.0, 2.0
DELTA_MAX  = np.deg2rad(30)
DELTA_RATE = np.deg2rad(40)

OVX_MIN, OVX_MAX = -1.0,1.0 #obstale form and velocity limit
OVY_MIN, OVY_MAX = -0.2,0.2
OR_MIN, OR_MAX = 0.1,0.5

# cost function weights
W_EY   = 6.0   # 横向误差 y 偏差 (lateral error)
W_EPSI = 6.0   # 航向角误差 psi 偏差 (heading error)
W_V    = 2.0   # 度误差 (deviation from v_ref)
W_U    = 1e-2  # 控制量本身的权重 (penalize using a, delta)
# W_DU (控制率) 将通过状态扩充和约束来处理
RHO_SLACK = 1e3  # 对约束违背的惩罚 (soft constraints)

OUT_PATH = "mpc_dataset.h5"
np.random.seed(None)

# ===== 车辆动力学模型 (f_cont, 保持不变) =====
def f_cont(st, u):
    # st = [x, y, psi, v]
    x, y, psi, v = st[0], st[1], st[2], st[3]
    a, delt      = u[0], u[1]
    dx   = v*ca.cos(psi)
    dy   = v*ca.sin(psi)
    dpsi = (v/L)*ca.tan(delt)
    dv   = a
    return ca.vertcat(dx, dy, dpsi, dv)

# ===== rk4 (保持不变, 用于主循环仿真) =====
def rk4(st, u, dt=DT):
    k1 = f_cont(st,u)
    k2 = f_cont(st + dt/2*k1, u)
    k3 = f_cont(st + dt/2*k2, u)
    k4 = f_cont(st + dt   *k3, u)
    return st + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ===============================================
# ===== ✨ 新的 NMPC Acados 求解器 ✨ =====
# ===============================================
class NMPC_Acados:
    def __init__(self, model_name="nmpc_acados_solver"):
        self.N = N
        self.K = K_NEAR
        self.DT = DT
        self.model_name = model_name

        self.solver = self._build_solver()

        # 维度信息 (用于解包)
        self.nx = self.solver.acados_ocp.dims.nx
        self.nu = self.solver.acados_ocp.dims.nu
        self.np = self.solver.acados_ocp.dims.np

    def _build_solver(self):
        print(f"Building Acados NLP (N={self.N}, K={self.K})...")
        ocp = AcadosOcp()
        ocp.model = AcadosModel()

        # --- 1. 定义维度 ---
        nx = 5
        nu = 2
        np_k = 1 + 3 * self.K # 7
        nh = 1 + self.K # 3
        nsh = self.K 
        
        # ✨ 修复 1：明确定义所有阶段的维度 (k=0, k=1...N-1, k=N)
        ocp.dims.N = self.N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = np_k
        
        # 初始阶段 (k=0) 约束
        ocp.dims.nh_0 = nh
        ocp.dims.nsh_0 = nsh
        
        # 中间阶段 (k=1...N-1) 约束
        ocp.dims.nh = nh   
        ocp.dims.nsh = nsh 
        
        # 终端阶段 (k=N) 约束
        ocp.dims.nh_e = 0

        # --- 2. 定义符号 ---
        x_aug_sym = ca.SX.sym('x_aug', nx)
        u_sym = ca.SX.sym('u', nu)
        p_sym = ca.SX.sym('p', np_k)

        # --- 3. 定义动力学模型 ---
        st_sym = x_aug_sym[0:4] # [x, y, psi, v]
        delta_prev_sym = x_aug_sym[4]
        f_cont_expr = f_cont(st_sym, u_sym)
        ocp.model.f_expl_expr = ca.vertcat(f_cont_expr, u_sym[1])
        ocp.model.x = x_aug_sym
        ocp.model.u = u_sym
        ocp.model.p = p_sym
        ocp.model.name = self.model_name
        
        # ✨ 修复 2：为参数 p 提供一个正确形状的初始值 (修复您当前的 Bug)
        ocp.parameter_values = np.zeros(np_k)

        # --- 4. 定义代价函数 (非线性最小二乘) ---
        ocp.cost.cost_type_0 = 'NONLINEAR_LS' # 初始代价 (k=0)
        ocp.cost.cost_type = 'NONLINEAR_LS'   # 阶段代价 (k=1...N-1)
        ocp.cost.cost_type_e = 'NONLINEAR_LS' # 终端代价 (k=N)

        v_ref_sym = p_sym[0]
        
        # 初始代价 y_0 (k=0) - 仅状态
        ocp.model.cost_y_expr_0 = ca.vertcat(
            x_aug_sym[1],                 # ey
            x_aug_sym[2],                 # epsi
            x_aug_sym[3] - v_ref_sym      # v_err
        )
        ocp.cost.W_0 = np.diag([W_EY, W_EPSI, W_V])
        ocp.dims.ny_0 = ocp.model.cost_y_expr_0.shape[0]
        # ✨ 修复 3：为 yref_0 提供一个正确形状的初始值
        ocp.cost.yref_0 = np.zeros(ocp.dims.ny_0)

        # 阶段代价 y (k=1...N-1) - 状态 + 控制
        ocp.model.cost_y_expr = ca.vertcat(
            x_aug_sym[1],                 # ey
            x_aug_sym[2],                 # epsi
            x_aug_sym[3] - v_ref_sym,     # v_err
            u_sym[0],                     # a
            u_sym[1]                     # delta
        )
        ocp.cost.W = np.diag([W_EY, W_EPSI, W_V, W_U, W_U])
        ocp.dims.ny = ocp.model.cost_y_expr.shape[0]
        # ✨ 修复 3：为 yref 提供一个正确形状的初始值
        ocp.cost.yref = np.zeros(ocp.dims.ny)

        # 终端代价 y_e (k=N) - 仅状态
        ocp.model.cost_y_expr_e = ca.vertcat(
            x_aug_sym[1],
            x_aug_sym[2],
            x_aug_sym[3] - v_ref_sym
        )
        ocp.cost.W_e = np.diag([W_EY, W_EPSI, W_V])
        ocp.dims.ny_e = ocp.model.cost_y_expr_e.shape[0]
        # ✨ 修复 3：为 yref_e 提供一个正确形状的初始值
        ocp.cost.yref_e = np.zeros(ocp.dims.ny_e)


        # --- 5. 定义约束 ---
        
        # A. 状态盒子约束 (Box on x)
        # ✨ 修复 4：明确定义初始、中间、终端的状态盒子约束
        ocp.constraints.idxbx_0 = np.array([1, 3])
        ocp.constraints.lbx_0 = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx_0 = np.array([Y_MAX, V_MAX])

        ocp.constraints.idxbx = np.array([1, 3])
        ocp.constraints.lbx = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx = np.array([Y_MAX, V_MAX])
        
        ocp.constraints.idxbx_e = np.array([1, 3])
        ocp.constraints.lbx_e = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx_e = np.array([Y_MAX, V_MAX])

        # B. 控制盒子约束 (Box on u) (k=0...N-1)
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbu = np.array([A_MIN, -DELTA_MAX])
        ocp.constraints.ubu = np.array([A_MAX, DELTA_MAX])

        # --- C. 阶段非线性约束 (h) ---
        h_expr_list = []
        h_delta_rate = u_sym[1] - delta_prev_sym
        h_expr_list.append(h_delta_rate)

        for j in range(self.K):
            obs_x_sym = p_sym[1 + j*3 + 0]
            obs_y_sym = p_sym[1 + j*3 + 1]
            obs_r_sym = p_sym[1 + j*3 + 2]
            dx = x_aug_sym[0] - obs_x_sym
            dy = x_aug_sym[1] - obs_y_sym
            rj = obs_r_sym
            h_obs_j = (dx*dx + dy*dy) - (D_SAFE + R_EGO + rj)**2
            h_expr_list.append(h_obs_j)

        # ←—— 关键：使用 acados 期望的属性名 con_h_expr
        ocp.model.con_h_expr = ca.vertcat(*h_expr_list)

        # 保证第 0 阶段也显式使用相同表达式（你已经声明了 nh_0）
        ocp.model.con_h_expr_0 = ocp.model.con_h_expr

        # 如果你不在末端使用 h 约束，con_h_expr_e 可留空或不设置（nh_e = 0），
        # 若设置终端约束则要显式赋值:
        # ocp.model.con_h_expr_e = ca.vertcat()  # 仅当 nh_e > 0 时才需要


        # D. 阶段约束边界 (lh, uh)
        # (1 个 delta_rate 约束) + (K 个 障碍物约束)
        lh_list = [-DELTA_RATE * DT] + [0.0] * self.K
        #uh_list = [DELTA_RATE * DT] + [np.inf] * self.K  # <-- 用 np.inf 代替 ca.inf
        uh_list = [DELTA_RATE * DT] + [1e8] * self.K
        base_lh = np.array(lh_list, dtype=np.float64)
        base_uh = np.array(uh_list, dtype=np.float64)

        ocp.constraints.lh_0 = base_lh
        ocp.constraints.uh_0 = base_uh
        ocp.constraints.lh = base_lh
        ocp.constraints.uh = base_uh

        # E. 软约束设置
        # idxsh 是约束索引（0-based），这里第0个是 delta_rate，1..K 是障碍物
        base_idxsh = np.arange(1, self.K + 1, dtype=np.int32)  # [1,2,...]，int32
        ocp.constraints.idxsh_0 = base_idxsh
        ocp.constraints.idxsh = base_idxsh

        # 为 K 个障碍物约束设置惩罚（必须是 float 数组，长度 = nsh_0）
        base_Zl = np.array([RHO_SLACK] * self.K, dtype=np.float64)
        base_Zu = np.array([RHO_SLACK] * self.K, dtype=np.float64)
        base_zl = np.array([0.0] * self.K, dtype=np.float64)
        base_zu = np.array([0.0] * self.K, dtype=np.float64)

        ocp.cost.Zl_0 = base_Zl
        ocp.cost.Zu_0 = base_Zu
        ocp.cost.zl_0 = base_zl
        ocp.cost.zu_0 = base_zu

        ocp.cost.Zl = base_Zl
        ocp.cost.Zu = base_Zu
        ocp.cost.zl = base_zl
        ocp.cost.zu = base_zu


        # --- 6. 求解器设置 ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_steps = 4 
        ocp.solver_options.tf = self.N * self.DT

        # ✨ V V V 在这里添加 V V V ✨
        print("\n--- Acados Sanity Check ---")
        print(f"K = {self.K}")
        print(f"Declared nh_0 (dims): {ocp.dims.nh_0}")
        print(f"Provided lh_0 (constraints) len: {len(ocp.constraints.lh_0)}")
        print(f"Provided uh_0 (constraints) len: {len(ocp.constraints.uh_0)}")
        print(f"Declared nsh_0 (dims): {ocp.dims.nsh_0}")
        print(f"Provided idxsh_0 (constraints) len: {len(ocp.constraints.idxsh_0)}")
        print(f"Expression h (model) len: {ocp.model.con_h_expr.shape[0]}")
        print("---------------------------\n")
        # ✨ ^ ^ ^ 添加到这里 ^ ^ ^ ✨
        print("con_h_expr exists:", hasattr(ocp.model, "con_h_expr"))
        if hasattr(ocp.model, "con_h_expr"):
            print("con_h_expr.shape:", ocp.model.con_h_expr.shape) 
        print("con_h_expr_0 exists:", hasattr(ocp.model, "con_h_expr_0"))
        if hasattr(ocp.model, "con_h_expr_0"):
            print("con_h_expr_0.shape:", ocp.model.con_h_expr_0.shape)



        # --- 7. 创建求解器 ---
        json_file = self.model_name + '.json'
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True)
        
        print("Acados solver build complete.")
        return solver

    def solve(self, x0_aug, v_ref, obs_xy_2d, obs_r_2d, X_guess, U_guess):
        
        # 1. 设置初始状态
        self.solver.set(0, 'x', x0_aug)
        
        # 2. 设置时变参数 p
        p_matrix = np.zeros((self.N, self.np))
        p_matrix[:, 0] = v_ref
        for j in range(self.K):
            p_matrix[:, 1 + j*3 + 0] = obs_xy_2d[j*2, :]     # obs_x
            p_matrix[:, 1 + j*3 + 1] = obs_xy_2d[j*2 + 1, :] # obs_y
            p_matrix[:, 1 + j*3 + 2] = obs_r_2d[j, :]        # obs_r
        
        for k in range(self.N):
            self.solver.set(k, 'p', p_matrix[k, :])

        # 3. 设置 Warm Start (对于 RTI 至关重要)
        for k in range(self.N):
            self.solver.set(k, 'x', X_guess[:, k])
            self.solver.set(k, 'u', U_guess[:, k])
        self.solver.set(self.N, 'x', X_guess[:, self.N])
        
        # 4. 求解
        status = self.solver.solve()
        
        if status != 0:
            print(f"Acados solver failed with status {status} at time step.")
            # 即便失败，也尝试获取上一步的解 (RTI 特性)
        
        # 5. 获取结果
        U_opt = np.array([self.solver.get(k, 'u') for k in range(self.N)]).T
        X_opt_aug = np.array([self.solver.get(k, 'x') for k in range(self.N + 1)]).T
        
        return {
            "success": status == 0, 
            "U": U_opt,         # [nu, N]
            "X_aug": X_opt_aug  # [nx, N+1] (5D 状态)
        }

# ===== random obstacles and logs (保持不变) =====
def gen_static_obstacles(M=2, x_min=2.0, x_max=10.0):
    obs = []
    for i in range(M):
        xi = np.random.uniform(x_min, x_max) 
        yi = np.random.uniform(Y_MIN, Y_MAX)
        vxi = np.random.uniform(OVX_MIN, OVX_MAX)
        vyi = np.random.uniform(OVY_MIN, OVY_MAX)
        ri = np.random.uniform(OR_MIN, OR_MAX)
        o = {"id":i, "x": xi, "y": yi, "vx": vxi, "vy": vyi, "r": ri}
        obs.append(o)
        print(f"[gen_static_obstacles] Obstacle {i}: x={o['x']:.2f}, y={o['y']:.2f}, vx={o['vx']}, vy={o['vy']}, r={o['r']}")
    return obs

def propagate_obstacles(obstacles, dt=DT):
    for i, o in enumerate(obstacles):
        o["x"] += o["vx"]*dt
        o["y"] += o["vy"]*dt

def nearest_k_obstacles(obstacles, x, y, K=K_NEAR):
    d2 = [(((o["x"]-x)**2 + (o["y"]-y)**2), o) for o in obstacles]
    d2.sort(key=lambda t: t[0])
    sel = [t[1] for t in d2[:K]] + [None]*max(0, K-len(d2))
    out = []
    for j in range(K):
        out.append(sel[j] if sel[j] is not None else
           {"id": -1, "x": x+50.0, "y": 0.0, "vx":0.0, "vy":0.0, "r":0.0})
    return out

class EpisodeLogger:
    # ... (此类保持不变, 已折叠) ...
    def __init__(self, K_obs=K_NEAR):
        self.buf = []; 
        self.K = K_obs
    def step(self, t, state, ctrl, obstacles_near, label_ref, boundaries):
        env = []
        x, y = state["x"], state["y"]
        dist_margin_list = []
        for j in range(self.K):
            o = obstacles_near[j]
            dx, dy = o["x"]-x, o["y"]-y
            dist   = math.hypot(dx, dy)
            margin = dist - (R_EGO + D_SAFE + o.get("r", 0.0))
            env += [dx, dy, margin, o["vx"], o["vy"]]
        env += [boundaries["d_left"], boundaries["d_right"], state.get("ey",0.0), state.get("epsi",0.0)]
        obs_xy = []
        obs_id = []
        for j in range(self.K):
            o = obstacles_near[j]
            obs_xy += [o["x"], o["y"], o["vx"], o["vy"], o.get("r", 0.0)]
            obs_id.append(int(o.get("id", -1)))
        self.buf.append({
            "t": t,
            "state": [state["x"], state["y"], state["psi"], state["v"]],
            "ctrl":  [ctrl["a"], ctrl["delta"]],
            "env":   env,
            "label_ref": [label_ref[0], label_ref[1]],
            "obs_xy": obs_xy, 
            "obs_id": obs_id,
        })
    def to_h5(self, h5file, traj_name, mode ="a"):
        os.makedirs(os.path.dirname(h5file) or ".", exist_ok=True)
        t   = np.array([r["t"] for r in self.buf], dtype=np.float32)
        st  = np.array([r["state"] for r in self.buf], dtype=np.float32)
        ct  = np.array([r["ctrl"]  for r in self.buf], dtype=np.float32)
        env = np.array([r["env"]   for r in self.buf], dtype=np.float32)
        ref = np.array([r["label_ref"] for r in self.buf], dtype=np.float32)
        obs_dim = self.K * 5
        obs = np.array([r.get("obs_xy", [np.nan]*obs_dim) for r in self.buf], dtype=np.float32)
        ids = np.array([r.get("obs_id", [-1]*self.K)      for r in self.buf], dtype=np.int32) 
        d_margin = np.array([r.get("dist_margin", [np.nan]*self.K) for r in self.buf], dtype=np.float32)
        with h5py.File(h5file, mode) as f:
            if traj_name in f:
                del f[traj_name]
            g = f.create_group(traj_name)
            g.create_dataset("t", data=t)
            g.create_dataset("state", data=st)
            g.create_dataset("ctrl", data=ct)
            g.create_dataset("env", data=env)
            g.create_dataset("label_ref", data=ref)
            g.create_dataset("obs_xy", data=obs)
            g.create_dataset("obs_id", data=ids)  
            g.attrs["R_EGO"] = float(R_EGO)
            g.create_dataset("dist_margin", data=d_margin)

# ===============================================
# ===== ✨ 主循环 (已更新为 Acados) ✨ =====
# ===============================================
def run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH):
    
    print("Initializing Acados MPC solver (may take a moment for C-compilation)...")
    start_build_time = time.time()
    mpc = NMPC_Acados(model_name="nmpc_acados_solver")
    build_time = time.time() - start_build_time
    print(f"Solver initialization complete. Build time: {build_time:.3f} s")
    
    solve_times = []

    for ep in range(num_eps):
        # ✨ 状态扩充为 5D: [x, y, psi, v, delta_prev]
        x_init_4d = np.array([0.0,
                             np.random.uniform(-0.2, 0.2),
                             np.random.uniform(-0.05, 0.05),
                             np.random.uniform(V_MIN, V_MAX)], dtype=np.float64)
        x0 = np.append(x_init_4d, 0.0) # 初始 delta_prev = 0
        
        v_ref = np.random.uniform(0.2, V_MAX)
        obstacles = gen_static_obstacles(M=2, x_min=2.0, x_max=10.0)
        
        # ✨ 初始化 X 和 U 的猜测 (用于 warm start)
        X_prev = np.zeros((mpc.nx, mpc.N + 1))
        X_prev[0, :] = np.linspace(x0[0], x0[0] + v_ref * mpc.N * mpc.DT, mpc.N + 1) # 简单 x 猜测
        X_prev[1, :] = x0[1] # y
        X_prev[2, :] = x0[2] # psi
        X_prev[3, :] = x0[3] # v
        X_prev[4, :] = x0[4] # delta_prev
        U_prev = np.zeros((mpc.nu, mpc.N))

        logger = EpisodeLogger(K_obs=K_NEAR)
        st = x0.copy(); t = 0.0

        for k in range(steps_per_ep):
            #near = nearest_k_obstacles(st[0], st[1], K=K_NEAR)
            near = nearest_k_obstacles(obstacles, st[0], st[1], K=K_NEAR)
            
            obs_mat = np.zeros((2*K_NEAR, N), dtype=np.float64)
            obs_r_mat = np.zeros((K_NEAR, N), dtype=np.float64)      
            for j in range(K_NEAR):
                o = near[j]
                ox, oy, vx, vy = o["x"], o["y"], o["vx"], o["vy"]
                rj = float(o.get("r", 0.0))  
                for h in range(N):
                    obs_mat[2*j,   h] = ox + vx*(h*DT)
                    obs_mat[2*j+1, h] = oy + vy*(h*DT)
                    obs_r_mat[j,   h] = rj 

            # --- 求解 ---
            t_start_solve = time.time()
            sol = mpc.solve(st, v_ref, obs_mat, obs_r_mat, X_prev, U_prev)
            t_end_solve = time.time()
            solve_times.append(t_end_solve - t_start_solve)
            
            U_opt = sol["U"]
            X_opt_aug = sol["X_aug"] # 5D 状态轨迹

            if not sol["success"]:
                print(f"[Warning] Solver failed at step {k}, using previous control.")
                U_opt = U_prev # 复用上一步的控制

            # (其余逻辑保持不变)
            a, delta = float(U_opt[0,0]), float(U_opt[1,0])
            
            # ✨ 使用 4D 状态进行仿真
            st_4d = st[0:4]
            st_ca = ca.DM(st_4d.reshape(4,1))
            u_ca  = ca.DM(np.array([[a],[delta]], dtype=np.float64))
            st1_4d = rk4(st_ca, u_ca, DT).full().flatten()

            # ✨ 构造 5D 的下一个状态
            st1 = np.array([
                st1_4d[0], st1_4d[1], st1_4d[2], st1_4d[3], 
                delta  # 当前施加的 delta 成为下一个 delta_prev
            ])

            d_left  = Y_MAX - st[1]
            d_right = st[1] - Y_MIN
            ey, epsi = st[1], st[2]
            label_ref = (float(X_opt_aug[0,1]), float(X_opt_aug[1,1]))
            
            # ✨ 日志记录 4D 状态
            logger.step(
                t=t,
                state={"x":st[0], "y":st[1], "psi":st[2], "v":st[3], "ey":ey, "epsi":epsi},
                ctrl={"a":a, "delta":delta},
                obstacles_near=near,
                label_ref=label_ref,
                boundaries={"d_left":d_left, "d_right":d_right},
            )

            st = st1; t += DT
            propagate_obstacles(obstacles, DT)

            if k % 10 == 0:
                print(f"[Ep {ep}, Step {k}] t={t:.1f}s, x={st[0]:.2f}, y={st[1]:.2f}, "
                      f"v={st[3]:.2f}, a={a:.2f}, delta={delta:.3f}, "
                      f"SolveTime: {solve_times[-1]*1000:.2f} ms")

            # ✨ 更新 X 和 U 的猜测 (Warm Start Shifting)
            X_shift = np.zeros_like(X_opt_aug)
            X_shift[:, :-1] = X_opt_aug[:, 1:]
            X_shift[:, -1]  = X_opt_aug[:, -1] # 简单重复最后一个
            X_prev = X_shift

            U_shift = np.zeros_like(U_opt)
            U_shift[:, :-1] = U_opt[:, 1:]
            U_shift[:, -1]  = U_opt[:, -1] # 简单重复最后一个
            U_prev = U_shift

        traj_name = f"traj_ep{ep:03d}"
        mode = "w" if ep == 0 else "a"
        logger.to_h5(out_path, traj_name, mode=mode)
        print(f"[EP {ep}] saved -> {out_path}:{traj_name}, steps={steps_per_ep}")

    print(f"\nDone. File: {out_path}")
    
    # 打印求解时间统计
    if solve_times:
        print("\n--- Solve Time Statistics ---")
        if len(solve_times) > 1:
            print(f"First solve (RTI compilation): {solve_times[0]*1000:.2f} ms")
            avg_time = np.mean(solve_times[1:])
            max_time = np.max(solve_times[1:])
            min_time = np.min(solve_times[1:])
            print(f"Avg (excl. 1st): {avg_time*1000:.2f} ms")
            print(f"Max (excl. 1st): {max_time*1000:.2f} ms")
            print(f"Min (excl. 1st): {min_time*1000:.2f} ms")
        else:
            avg_time = np.mean(solve_times)
            print(f"Avg solve time: {avg_time*1000:.2f} ms")


if __name__ == "__main__":
    run_episodes(num_eps=2, steps_per_ep=80, out_path=OUT_PATH)