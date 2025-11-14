#!/usr/bin/env python3
import os, math
import numpy as np
import casadi as ca
import h5py
import subprocess
import time

# implementation Acados
from acados_template import get_tera
get_tera(tera_version='0.0.34', force_download=True)

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# vehicle dynamic model
L = 0.6 #wheel base
R_EGO = 0.5 #radius
DT = 0.1 #integral step length ,the main MPC hyperparameter 1 sampling time
N  = 20  #prediction horizon, the main MPC Hyperparameter 2 prediction of the time horizon steps
#predicted time interval lenght = 20*0.1=2s
K_NEAR = 2 #consider nearest 2 obstacles the mean MPC Hyperparameter 3 considered nearest obstacles
D_SAFE = 0.1

Y_MIN, Y_MAX = -2, 2 #vertical position limit
V_MIN, V_MAX = 0.0, 3.0 #v range, original 0.0 m/s
A_MIN, A_MAX = -3.0, 2.0 #acc range
DELTA_MAX  = np.deg2rad(30) #max yaw angle
DELTA_RATE = np.deg2rad(60) #max yaw rate

OVX_MIN, OVX_MAX = -1.0,1.0 #obstale form and velocity limit
OVY_MIN, OVY_MAX = -0.2,0.2
OR_MIN, OR_MAX = 0.1,0.5

# cost function weights - 使用与 CasADi 版本相同的权重
W_EY   = 6.0   # y lateral error
W_EPSI = 6.0   # psi heading error
W_V    = 2.0   # deviation from v_ref
W_U    = 1e-2  # control magnitude penalty
W_DU   = 1e-2  # control rate penalty
RHO_SLACK = 1e3  # soft constraints penalty (与 CasADi 版本一致)

OUT_PATH = "mpc_dataset.h5"
np.random.seed(None)

# ===== Fahrzeugdynamikmodell (f_cont, same) =====
def f_cont(st, u):
    # st = [x, y, psi, v]
    x, y, psi, v = st[0], st[1], st[2], st[3]
    a, delt      = u[0], u[1]
    dx   = v*ca.cos(psi)
    dy   = v*ca.sin(psi)
    dpsi = (v/L)*ca.tan(delt)
    dv   = a
    return ca.vertcat(dx, dy, dpsi, dv)

# ===== rk4 (same used in main loop simulation) =====
def rk4(st, u, dt=DT):
    k1 = f_cont(st,u)
    k2 = f_cont(st + dt/2*k1, u)
    k3 = f_cont(st + dt/2*k2, u)
    k4 = f_cont(st + dt  *k3, u)
    return st + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ===== NMPC Acados solver with EXPLICIT SLACK =====

class NMPC_Acados_ExplicitSlack:
    def __init__(self, model_name="nmpc_acados_explicit_slack"):
        self.N = N
        self.K = K_NEAR
        self.DT = DT
        self.model_name = model_name

        self.solver = self._build_solver()

        # dimension informations for main loop usage
        self.nx = self.solver.acados_ocp.dims.nx
        self.nu = self.solver.acados_ocp.dims.nu
        self.np = self.solver.acados_ocp.dims.np

    def _build_solver(self):
        print(f"Building Acados NLP with EXPLICIT SLACK (N={self.N}, K={self.K})...")
        ocp = AcadosOcp()
        ocp.model = AcadosModel()

        # 1. 定义维度：关键修改 - 将松弛变量 S 作为控制输入
        nx = 5  # [x, y, psi, v, delta_prev]
        nu = 2 + self.K  # [a, delta, S_0, S_1, ..., S_{K-1}] - 显式松弛变量
        np_k = 1 + 3 * self.K + nu  # [v_ref, obs_x, obs_y, obs_r] * K + u_prev
        nh = 1 + self.K  # delta_rate 约束 + K 个障碍物约束

        ocp.dims.N = self.N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.np = np_k

        # 约束维度（没有软约束，全部是硬约束）
        ocp.dims.nh_0 = nh
        ocp.dims.nh = nh
        ocp.dims.nh_e = 0

        # 2. 定义符号变量
        x_aug_sym = ca.SX.sym('x_aug', nx)
        u_sym = ca.SX.sym('u', nu)  # [a, delta, S_0, S_1]
        p_sym = ca.SX.sym('p', np_k)

        # 3. 定义动力学（只涉及 a 和 delta）
        st_sym = x_aug_sym[0:4]  # [x, y, psi, v]
        delta_prev_sym = x_aug_sym[4]
        a_sym = u_sym[0]
        delta_sym = u_sym[1]
        u_vehicle = ca.vertcat(a_sym, delta_sym)

        f_cont_expr = f_cont(st_sym, u_vehicle)
        ocp.model.f_expl_expr = ca.vertcat(f_cont_expr, delta_sym)  # 增广状态
        ocp.model.x = x_aug_sym
        ocp.model.u = u_sym
        ocp.model.p = p_sym
        ocp.model.name = self.model_name

        ocp.parameter_values = np.zeros(np_k)

        # 4. 定义代价函数 - 完全模仿 CasADi 版本
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        v_ref_sym = p_sym[0]
        u_prev_sym = p_sym[1 + 3*self.K : 1 + 3*self.K + nu]  # U_PREV 参数

        # 初始阶段 (k=0) - 只有状态项
        ocp.model.cost_y_expr_0 = ca.vertcat(
            x_aug_sym[1],                 # ey
            x_aug_sym[2],                 # epsi
            x_aug_sym[3] - v_ref_sym      # v_err
        )
        ocp.cost.W_0 = np.diag([W_EY, W_EPSI, W_V])
        ocp.dims.ny_0 = ocp.model.cost_y_expr_0.shape[0]
        ocp.cost.yref_0 = np.zeros(ocp.dims.ny_0)

        # 中间阶段 (k=1...N-1) - 状态 + 控制 + 控制增量 + 松弛变量惩罚
        # 关键：手动添加松弛变量到代价函数，并添加控制增量惩罚
        slack_terms = []
        for j in range(self.K):
            slack_terms.append(u_sym[2 + j])  # S_j

        # 控制增量：du = u - u_prev（完全模仿 CasADi 版本）
        du_sym = u_sym - u_prev_sym

        ocp.model.cost_y_expr = ca.vertcat(
            x_aug_sym[1],                 # ey
            x_aug_sym[2],                 # epsi
            x_aug_sym[3] - v_ref_sym,     # v_err
            u_sym[0],                     # a
            u_sym[1],                     # delta
            du_sym[0],                    # da (控制增量 - 加速度)
            du_sym[1],                    # d_delta (控制增量 - 转向)
            *slack_terms                  # S_0, S_1, ...（显式松弛变量）
        )

        # 权重矩阵：状态权重 + 控制权重 + 控制增量权重 + 松弛变量权重
        W_weights = [W_EY, W_EPSI, W_V, W_U, W_U, W_DU, W_DU] + [RHO_SLACK] * self.K
        ocp.cost.W = np.diag(W_weights)
        ocp.dims.ny = ocp.model.cost_y_expr.shape[0]
        ocp.cost.yref = np.zeros(ocp.dims.ny)

        # 终端阶段 (k=N) - 只有状态
        ocp.model.cost_y_expr_e = ca.vertcat(
            x_aug_sym[1],
            x_aug_sym[2],
            x_aug_sym[3] - v_ref_sym
        )
        ocp.cost.W_e = np.diag([W_EY, W_EPSI, W_V])
        ocp.dims.ny_e = ocp.model.cost_y_expr_e.shape[0]
        ocp.cost.yref_e = np.zeros(ocp.dims.ny_e)

        # 5. 定义约束

        # A. 状态边界
        ocp.constraints.idxbx_0 = np.array([1, 3])
        ocp.constraints.lbx_0 = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx_0 = np.array([Y_MAX, V_MAX])

        ocp.constraints.idxbx = np.array([1, 3])
        ocp.constraints.lbx = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx = np.array([Y_MAX, V_MAX])

        ocp.constraints.idxbx_e = np.array([1, 3])
        ocp.constraints.lbx_e = np.array([Y_MIN, V_MIN])
        ocp.constraints.ubx_e = np.array([Y_MAX, V_MAX])

        # B. 控制边界：a, delta, S_0, S_1, ...
        ocp.constraints.idxbu = np.array(list(range(nu)))
        lbu_list = [A_MIN, -DELTA_MAX] + [0.0] * self.K  # S >= 0
        ubu_list = [A_MAX, DELTA_MAX] + [1e6] * self.K   # S 上界设为大值
        ocp.constraints.lbu = np.array(lbu_list)
        ocp.constraints.ubu = np.array(ubu_list)

        # C. 非线性约束 - 完全模仿 CasADi 版本
        h_expr_list = []

        # (1) delta_rate 约束
        h_delta_rate = u_sym[1] - delta_prev_sym
        h_expr_list.append(h_delta_rate)

        # (2) 障碍物约束：g_obs <= S[j]  =>  g_obs - S[j] <= 0
        obs_param_offset = 1  # v_ref 占 1 个位置
        for j in range(self.K):
            obs_x_sym = p_sym[obs_param_offset + j*3 + 0]
            obs_y_sym = p_sym[obs_param_offset + j*3 + 1]
            obs_r_sym = p_sym[obs_param_offset + j*3 + 2]
            dx = x_aug_sym[0] - obs_x_sym
            dy = x_aug_sym[1] - obs_y_sym
            rj = obs_r_sym

            # 关键：与 CasADi 版本完全一致的约束形式
            g_obs = (D_SAFE + R_EGO + rj)**2 - (dx*dx + dy*dy)
            S_j = u_sym[2 + j]  # 显式松弛变量
            h_obs_j = g_obs - S_j  # g_obs <= S_j
            h_expr_list.append(h_obs_j)

        ocp.model.con_h_expr = ca.vertcat(*h_expr_list)
        ocp.model.con_h_expr_0 = ocp.model.con_h_expr

        # D. 约束边界（硬约束，不使用软约束）
        lh_list = [-DELTA_RATE * DT] + [-1e8] * self.K  # g_obs - S <= 0
        uh_list = [DELTA_RATE * DT] + [0.0] * self.K
        ocp.constraints.lh_0 = np.array(lh_list, dtype=np.float64)
        ocp.constraints.uh_0 = np.array(uh_list, dtype=np.float64)
        ocp.constraints.lh = np.array(lh_list, dtype=np.float64)
        ocp.constraints.uh = np.array(uh_list, dtype=np.float64)

        # E. 不使用软约束（移除 idxsh, Zl, Zu 等配置）

        # --- 6. 求解器设置 ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.sim_method_num_steps = 4
        ocp.solver_options.tf = self.N * self.DT

        print("\n--- Acados EXPLICIT SLACK Sanity Check ---")
        print(f"K = {self.K}")
        print(f"nu (control dim) = {nu} ([a, delta] + {self.K} slack vars)")
        print(f"np (parameter dim) = {np_k} (v_ref + 3*K obs + {nu} u_prev)")
        print(f"Declared nh_0 (dims): {ocp.dims.nh_0}")
        print(f"Expression h (model) len: {ocp.model.con_h_expr.shape[0]}")
        print(f"Cost ny (middle stage): {ocp.dims.ny} (3 state + 2 ctrl + 2 du + {self.K} slack)")
        print(f"RHO_SLACK = {RHO_SLACK}, W_DU = {W_DU}")
        print("------------------------------------------\n")

        # --- 7. 创建求解器 ---
        json_file = self.model_name + '.json'
        solver = AcadosOcpSolver(ocp, json_file=json_file, build=True, generate=True)

        print("Acados solver build complete.")
        return solver

    def solve(self, x0_aug, v_ref, obs_xy_2d, obs_r_2d, X_guess, U_guess, U_prev):

        # 1. 设置初始状态
        self.solver.set(0, 'x', x0_aug)

        # 2. 设置时变参数 p（包含 U_PREV）
        p_matrix = np.zeros((self.N, self.np))
        p_matrix[:, 0] = v_ref

        # 障碍物参数
        for j in range(self.K):
            p_matrix[:, 1 + j*3 + 0] = obs_xy_2d[j*2, :]     # obs_x
            p_matrix[:, 1 + j*3 + 1] = obs_xy_2d[j*2 + 1, :] # obs_y
            p_matrix[:, 1 + j*3 + 2] = obs_r_2d[j, :]        # obs_r

        # U_PREV 参数（关键：使传入的 U_prev 成为参数）
        u_prev_offset = 1 + 3 * self.K
        for k in range(self.N):
            p_matrix[k, u_prev_offset:u_prev_offset + self.nu] = U_prev[:, k]

        for k in range(self.N):
            self.solver.set(k, 'p', p_matrix[k, :])

        # 3. 设置 Warm Start
        for k in range(self.N):
            self.solver.set(k, 'x', X_guess[:, k])
            self.solver.set(k, 'u', U_guess[:, k])  # 包含松弛变量
        self.solver.set(self.N, 'x', X_guess[:, self.N])

        # 4. 设置参考轨迹 yref（与原版一致）
        y_ref_current = x0_aug[1]
        threshold = 0.05

        if abs(y_ref_current) < threshold:
            yref_0 = np.array([0.0, 0.0, 0.0])
            self.solver.set(0, 'yref', yref_0)

            for k in range(1, self.N):
                yref_k = np.zeros(self.nu + 3)  # 包含松弛变量维度
                self.solver.set(k, 'yref', yref_k)

            yref_e = np.array([0.0, 0.0, 0.0])
            self.solver.set(self.N, 'yref', yref_e)
        else:
            if abs(y_ref_current) > 0.5:
                decay_rate = 0.85
            elif abs(y_ref_current) > 0.2:
                decay_rate = 0.90
            else:
                decay_rate = 0.95

            y_ref_0 = y_ref_current * decay_rate
            yref_0 = np.array([y_ref_0, 0.0, 0.0])
            self.solver.set(0, 'yref', yref_0)

            for k in range(1, self.N):
                y_ref_k = y_ref_current * (decay_rate ** k)
                yref_k = np.zeros(self.nu + 3)
                yref_k[0] = y_ref_k  # ey
                self.solver.set(k, 'yref', yref_k)

            yref_e = np.array([0.0, 0.0, 0.0])
            self.solver.set(self.N, 'yref', yref_e)

        # 5. 求解
        status = self.solver.solve()

        if status != 0:
            print(f"Acados solver failed with status {status}")

        # 6. 获取结果
        U_opt = np.array([self.solver.get(k, 'u') for k in range(self.N)]).T
        X_opt_aug = np.array([self.solver.get(k, 'x') for k in range(self.N + 1)]).T

        return {
            "success": status == 0,
            "U": U_opt,         # [nu, N] - 包含松弛变量
            "X_aug": X_opt_aug  # [nx, N+1] (5D 状态)
        }

# ===== random obstacles and logs =====
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

# =====  Main Loop -- Acados with Explicit Slack =====

def run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH):

    print("Initializing Acados MPC solver with EXPLICIT SLACK...")
    start_build_time = time.time()
    mpc = NMPC_Acados_ExplicitSlack(model_name="nmpc_acados_explicit_slack")
    build_time = time.time() - start_build_time
    print(f"Solver initialization complete. Build time: {build_time:.3f} s")

    solve_times = []

    for ep in range(num_eps):
        x_init_4d = np.array([0.0,
                             np.random.uniform(-0.2, 0.2),
                             np.random.uniform(-0.05, 0.05),
                             np.random.uniform(V_MIN, V_MAX)], dtype=np.float64)
        x0 = np.append(x_init_4d, 0.0) # 初始 delta_prev = 0

        v_ref = np.random.uniform(0.2, V_MAX)
        obstacles = gen_static_obstacles(M=2, x_min=2.0, x_max=10.0)

        # 初始化 X 和 U 的猜测（U 现在包含松弛变量）
        X_prev = np.zeros((mpc.nx, mpc.N + 1))
        X_prev[0, :] = np.linspace(x0[0], x0[0] + v_ref * mpc.N * mpc.DT, mpc.N + 1)
        X_prev[1, :] = x0[1]
        X_prev[2, :] = x0[2]
        X_prev[3, :] = x0[3]
        X_prev[4, :] = x0[4]
        U_prev = np.zeros((mpc.nu, mpc.N))  # nu = 2 + K

        logger = EpisodeLogger(K_obs=K_NEAR)
        st = x0.copy(); t = 0.0

        print(f"\n=== Episode {ep} Initialization ===")
        print(f"x0_aug: {x0}")

        for k in range(steps_per_ep):
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

            # --- solve ---
            t_start_solve = time.time()
            sol = mpc.solve(st, v_ref, obs_mat, obs_r_mat, X_prev, U_prev, U_prev)
            t_end_solve = time.time()
            solve_times.append(t_end_solve - t_start_solve)

            U_opt = sol["U"]  # [nu, N] - 包含 [a, delta, S_0, S_1]
            X_opt_aug = sol["X_aug"]

            if not sol["success"]:
                print(f"[Warning] Solver failed at step {k}, emergency brake")
                U_opt = U_prev.copy()
                U_opt[0, :] = -2.0
                U_opt[1, :] = 0.0

            # 提取实际控制输入（a 和 delta）
            a, delta = float(U_opt[0,0]), float(U_opt[1,0])

            # 提取松弛变量用于诊断
            slack_vals = U_opt[2:, 0]

            # 诊断输出
            if k % 10 == 0:
                print(f"\n--- Step {k} Debug ---")
                print(f"State: x={st[0]:.2f}, y={st[1]:.2f}, psi={st[2]:.4f}, v={st[3]:.2f}")
                print(f"Control: a={a:.3f}, delta={delta:.4f} ({np.rad2deg(delta):.2f}°)")
                print(f"Slack vars: {slack_vals}")

                for j, o in enumerate(near[:K_NEAR]):
                    dx, dy = o['x']-st[0], o['y']-st[1]
                    dist = math.hypot(dx, dy)
                    safety_dist = R_EGO + D_SAFE + o['r']
                    print(f"  Obs {j}: dist={dist:.2f}, safety={safety_dist:.2f}, margin={dist-safety_dist:.2f}, S={slack_vals[j]:.4f}")

            # 使用 4D 状态进行仿真
            st_4d = st[0:4]
            st_ca = ca.DM(st_4d.reshape(4,1))
            u_ca  = ca.DM(np.array([[a],[delta]], dtype=np.float64))
            st1_4d = rk4(st_ca, u_ca, DT).full().flatten()

            st1 = np.array([
                st1_4d[0], st1_4d[1], st1_4d[2], st1_4d[3],
                delta
            ])

            d_left  = Y_MAX - st[1]
            d_right = st[1] - Y_MIN
            ey, epsi = st[1], st[2]
            label_ref = (float(X_opt_aug[0,1]), float(X_opt_aug[1,1]))

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

            # Warm Start Shifting
            X_shift = np.zeros_like(X_opt_aug)
            X_shift[:, :-1] = X_opt_aug[:, 1:]
            X_shift[:, -1]  = X_opt_aug[:, -1]
            X_prev = X_shift

            U_shift = np.zeros_like(U_opt)
            U_shift[:, :-1] = U_opt[:, 1:]
            U_shift[:, -1]  = U_opt[:, -1]
            U_prev = U_shift

        traj_name = f"traj_ep{ep:03d}"
        mode = "w" if ep == 0 else "a"
        logger.to_h5(out_path, traj_name, mode=mode)
        print(f"[EP {ep}] saved -> {out_path}:{traj_name}, steps={steps_per_ep}")

    print(f"\nDone. File: {out_path}")

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
    run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH)
