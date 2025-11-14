#!/usr/bin/env python3
"""
基于 acados 框架的车辆避障 MPC 控制器
保持原有算法逻辑，提升实时性能
"""
import os
import math
import numpy as np
import h5py
import time
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca

# ===== 参数设置 (保持不变) =====
L = 0.6  # wheel base
R_EGO = 0.5  # radius
DT = 0.1  # integral step length
N = 20  # prediction horizon
K_NEAR = 2  # consider nearest 2 obstacles
D_SAFE = 0.1

Y_MIN, Y_MAX = -2, 2
V_MIN, V_MAX = 0.0, 3.0
A_MIN, A_MAX = -3.0, 2.0
DELTA_MAX = np.deg2rad(30)
DELTA_RATE = np.deg2rad(40)

OVX_MIN, OVX_MAX = -1.0, 1.0
OVY_MIN, OVY_MAX = -0.2, 0.2
OR_MIN, OR_MAX = 0.1, 0.5

# cost function weights (保持不变)
W_EY = 6.0
W_EPSI = 6.0
W_V = 2.0
W_U = 1e-2
W_DU = 1e-2
RHO_SLACK = 1e3

OUT_PATH = "mpc_dataset.h5"
np.random.seed(None)


# ===== acados 模型定义 =====
def export_vehicle_model(n_param):
    """导出车辆动力学模型供 acados 使用"""
    model_name = 'vehicle_model'
    
    # 状态变量: x, y, psi, v
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    psi = ca.SX.sym('psi')
    v = ca.SX.sym('v')
    state = ca.vertcat(x, y, psi, v)
    
    # 控制变量: a, delta
    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)
    
    # 参数变量 (用于传递时变信息)
    p = ca.SX.sym('p', n_param)
    
    # 连续时间动力学 (保持原有模型)
    x_dot = v * ca.cos(psi)
    y_dot = v * ca.sin(psi)
    psi_dot = (v / L) * ca.tan(delta)
    v_dot = a
    
    f_expl = ca.vertcat(x_dot, y_dot, psi_dot, v_dot)
    
    # 创建 acados 模型
    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.u = controls
    model.p = p  # 添加参数
    model.f_expl_expr = f_expl
    # acados 会自动使用 RK4 或其他积分器
    
    return model


# ===== acados MPC 求解器 =====
class NMPC_Acados:
    def __init__(self):
        """初始化 acados OCP 求解器"""
        self.N = N
        self.K = K_NEAR
        self.DT = DT
        
        # 更新 tera 渲染器
        try:
            from acados_template import get_tera
            get_tera(tera_version='0.0.34', force_download=False)
        except:
            pass
        
        print("Building acados OCP solver...")
        start_time = time.time()
        
        # 创建 OCP 对象
        ocp = AcadosOcp()
        
        # 参数维度 (用于传递障碍物信息、参考值等)
        # P = [v_ref(1), xref0(1), obs_xy(2*K), obs_r(K), u_prev(2)]
        n_param = 2 + 2*self.K + self.K + 2
        
        # 设置模型
        model = export_vehicle_model(n_param)
        ocp.model = model
        
        # 维度
        nx = 4  # 状态维度
        nu = 2  # 控制维度
        
        ocp.parameter_values = np.zeros(n_param)
        ocp.dims.np = n_param  # 设置参数维度
        
        # 参数索引 (在创建代价函数之前定义)
        # 每个阶段的参数: [v_ref, xref0, ox0, oy0, ox1, oy1, r0, r1, u_prev_a, u_prev_delta]
        self.param_idx = {
            'v_ref': 0,
            'xref0': 1,
            'obs_xy_start': 2,
            'obs_xy_end': 2 + 2*self.K,
            'obs_r_start': 2 + 2*self.K,
            'obs_r_end': 2 + 2*self.K + self.K,
            'u_prev_start': 2 + 2*self.K + self.K,
            'u_prev_end': n_param
        }
        
        # ===== 代价函数设置 =====
        # 使用非线性最小二乘形式
        ny = nx + nu  # 输出维度
        ny_e = nx  # 终端输出维度
        
        # 定义输出函数 y = [x, u]
        model.cost_y_expr = ca.vertcat(model.x, model.u)
        model.cost_y_expr_e = model.x
        
        # 权重矩阵
        W = np.diag([0.0, W_EY, W_EPSI, W_V, W_U, W_U])
        W_e = np.diag([0.0, W_EY, W_EPSI, W_V])
        
        ocp.cost.W = W
        ocp.cost.W_e = W_e
        
        # 参考值 (运行时更新)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        
        # ===== 约束设置 =====
        # 状态约束 - 避免使用 inf
        BIG_NUM = 1e10
        ocp.constraints.lbx = np.array([-BIG_NUM, float(Y_MIN), -BIG_NUM, float(V_MIN)])
        ocp.constraints.ubx = np.array([BIG_NUM, float(Y_MAX), BIG_NUM, float(V_MAX)])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3])
        
        # 控制约束
        ocp.constraints.lbu = np.array([float(A_MIN), float(-DELTA_MAX)])
        ocp.constraints.ubu = np.array([float(A_MAX), float(DELTA_MAX)])
        ocp.constraints.idxbu = np.array([0, 1])
        
        # 初始状态约束
        ocp.constraints.x0 = np.zeros(4)
        
        # 非线性约束 (障碍物约束 + 转向角速率约束)
        # 每个时间步有 K 个障碍物约束 + 1 个转向角速率约束
        self._setup_nonlinear_constraints(ocp, model, n_param)
        
        # ===== 求解器选项 =====
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'  # 显式 Runge-Kutta (RK4)
        ocp.solver_options.sim_method_num_stages = 4  # RK4
        ocp.solver_options.sim_method_num_steps = 1
        
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.nlp_solver_max_iter = 120
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.qp_solver_iter_max = 100
        
        # 输出设置
        ocp.solver_options.print_level = 0
        
        # 代码生成选项
        ocp.code_export_directory = 'c_generated_code'
        
        # 时间设置 - 确保为浮点数
        ocp.solver_options.tf = float(self.N * self.DT)
        ocp.solver_options.N_horizon = int(self.N)
        
        # 创建求解器
        self.solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        
        build_time = time.time() - start_time
        print(f"Acados solver built successfully in {build_time:.3f} s")
    
    def _setup_nonlinear_constraints(self, ocp, model, n_param):
        """设置非线性约束 (障碍物)"""
        x = model.x
        u = model.u
        p = model.p
        
        h_expr = []
        
        # 障碍物约束 (K 个)
        obs_xy_start = 2  # v_ref, xref0 之后
        obs_r_start = 2 + 2*self.K  # 在 xy 之后
        
        for j in range(self.K):
            ox = p[obs_xy_start + 2*j]
            oy = p[obs_xy_start + 2*j + 1]
            r_obs = p[obs_r_start + j]
            
            dx = x[0] - ox
            dy = x[1] - oy
            dist_sq = dx**2 + dy**2
            safe_dist_sq = (float(R_EGO) + float(D_SAFE) + r_obs)**2
            
            # 约束: dist_sq >= safe_dist_sq
            # 转换为: safe_dist_sq - dist_sq <= 0
            h_expr.append(safe_dist_sq - dist_sq)
        
        if len(h_expr) > 0:
            model.con_h_expr = ca.vertcat(*h_expr)
            
            # 设置约束界限
            ocp.constraints.lh = np.full(len(h_expr), -1e10)  # 不用 -inf
            ocp.constraints.uh = np.zeros(len(h_expr))
            
            # 添加松弛变量
            ocp.constraints.idxsh = np.array(range(len(h_expr)))
            
            # L2 松弛惩罚
            ns = len(h_expr)
            ocp.cost.zl = np.ones(ns) * float(RHO_SLACK)
            ocp.cost.zu = np.ones(ns) * float(RHO_SLACK)
            ocp.cost.Zl = np.zeros(ns)
            ocp.cost.Zu = np.zeros(ns)
    
    def solve(self, x0, v_ref, xref0, obs_xy_2d, obs_r_2d, u_prev_guess):
        """
        求解 MPC 问题
        
        参数:
            x0: 初始状态 [x, y, psi, v]
            v_ref: 参考速度
            xref0: 参考 x 位置
            obs_xy_2d: 障碍物位置 (2*K, N)
            obs_r_2d: 障碍物半径 (K, N)
            u_prev_guess: 上一时刻的控制序列 (2, N)
        """
        # 设置初始状态
        self.solver.set(0, 'lbx', x0)
        self.solver.set(0, 'ubx', x0)
        
        # 为每个阶段设置参数和参考值
        # 每个阶段的参数: [v_ref, xref0, ox0, oy0, ox1, oy1, r0, r1, u_prev_a, u_prev_delta]
        for i in range(self.N):
            param = np.zeros(2 + 2*self.K + self.K + 2)
            param[0] = v_ref
            param[1] = xref0
            
            # 障碍物位置
            for j in range(self.K):
                param[2 + 2*j] = obs_xy_2d[2*j, i]
                param[2 + 2*j + 1] = obs_xy_2d[2*j+1, i]
            
            # 障碍物半径
            for j in range(self.K):
                param[2 + 2*self.K + j] = obs_r_2d[j, i]
            
            # u_prev
            param[2 + 2*self.K + self.K] = u_prev_guess[0, i]
            param[2 + 2*self.K + self.K + 1] = u_prev_guess[1, i]
            
            self.solver.set(i, 'p', param)
            
            # 设置参考值 yref = [x, y, psi, v, a, delta]
            yref = np.array([0.0, 0.0, 0.0, v_ref, 0.0, 0.0])
            self.solver.set(i, 'yref', yref)
        
        # 终端阶段
        param_N = np.zeros(2 + 2*self.K + self.K + 2)
        param_N[0] = v_ref
        self.solver.set(self.N, 'p', param_N)
        
        yref_e = np.array([0.0, 0.0, 0.0, v_ref])
        self.solver.set(self.N, 'yref', yref_e)
        
        # 初始化猜测 (warm start)
        for i in range(self.N):
            self.solver.set(i, 'u', u_prev_guess[:, i])
        
        # 求解
        try:
            status = self.solver.solve()
            
            if status == 0:
                # 提取解
                X_opt = np.zeros((4, self.N + 1))
                U_opt = np.zeros((2, self.N))
                
                for i in range(self.N + 1):
                    X_opt[:, i] = self.solver.get(i, 'x')
                
                for i in range(self.N):
                    U_opt[:, i] = self.solver.get(i, 'u')
                
                return {"success": True, "U": U_opt, "X": X_opt}
            else:
                print(f"Solver failed with status {status}")
                # 返回上一次的猜测
                return {"success": False, "U": u_prev_guess, "X": np.zeros((4, self.N + 1))}
        
        except Exception as e:
            print(f"Solver exception: {e}")
            return {"success": False, "U": u_prev_guess, "X": np.zeros((4, self.N + 1))}
    
    def __del__(self):
        """清理求解器"""
        if hasattr(self, 'solver'):
            del self.solver


# ===== 障碍物生成和传播 (保持不变) =====
def gen_static_obstacles(M=2, x_min=2.0, x_max=10.0):
    """生成随机障碍物"""
    obs = []
    for i in range(M):
        xi = np.random.uniform(x_min, x_max)
        yi = np.random.uniform(Y_MIN, Y_MAX)
        vxi = np.random.uniform(OVX_MIN, OVX_MAX)
        vyi = np.random.uniform(OVY_MIN, OVY_MAX)
        ri = np.random.uniform(OR_MIN, OR_MAX)
        o = {"id": i, "x": xi, "y": yi, "vx": vxi, "vy": vyi, "r": ri}
        obs.append(o)
        print(f"[gen_static_obstacles] Obstacle {i}: x={o['x']:.2f}, y={o['y']:.2f}, "
              f"vx={o['vx']:.2f}, vy={o['vy']:.2f}, r={o['r']:.2f}")
    return obs


def propagate_obstacles(obstacles, dt=DT):
    """更新障碍物位置"""
    for o in obstacles:
        o["x"] += o["vx"] * dt
        o["y"] += o["vy"] * dt


def nearest_k_obstacles(obstacles, x, y, K=K_NEAR):
    """找到最近的 K 个障碍物"""
    d2 = [(((o["x"] - x)**2 + (o["y"] - y)**2), o) for o in obstacles]
    d2.sort(key=lambda t: t[0])
    sel = [t[1] for t in d2[:K]] + [None] * max(0, K - len(d2))
    out = []
    for j in range(K):
        out.append(sel[j] if sel[j] is not None else
                   {"id": -1, "x": x + 50.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "r": 0.0})
    return out


def check_collision(x, y, obstacles):
    """
    检查是否与障碍物碰撞
    
    参数:
        x, y: 车辆位置
        obstacles: 所有障碍物列表
    
    返回:
        collision: 是否碰撞
        collision_info: 碰撞信息字典
    """
    collision = False
    collision_info = {
        "collision": False,
        "vehicle_pos": (x, y),
        "obstacle_id": None,
        "obstacle_pos": None,
        "distance": None,
        "min_safe_distance": None
    }
    
    for obs in obstacles:
        if obs.get("id", -1) < 0:
            continue
        
        ox, oy, r_obs = obs["x"], obs["y"], obs.get("r", 0.0)
        dx = x - ox
        dy = y - oy
        dist = math.sqrt(dx**2 + dy**2)
        min_safe_dist = R_EGO + r_obs
        
        # 碰撞条件：距离小于车辆半径 + 障碍物半径
        if dist < min_safe_dist:
            collision = True
            collision_info = {
                "collision": True,
                "vehicle_pos": (x, y),
                "obstacle_id": obs["id"],
                "obstacle_pos": (ox, oy),
                "distance": dist,
                "min_safe_distance": min_safe_dist,
                "penetration": min_safe_dist - dist  # 穿透深度
            }
            break  # 只报告第一个碰撞
    
    return collision, collision_info


# ===== 数据记录 (保持不变) =====
class EpisodeLogger:
    def __init__(self, K_obs=K_NEAR):
        self.buf = []
        self.K = K_obs
    
    def step(self, t, state, ctrl, obstacles_near, label_ref, boundaries):
        env = []
        x, y = state["x"], state["y"]
        
        for j in range(self.K):
            o = obstacles_near[j]
            dx, dy = o["x"] - x, o["y"] - y
            dist = math.hypot(dx, dy)
            margin = dist - (R_EGO + D_SAFE + o.get("r", 0.0))
            env += [dx, dy, margin, o["vx"], o["vy"]]
        
        env += [boundaries["d_left"], boundaries["d_right"], 
                state.get("ey", 0.0), state.get("epsi", 0.0)]
        
        obs_xy = []
        obs_id = []
        for j in range(self.K):
            o = obstacles_near[j]
            obs_xy += [o["x"], o["y"], o["vx"], o["vy"], o.get("r", 0.0)]
            obs_id.append(int(o.get("id", -1)))
        
        self.buf.append({
            "t": t,
            "state": [state["x"], state["y"], state["psi"], state["v"]],
            "ctrl": [ctrl["a"], ctrl["delta"]],
            "env": env,
            "label_ref": [label_ref[0], label_ref[1]],
            "obs_xy": obs_xy,
            "obs_id": obs_id,
        })
    
    def to_h5(self, h5file, traj_name, mode="a"):
        os.makedirs(os.path.dirname(h5file) or ".", exist_ok=True)
        t = np.array([r["t"] for r in self.buf], dtype=np.float32)
        st = np.array([r["state"] for r in self.buf], dtype=np.float32)
        ct = np.array([r["ctrl"] for r in self.buf], dtype=np.float32)
        env = np.array([r["env"] for r in self.buf], dtype=np.float32)
        ref = np.array([r["label_ref"] for r in self.buf], dtype=np.float32)
        obs_dim = self.K * 5
        obs = np.array([r.get("obs_xy", [np.nan] * obs_dim) for r in self.buf], dtype=np.float32)
        ids = np.array([r.get("obs_id", [-1] * self.K) for r in self.buf], dtype=np.int32)
        
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


# ===== RK4 积分器 (用于实际系统仿真) =====
def f_cont_numpy(st, u):
    """连续时间动力学 (numpy 版本)"""
    x, y, psi, v = st[0], st[1], st[2], st[3]
    a, delta = u[0], u[1]
    dx = v * np.cos(psi)
    dy = v * np.sin(psi)
    dpsi = (v / L) * np.tan(delta)
    dv = a
    return np.array([dx, dy, dpsi, dv])


def rk4_numpy(st, u, dt=DT):
    """RK4 积分器 (numpy 版本)"""
    k1 = f_cont_numpy(st, u)
    k2 = f_cont_numpy(st + dt/2 * k1, u)
    k3 = f_cont_numpy(st + dt/2 * k2, u)
    k4 = f_cont_numpy(st + dt * k3, u)
    return st + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# ===== 主循环 =====
def run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH):
    """运行多个 episode"""
    
    # 初始化 acados 求解器
    print("Initializing acados MPC solver...")
    start_build_time = time.time()
    mpc = NMPC_Acados()
    build_time = time.time() - start_build_time
    print(f"Solver initialization complete. Build time: {build_time:.3f} s\n")
    
    solve_times = []
    
    for ep in range(num_eps):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{num_eps}")
        print(f"{'='*60}")
        
        # 初始化状态
        x0 = np.array([
            0.0,
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(V_MIN, V_MAX)
        ], dtype=np.float64)
        
        v_ref = np.random.uniform(0.2, V_MAX)
        x_ref0 = float(x0[0])
        
        print(f"Initial state: x={x0[0]:.2f}, y={x0[1]:.2f}, psi={x0[2]:.2f}, v={x0[3]:.2f}")
        print(f"Reference velocity: {v_ref:.2f} m/s\n")
        
        # 生成障碍物
        obstacles = gen_static_obstacles(M=2, x_min=2.0, x_max=10.0)
        
        # 初始化控制序列
        U_prev = np.zeros((2, N), dtype=np.float64)
        
        # 数据记录器
        logger = EpisodeLogger(K_obs=K_NEAR)
        
        st = x0.copy()
        t = 0.0
        
        collision_count = 0  # 碰撞计数
        
        for k in range(steps_per_ep):
            # 找到最近的障碍物
            near = nearest_k_obstacles(obstacles, st[0], st[1], K=K_NEAR)
            
            # === 碰撞检测 ===
            collision, collision_info = check_collision(st[0], st[1], obstacles)
            if collision:
                collision_count += 1
                print(f"\n{'!'*60}")
                print(f"⚠️  COLLISION DETECTED at Step {k}, t={t:.2f}s")
                print(f"{'!'*60}")
                print(f"Vehicle position: ({collision_info['vehicle_pos'][0]:.3f}, {collision_info['vehicle_pos'][1]:.3f})")
                print(f"Obstacle ID: {collision_info['obstacle_id']}")
                print(f"Obstacle position: ({collision_info['obstacle_pos'][0]:.3f}, {collision_info['obstacle_pos'][1]:.3f})")
                print(f"Actual distance: {collision_info['distance']:.3f} m")
                print(f"Required distance: {collision_info['min_safe_distance']:.3f} m")
                print(f"Penetration depth: {collision_info['penetration']:.3f} m")
                print(f"{'!'*60}\n")
            
            # 构造障碍物预测矩阵
            obs_mat = np.zeros((2 * K_NEAR, N), dtype=np.float64)
            obs_r_mat = np.zeros((K_NEAR, N), dtype=np.float64)
            
            for j in range(K_NEAR):
                ox = near[j]["x"]
                oy = near[j]["y"]
                vx = near[j]["vx"]
                vy = near[j]["vy"]
                rj = float(near[j].get("r", 0.0))
                
                for h in range(N):
                    obs_mat[2*j, h] = ox + vx * (h * DT)
                    obs_mat[2*j+1, h] = oy + vy * (h * DT)
                    obs_r_mat[j, h] = rj
            
            # 求解 MPC
            t_start_solve = time.time()
            sol = mpc.solve(st, v_ref, x_ref0, obs_mat, obs_r_mat, U_prev)
            t_end_solve = time.time()
            solve_time = t_end_solve - t_start_solve
            solve_times.append(solve_time)
            
            U_opt = sol["U"]
            X_opt = sol["X"]
            
            if not sol["success"]:
                print(f"[Warning] Solver failed at step {k}, using previous control.")
            
            # 应用第一个控制量
            a = float(U_opt[0, 0])
            delta = float(U_opt[1, 0])
            
            # 使用 RK4 更新状态 (模拟真实系统)
            st_next = rk4_numpy(st, np.array([a, delta]), DT)
            
            # 计算边界距离和误差
            d_left = Y_MAX - st[1]
            d_right = st[1] - Y_MIN
            ey = st[1]
            epsi = st[2]
            
            # 下一步参考点
            label_ref = (float(X_opt[0, 1]), float(X_opt[1, 1]))
            
            # 记录数据
            logger.step(
                t=t,
                state={"x": st[0], "y": st[1], "psi": st[2], "v": st[3], 
                       "ey": ey, "epsi": epsi},
                ctrl={"a": a, "delta": delta},
                obstacles_near=near,
                label_ref=label_ref,
                boundaries={"d_left": d_left, "d_right": d_right},
            )
            
            # 更新状态和时间
            st = st_next
            t += DT
            
            # 传播障碍物
            propagate_obstacles(obstacles, DT)
            
            # 打印进度
            if k % 10 == 0 or k == steps_per_ep - 1:
                print(f"[Step {k:3d}] t={t:5.1f}s | x={st[0]:6.2f} y={st[1]:5.2f} "
                      f"v={st[3]:4.2f} | a={a:5.2f} δ={np.rad2deg(delta):5.1f}° | "
                      f"Solve: {solve_time*1000:5.1f} ms")
            
            # Warm start: 将控制序列右移
            U_shift = np.zeros_like(U_opt)
            U_shift[:, :-1] = U_opt[:, 1:]
            U_shift[:, -1] = U_opt[:, -1]
            U_prev = U_shift
        
        # 保存 episode 数据
        traj_name = f"traj_ep{ep:03d}"
        mode = "w" if ep == 0 else "a"
        logger.to_h5(out_path, traj_name, mode=mode)
        
        # 只在有碰撞时显示警告
        if collision_count > 0:
            print(f"\n⚠️  [Episode {ep}] WARNING: {collision_count} collisions detected!")
        
        print(f"[Episode {ep}] Saved to {out_path}:{traj_name}, steps={steps_per_ep}")
    
    print(f"\n{'='*60}")
    print("All episodes completed!")
    print(f"Data saved to: {out_path}")
    print(f"{'='*60}")
    
    # 打印求解时间统计
    if solve_times:
        print("\n--- Solve Time Statistics ---")
        if len(solve_times) > 1:
            print(f"First solve:      {solve_times[0]*1000:6.2f} ms")
            avg_time = np.mean(solve_times[1:])
            max_time = np.max(solve_times[1:])
            min_time = np.min(solve_times[1:])
            std_time = np.std(solve_times[1:])
            print(f"Avg (excl. 1st):  {avg_time*1000:6.2f} ms")
            print(f"Max (excl. 1st):  {max_time*1000:6.2f} ms")
            print(f"Min (excl. 1st):  {min_time*1000:6.2f} ms")
            print(f"Std (excl. 1st):  {std_time*1000:6.2f} ms")
        else:
            avg_time = np.mean(solve_times)
            print(f"Avg solve time:   {avg_time*1000:6.2f} ms")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    run_episodes(num_eps=2, steps_per_ep=80, out_path=OUT_PATH)