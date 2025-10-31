#!/usr/bin/env python3
import os, math
import numpy as np
import casadi as ca
import h5py
import subprocess  # 导入 subprocess
import time        # 导入 time

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
W_DU   = 1e-2  # 控制增量的权重 (penalize changes in a, delta)
RHO_SLACK = 1e3  # 对约束违背的惩罚 (soft constraints) 1e4->1e3

W_XPROG = 1.0  # 顶部权重

OUT_PATH = "mpc_dataset.h5"
#np.random.seed(42)
np.random.seed(None)          # 使用系统熵，不固定

# ===== vehicle dynamic model (保持不变) =====
def f_cont(st, u):
    x, y, psi, v = st[0], st[1], st[2], st[3]
    a, delt      = u[0], u[1]
    dx   = v*ca.cos(psi)  # x方向速度
    dy   = v*ca.sin(psi)  # y方向速度
    dpsi = (v/L)*ca.tan(delt)  # 角速度(阿克曼转向)
    dv   = a  # 加速度
    return ca.vertcat(dx, dy, dpsi, dv)

def rk4(st, u, dt=DT):
    k1 = f_cont(st,u)
    k2 = f_cont(st + dt/2*k1, u)
    k3 = f_cont(st + dt/2*k2, u)
    k4 = f_cont(st + dt   *k3, u)
    return st + dt/6*(k1 + 2*k2 + 2*k3 + k4)  #用4阶龙格库塔法精确计算车辆状态更新，比简单的欧拉法准确


# ===== NMPC (重构为 C-Code-Generation 版本) =====
class NMPC_CodeGen:
    def __init__(self, plugin_name="nmpc_explicit_solver"):
        # 这些维度必须是固定的，以便生成C代码
        self.N = N
        self.K = K_NEAR
        self.DT = DT
        
        self.plugin_name = plugin_name
        self.solver_generated = False
        
        # 1. 定义变量和参数的 "Slicers" (切片)
        # 这帮助我们将扁平的向量映射回结构化矩阵
        
        # 决策变量 V (x)
        self.nx_X = 4 * (self.N + 1)
        self.nx_U = 2 * self.N
        self.nx_S = self.K * self.N
        self.nx = self.nx_X + self.nx_U + self.nx_S # V 的总维度
        
        self.v_x_start, self.v_x_end = 0, self.nx_X
        self.v_u_start, self.v_u_end = self.nx_X, self.nx_X + self.nx_U
        self.v_s_start, self.v_s_end = self.v_u_end, self.v_u_end + self.nx_S

        # 参数 P (p)
        self.np_X0 = 4
        self.np_VREF = 1
        self.np_XREF0 = 1
        self.np_OBS_R = self.K * self.N
        self.np_OBS_XY = 2 * self.K * self.N
        self.np_U_PREV = 2 * self.N
        self.np = self.np_X0 + self.np_VREF + self.np_XREF0 + self.np_OBS_R + self.np_OBS_XY + self.np_U_PREV
        
        self.p_x0_start, self.p_x0_end = 0, self.np_X0
        self.p_vref_start, self.p_vref_end = self.np_X0, self.np_X0 + self.np_VREF
        self.p_xref0_start, self.p_xref0_end = self.p_vref_end, self.p_vref_end + self.np_XREF0
        self.p_obs_r_start, self.p_obs_r_end = self.p_xref0_end, self.p_xref0_end + self.np_OBS_R
        self.p_obs_xy_start, self.p_obs_xy_end = self.p_obs_r_end, self.p_obs_r_end + self.np_OBS_XY
        self.p_uprev_start, self.p_uprev_end = self.p_obs_xy_end, self.p_obs_xy_end + self.np_U_PREV

        # 2. 构建求解器
        self._build_solver()
        
        # 3. 初始化猜测
        self._init_guess()

    def _build_solver(self):
        print(f"Building NLP for C-Code Generation (N={self.N}, K={self.K})...")
        
        # --- 1. 定义符号变量 (扁平化的) ---
        V = ca.MX.sym('V', self.nx) # 决策变量 x
        P = ca.MX.sym('P', self.np) # 参数 p

        # --- 2. 使用 Slicers 恢复结构化视图 ---
        X = ca.reshape(V[self.v_x_start:self.v_x_end], 4, self.N + 1)
        U = ca.reshape(V[self.v_u_start:self.v_u_end], 2, self.N)
        S = ca.reshape(V[self.v_s_start:self.v_s_end], self.K, self.N)

        X0     = P[self.p_x0_start:self.p_x0_end]
        V_REF  = P[self.p_vref_start:self.p_vref_end]
        XREF0  = P[self.p_xref0_start:self.p_xref0_end]
        OBS_R  = ca.reshape(P[self.p_obs_r_start:self.p_obs_r_end], self.K, self.N)
        OBS_XY = ca.reshape(P[self.p_obs_xy_start:self.p_obs_xy_end], 2 * self.K, self.N)
        U_PREV = ca.reshape(P[self.p_uprev_start:self.p_uprev_end], 2, self.N)

        # --- 3. 定义变量边界 (lbx, ubx) ---
        lbx = ca.DM.zeros(self.nx, 1)
        ubx = ca.DM.zeros(self.nx, 1)

        lbx[self.v_x_start:self.v_x_end:4] = -ca.inf     # x
        ubx[self.v_x_start:self.v_x_end:4] = ca.inf
        lbx[self.v_x_start+1:self.v_x_end:4] = Y_MIN     # y
        ubx[self.v_x_start+1:self.v_x_end:4] = Y_MAX
        lbx[self.v_x_start+2:self.v_x_end:4] = -ca.inf   # psi
        ubx[self.v_x_start+2:self.v_x_end:4] = ca.inf
        lbx[self.v_x_start+3:self.v_x_end:4] = V_MIN     # v
        ubx[self.v_x_start+3:self.v_x_end:4] = V_MAX

        lbx[self.v_u_start:self.v_u_end:2] = A_MIN       # a
        ubx[self.v_u_start:self.v_u_end:2] = A_MAX
        lbx[self.v_u_start+1:self.v_u_end:2] = -DELTA_MAX # delta
        ubx[self.v_u_start+1:self.v_u_end:2] = DELTA_MAX

        lbx[self.v_s_start:self.v_s_end] = 0.0           # slack
        ubx[self.v_s_start:self.v_s_end] = ca.inf
        
        self.lbx = lbx
        self.ubx = ubx

        # --- 4. 定义约束 (g) 和代价 (f) ---
        cost = 0.0
        g = []
        lbg = []
        ubg = []

        # 初始约束
        g.append(X[:, 0] - X0)
        lbg.append(ca.DM.zeros(4, 1))
        ubg.append(ca.DM.zeros(4, 1))

        for k in range(self.N):
            # 动态约束
            #x_next = rk4(X[:, k], U[:, k], self.DT) #RK4积分器
            x_next = X[:, k] + f_cont(X[:, k], U[:, k]) * self.DT #Euler积分器
            g.append(X[:, k+1] - x_next)
            lbg.append(ca.DM.zeros(4, 1))
            ubg.append(ca.DM.zeros(4, 1))

            # 控制率约束
            if k > 0:
                ddelta = U[1, k] - U[1, k-1]
                g.append(ddelta)
                lbg.append(-DELTA_RATE * self.DT)
                ubg.append(DELTA_RATE * self.DT)

            # 障碍物约束
            for j in range(self.K):
                ox = OBS_XY[2*j, k]
                oy = OBS_XY[2*j+1, k]
                dx = X[0, k] - ox
                dy = X[1, k] - oy
                rj = OBS_R[j, k]
                g_obs = (D_SAFE + R_EGO + rj)**2 - (dx*dx + dy*dy)
                
                # g_obs <= S[j,k]  -->  g_obs - S[j,k] <= 0
                g.append(g_obs - S[j, k])
                lbg.append(-ca.inf)
                ubg.append(0.0)

            # 代价函数
            ey, epsi = X[1, k], X[2, k]
            du = U[:, k] - U_PREV[:, k]
            
            cost_k = (W_EY * ey**2 + W_EPSI * epsi**2 + W_V * (X[3, k] - V_REF)**2
                      + W_U * ca.sumsqr(U[:, k]) + W_DU * ca.sumsqr(du)
                      + RHO_SLACK * ca.sumsqr(S[:, k]))
            
            # cost_k = (W_EY * ey**2 + W_EPSI * epsi**2 + W_V * (X[3, k] - V_REF)**2
            #           + W_U * ca.sumsqr(U[:, k]) + W_DU * ca.sumsqr(du)
            #           + RHO_SLACK * ca.sum1(S[:, k])) # <--- 改为 sum1 (L1范数, 即 sum)
            
            if k > 0:
                du_seq = U[:, k] - U[:, k-1]
                cost_k += W_DU * ca.sumsqr(du_seq)

            cost += cost_k

        # --- 5. 创建 NLP 求解器 ---
        g_flat = ca.vertcat(*g)
        self.lbg = ca.vertcat(*lbg)
        self.ubg = ca.vertcat(*ubg)

        nlp = {'x': V, 'p': P, 'f': cost, 'g': g_flat}
        
        s_opts = {"max_iter": 120, 
                  "tol": 1e-3,  # <--- 保持主要容忍度
                  "print_level": 0,
                  "linear_solver":"mumps", 
                  "hessian_approximation":"limited-memory",

                  "acceptable_tol": 1e-2,  # <--- 一个“还不错”的容忍度 (比 1e-3 宽松)
                  "acceptable_iter": 5,  # <--- 如果连续 5 次迭代都满足 1e-2，就退出
                  "acceptable_obj_change_tol": 1e-4  # <--- 补充：如果目标函数变化太小也退出
                  }
        
        opts = {"ipopt": s_opts}
        
        plugin_file = self.plugin_name + ".so"
        
        # --- 6. 【关键】C代码生成与编译 ---
        if os.path.exists(plugin_file):
            print(f"Found existing solver plugin: {plugin_file}")
        else:
            print(f"Generating C-code for {plugin_file}...")
            # 创建一个临时求解器用于生成
            temp_solver = ca.nlpsol('temp_solver', 'ipopt', nlp, opts)
            temp_solver.generate_dependencies(self.plugin_name + ".c")

            print("Compiling C-code...")
            # 使用 clang (因为您在 macOS 上) 和 -O3 优化
            compiler = "clang" 
            flags = ["-fPIC", "-shared", "-O3"]
            cmd = [compiler] + flags + [self.plugin_name + ".c", "-o", plugin_file]
            
            try:
                subprocess.run(cmd, check=True)
                print("Compilation successful.")
            except subprocess.CalledProcessError as e:
                print(f"Compilation failed: {e}")
                raise
            except FileNotFoundError:
                print(f"Error: Compiler '{compiler}' not found.")
                print("Please ensure 'clang' (or 'gcc' if you change the compiler) is installed and in your PATH.")
                raise

        # --- 7. 加载编译好的求解器 ---
        print("Loading compiled solver...")
        self.solver = ca.nlpsol("solver", "ipopt", plugin_file, opts)
        print("Solver build complete.")


    def _init_guess(self):
        # 为扁平的 V (x) 向量创建初始猜测
        Xg = np.zeros((4, self.N + 1))
        Xg[3, :] = 2.0  # 初始速度猜测
        Ug = np.zeros((2, self.N))
        Sg = np.full((self.K, self.N), 0.1)  # 初始松弛变量猜测

        self.x0_guess = ca.vertcat(
            ca.reshape(Xg, -1, 1),
            ca.reshape(Ug, -1, 1),
            ca.reshape(Sg, -1, 1)
        )
        # 存储解的结构，用于解包
        self.X_opt_prototype = Xg
        self.U_opt_prototype = Ug

    def solve(self, x0, v_ref, xref0, obs_xy_2d, obs_r_2d, u_prev_guess):
        
        # --- 1. 打包参数 P ---
        p_val = ca.vertcat(
            x0.reshape(self.np_X0, 1),
            float(v_ref),
            float(xref0),
            ca.reshape(obs_r_2d, -1, 1),
            ca.reshape(obs_xy_2d, -1, 1),
            ca.reshape(u_prev_guess, -1, 1)
        )

        # --- 2. 更新初始猜测 (Warm Start) ---
        # 注入 U 的猜测
        self.x0_guess[self.v_u_start:self.v_u_end] = ca.reshape(u_prev_guess, -1, 1)
        
        try:
            # --- 3. 求解 NLP ---
            sol = self.solver(
                x0=self.x0_guess,    # 初始猜测 (包含上一时刻的解)
                p=p_val,             # 参数
                lbx=self.lbx,        # 变量下界
                ubx=self.ubx,        # 变量上界
                lbg=self.lbg,        # 约束下界
                ubg=self.ubg         # 约束上界
            )
            
            # --- 4. 解包结果 V ---
            V_opt = sol['x']
            
            # 更新下一次迭代的初始猜测
            self.x0_guess = V_opt 
            
            X_opt_flat = V_opt[self.v_x_start:self.v_x_end]
            U_opt_flat = V_opt[self.v_u_start:self.v_u_end]

            # 转换回 numpy 矩阵
            X_opt = ca.reshape(X_opt_flat, 4, self.N + 1).full()
            U_opt = ca.reshape(U_opt_flat, 2, self.N).full()
            
            return {"success": True, "U": U_opt, "X": X_opt}

        except RuntimeError as e:
            print(f"Solver failed: {e}")
            # 失败时，返回上一个控制猜测，并重置 X/S 猜测
            U_dbg = u_prev_guess
            X_dbg_flat = self.x0_guess[self.v_x_start:self.v_x_end]
            X_dbg = ca.reshape(X_dbg_flat, 4, self.N + 1).full()
            
            # 重置猜测，避免卡死
            self._init_guess() 
            
            return {
                "success": False,
                "U": U_dbg,
                "X": X_dbg,
                "status": str(e)
            }

# ===== random obstacles and logs (保持不变) =====
def gen_static_obstacles(M=2, x_min=2.0, x_max=10.0):  #generate random obstacles
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

def propagate_obstacles(obstacles, dt=DT): #update obstacles position per time step
    for i, o in enumerate(obstacles):
        o["x"] += o["vx"]*dt
        o["y"] += o["vy"]*dt
        # print(f"[propagate_obstacles] Step update - Obstacle {i}: x={o['x']:.2f}, y={o['y']:.2f}")

def nearest_k_obstacles(obstacles, x, y, K=K_NEAR): #find the nearest K_NEAR obstacles
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
            margin = dist - (R_EGO + D_SAFE + o.get("r", 0.0)) #distance regarding radius of ego bot and obstacles with safe distance
            #env += [dx, dy, dist, o["vx"], o["vy"]]
            env += [dx, dy, margin, o["vx"], o["vy"]]
            #dist_margin_list.append(margin)
        env += [boundaries["d_left"], boundaries["d_right"], state.get("ey",0.0), state.get("epsi",0.0)]

        obs_xy = []
        obs_id = []
        # pack the nearest K obstacles
        for j in range(self.K):
            o = obstacles_near[j]
            #obs_xy += [o["x"], o["y"], o["vx"], o["vy"]]
            #obs_xy += [o["x"], o["y"], o["vx"], o["vy"], o.get("r", 0.0)]
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
            #"dist_margin": dist_margin_list,
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
        #obs = np.array([r["obs_xy"] for r in self.buf], dtype=np.float32)  # <--- 新增行
        ids = np.array([r.get("obs_id", [-1]*self.K)      for r in self.buf], dtype=np.int32) 
        d_margin = np.array([r.get("dist_margin", [np.nan]*self.K) for r in self.buf], dtype=np.float32)
        with h5py.File(h5file, mode) as f:
            if traj_name in f:
                del f[traj_name]         # 先删除同名组
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

# ===== main process (更新为使用 NMPC_CodeGen) =====
def run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH):
    
    # 【关键】实例化新的求解器
    # 这将在第一次运行时花费一些时间来编译C代码
    print("Initializing MPC solver (may take a moment for C-compilation)...")
    start_build_time = time.time()
    mpc = NMPC_CodeGen(plugin_name="nmpc_explicit_solver")
    build_time = time.time() - start_build_time
    print(f"Solver initialization complete. Build time: {build_time:.3f} s")
    
    solve_times = [] # 用于跟踪求解时间

    for ep in range(num_eps):
        x0 = np.array([0.0,
                       np.random.uniform(-0.2, 0.2),
                       np.random.uniform(-0.05, 0.05),
                       np.random.uniform(V_MIN, V_MAX)], dtype=np.float64)
        v_ref = np.random.uniform(0.2, V_MAX)
        x_ref0 = float(x0[0])

        obstacles = gen_static_obstacles(M=2, x_min=2.0, x_max=10.0)
        
        # U_prev 现在仅用于warm-start，从 mpc.solve 内部管理
        U_prev = np.zeros((2, N), dtype=np.float64)

        logger = EpisodeLogger(K_obs=K_NEAR)
        st = x0.copy(); t = 0.0

        for k in range(steps_per_ep): #80 horizon moves each episode
            near = nearest_k_obstacles(obstacles, st[0], st[1], K=K_NEAR)
            # 构造形状 (2*K_NEAR, N) 的障碍预测矩阵
            obs_mat = np.zeros((2*K_NEAR, N), dtype=np.float64)
            obs_r_mat = np.zeros((K_NEAR, N), dtype=np.float64)      
            for j in range(K_NEAR):
                ox, oy, vx, vy = near[j]["x"], near[j]["y"], near[j]["vx"], near[j]["vy"]
                rj     = float(near[j].get("r", 0.0))  
                for h in range(N):
                    obs_mat[2*j,   h] = ox + vx*(h*DT)
                    obs_mat[2*j+1, h] = oy + vy*(h*DT)
                    obs_r_mat[j,   h] = rj 

            # --- 求解 ---
            t_start_solve = time.time()
            sol = mpc.solve(st, v_ref, x_ref0, obs_mat, obs_r_mat, U_prev)
            t_end_solve = time.time()
            solve_times.append(t_end_solve - t_start_solve)
            
            U_opt, X_opt = sol["U"], sol["X"]

            if not sol["success"]:
                print(f"[Warning] Solver failed at step {k}, using previous control.")

            # (其余逻辑保持不变)
            a, delta = float(U_opt[0,0]), float(U_opt[1,0])
            st_ca = ca.DM(st.reshape(4,1))
            u_ca  = ca.DM(np.array([[a],[delta]], dtype=np.float64))
            st1 = rk4(st_ca, u_ca, DT).full().flatten()

            d_left  = Y_MAX - st[1]
            d_right = st[1] - Y_MIN
            ey, epsi = st[1], st[2]
            label_ref = (float(X_opt[0,1]), float(X_opt[1,1]))
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

            # 打印状态，每 10 步打印一次
            if k % 10 == 0:
                print(f"[Ep {ep}, Step {k}] t={t:.1f}s, x={st[0]:.2f}, y={st[1]:.2f}, "
                      f"v={st[3]:.2f}, a={a:.2f}, delta={delta:.2f}, "
                      f"SolveTime: {solve_times[-1]*1000:.2f} ms")


            # warm start 右移
            U_shift = np.zeros_like(U_opt)
            U_shift[:, :-1] = U_opt[:, 1:]
            U_shift[:, -1]  = U_opt[:, -1]
            U_prev = U_shift # 这个 U_prev 将在下个循环传递给 mpc.solve

        traj_name = f"traj_ep{ep:03d}"
        mode = "w" if ep == 0 else "a"
        logger.to_h5(out_path, traj_name, mode=mode)
        print(f"[EP {ep}] saved -> {out_path}:{traj_name}, steps={steps_per_ep}")

    print(f"\nDone. File: {out_path}")
    
    # 打印求解时间统计
    if solve_times:
        print("\n--- Solve Time Statistics ---")
        # 忽略第一次（可能包含 JIT 预热）
        if len(solve_times) > 1:
            print(f"First solve: {solve_times[0]*1000:.2f} ms")
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