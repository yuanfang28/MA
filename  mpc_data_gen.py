import os, math
import numpy as np
import casadi as ca
import h5py

# ===== 全局参数 =====
L = 2.7
DT = 0.1 #积分步长
N  = 20  #prediction horizon
K_NEAR = 3
D_SAFE = 0.6
Y_MIN, Y_MAX = -0.9, 0.9 

V_MIN, V_MAX = 0.0, 8.0
A_MIN, A_MAX = -3.0, 2.0
DELTA_MAX  = np.deg2rad(30)
DELTA_RATE = np.deg2rad(40)

W_EY, W_EPSI, W_V = 6.0, 6.0, 2.0
W_U, W_DU, RHO_SLACK = 1e-2, 1e-2, 1e4

OUT_PATH = "mpc_dataset.h5"
np.random.seed(42)

# ===== 车辆模型 =====
def f_cont(st, u):
    x, y, psi, v = st[0], st[1], st[2], st[3]
    a, delt      = u[0], u[1]
    dx   = v*ca.cos(psi)
    dy   = v*ca.sin(psi)
    dpsi = (v/L)*ca.tan(delt)
    dv   = a
    return ca.vertcat(dx, dy, dpsi, dv)

def rk4(st, u, dt=DT):
    k1 = f_cont(st,u)
    k2 = f_cont(st + dt/2*k1, u)
    k3 = f_cont(st + dt/2*k2, u)
    k4 = f_cont(st + dt   *k3, u)
    return st + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# ===== NMPC =====
class NMPC:
    def __init__(self):
        self.opti = ca.Opti()

        self.X = self.opti.variable(4, N+1)
        self.U = self.opti.variable(2, N)
        self.S = self.opti.variable(K_NEAR, N)

        # 参数（均为二维）
        self.X0     = self.opti.parameter(4, 1)
        self.V_REF  = self.opti.parameter(1, 1)
        self.XREF0  = self.opti.parameter(1, 1)

        # 关键修正：把 (K_NEAR,2,N) 改为 (2*K_NEAR, N)
        self.OBS_XY = self.opti.parameter(2*K_NEAR, N)
        self.U_PREV = self.opti.parameter(2, N)

        self._build_constraints_and_cost()

        p_opts = {"expand": True}
        s_opts = {"max_iter": 120, "tol": 1e-3, "print_level": 0,
                  "linear_solver":"mumps", "hessian_approximation":"limited-memory"}
        self.opti.solver('ipopt', p_opts, s_opts)

        self._init_guess()

    def _build_constraints_and_cost(self):
        cost = 0.0
        self.opti.subject_to(self.X[:,0] == self.X0)

        for k in range(N):
            x_next = rk4(self.X[:,k], self.U[:,k], DT)
            self.opti.subject_to(self.X[:,k+1] == x_next)

            self.opti.subject_to(Y_MIN <= self.X[1,k])
            self.opti.subject_to(self.X[1,k] <= Y_MAX)

            self.opti.subject_to(V_MIN <= self.X[3,k])
            self.opti.subject_to(self.X[3,k] <= V_MAX)
            self.opti.subject_to(A_MIN <= self.U[0,k])
            self.opti.subject_to(self.U[0,k] <= A_MAX)
            self.opti.subject_to(-DELTA_MAX <= self.U[1,k])
            self.opti.subject_to(self.U[1,k] <=  DELTA_MAX)

            if k > 0:
                ddelta = self.U[1,k] - self.U[1,k-1]
                self.opti.subject_to(ddelta <=  DELTA_RATE*DT)
                self.opti.subject_to(ddelta >= -DELTA_RATE*DT)

            # 障碍软约束：索引改为 2D
            for j in range(K_NEAR):
                ox = self.OBS_XY[2*j,   k]
                oy = self.OBS_XY[2*j+1, k]
                dx = self.X[0,k] - ox
                dy = self.X[1,k] - oy
                g  = (D_SAFE**2) - (dx*dx + dy*dy)
                self.opti.subject_to(g <= self.S[j,k])
                self.opti.subject_to(self.S[j,k] >= 0)

            ey, epsi = self.X[1,k], self.X[2,k]
            du = self.U[:,k] - self.U_PREV[:,k]

            cost += (W_EY*ey**2 + W_EPSI*epsi**2 + W_V*(self.X[3,k]-self.V_REF)**2
                     + W_U*ca.sumsqr(self.U[:,k]) + W_DU*ca.sumsqr(du)
                     + RHO_SLACK*ca.sumsqr(self.S[:,k]))

        self.opti.subject_to(Y_MIN <= self.X[1,N])
        self.opti.subject_to(self.X[1,N] <= Y_MAX)

        self.opti.minimize(cost)

    def _init_guess(self):
        Xg = np.zeros((4, N+1)); Xg[3,:] = 2.0
        Ug = np.zeros((2, N))
        self.opti.set_initial(self.X, Xg)
        self.opti.set_initial(self.U, Ug)
        self.opti.set_initial(self.S, 0.1)

    def solve(self, x0, v_ref, xref0, obs_xy_2d, u_prev_guess):
        # obs_xy_2d 形状必须是 (2*K_NEAR, N)
        self.opti.set_value(self.X0,   x0.reshape(4,1))
        self.opti.set_value(self.V_REF, np.array([[float(v_ref)]]))
        self.opti.set_value(self.XREF0, np.array([[float(xref0)]]))
        self.opti.set_value(self.OBS_XY, obs_xy_2d)
        self.opti.set_value(self.U_PREV, u_prev_guess)

        self.opti.set_initial(self.U, u_prev_guess)
        try:
            sol = self.opti.solve()
            U_opt = sol.value(self.U)
            X_opt = sol.value(self.X)
            return {"success": True, "U": U_opt, "X": X_opt}
        # except RuntimeError:
            # return {"success": False, "U": u_prev_guess, "X": self.opti.initial().value(self.X)}
        except RuntimeError as e:
            # 失败分支：用 debug.value 取最后一次迭代/初值
            U_dbg = self.opti.debug.value(self.U)
            X_dbg = self.opti.debug.value(self.X)
            return {
                "success": False,
                "U": U_dbg,
                "X": X_dbg,
                "status": str(e)
            }
# ===== 障碍与日志 =====
def gen_static_obstacles(M=6, x_min=8.0, x_max=40.0):
    obs = []
    for _ in range(M):
        xi = np.random.uniform(x_min, x_max)
        yi = np.random.uniform(Y_MIN+0.2, Y_MAX-0.2)
        obs.append({"x": xi, "y": yi, "vx": 0.0, "vy": 0.0, "r": 0.0})
    return obs

def propagate_obstacles(obstacles, dt=DT):
    for o in obstacles:
        o["x"] += o["vx"]*dt
        o["y"] += o["vy"]*dt

def nearest_k_obstacles(obstacles, x, y, K=K_NEAR):
    d2 = [(((o["x"]-x)**2 + (o["y"]-y)**2), o) for o in obstacles]
    d2.sort(key=lambda t: t[0])
    sel = [t[1] for t in d2[:K]] + [None]*max(0, K-len(d2))
    out = []
    for j in range(K):
        out.append(sel[j] if sel[j] is not None else {"x": x+50.0, "y": 0.0, "vx":0.0, "vy":0.0, "r":0.0})
    return out

class EpisodeLogger:
    def __init__(self, K_obs=K_NEAR):
        self.buf = []; 
        self.K = K_obs
    def step(self, t, state, ctrl, obstacles_near, label_ref, boundaries):
        env = []
        x, y = state["x"], state["y"]
        for j in range(self.K):
            o = obstacles_near[j]
            dx, dy = o["x"]-x, o["y"]-y
            dist   = math.hypot(dx, dy)
            env += [dx, dy, dist, o["vx"], o["vy"]]
        env += [boundaries["d_left"], boundaries["d_right"], state.get("ey",0.0), state.get("epsi",0.0)]
        obs_xy = []
        for j in range(self.K):
            o = obstacles_near[j]
            #obs_xy += [o["x"], o["y"], o["vx"], o["vy"]]
            obs_xy += [o["x"], o["y"], o["vx"], o["vy"], o.get("r", 0.0)]
        self.buf.append({
            "t": t,
            "state": [state["x"], state["y"], state["psi"], state["v"]],
            "ctrl":  [ctrl["a"], ctrl["delta"]],
            "env":   env,
            "label_ref": [label_ref[0], label_ref[1]],
            "obs_xy": obs_xy,  
        })
    def to_h5(self, h5file, traj_name):
        os.makedirs(os.path.dirname(h5file) or ".", exist_ok=True)
        t   = np.array([r["t"] for r in self.buf], dtype=np.float32)
        st  = np.array([r["state"] for r in self.buf], dtype=np.float32)
        ct  = np.array([r["ctrl"]  for r in self.buf], dtype=np.float32)
        env = np.array([r["env"]   for r in self.buf], dtype=np.float32)
        ref = np.array([r["label_ref"] for r in self.buf], dtype=np.float32)
        obs_dim = self.K * 5
        obs = np.array([r.get("obs_xy", [np.nan]*obs_dim) for r in self.buf], dtype=np.float32)
        #obs = np.array([r["obs_xy"] for r in self.buf], dtype=np.float32)  # <--- 新增行
        with h5py.File(h5file, "a") as f:
            if traj_name in f:
                del f[traj_name]         # 先删除同名组
            g = f.create_group(traj_name)
            g.create_dataset("t", data=t)
            g.create_dataset("state", data=st)
            g.create_dataset("ctrl", data=ct)
            g.create_dataset("env", data=env)
            g.create_dataset("label_ref", data=ref)
            g.create_dataset("obs_xy", data=obs)  # <--- 新增行

# ===== 主流程 =====
def run_episodes(num_eps=3, steps_per_ep=80, out_path=OUT_PATH):
    mpc = NMPC()
    for ep in range(num_eps):
        x0 = np.array([0.0,
                       np.random.uniform(-0.2, 0.2),
                       np.random.uniform(-0.05, 0.05),
                       np.random.uniform(1.0, 3.5)], dtype=np.float64)
        v_ref = np.random.uniform(1.5, 4.0)
        x_ref0 = float(x0[0])

        obstacles = gen_static_obstacles(M=6, x_min=10.0, x_max=45.0)
        U_prev = np.zeros((2, N), dtype=np.float64)

        logger = EpisodeLogger(K_obs=K_NEAR)
        st = x0.copy(); t = 0.0

        for k in range(steps_per_ep):
            near = nearest_k_obstacles(obstacles, st[0], st[1], K=K_NEAR)
            # 构造形状 (2*K_NEAR, N) 的障碍预测矩阵
            obs_mat = np.zeros((2*K_NEAR, N), dtype=np.float64)
            for j in range(K_NEAR):
                ox, oy, vx, vy = near[j]["x"], near[j]["y"], near[j]["vx"], near[j]["vy"]
                for h in range(N):
                    obs_mat[2*j,   h] = ox + vx*(h*DT)
                    obs_mat[2*j+1, h] = oy + vy*(h*DT)

            sol = mpc.solve(st, v_ref, x_ref0, obs_mat, U_prev)
            U_opt, X_opt = sol["U"], sol["X"]

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
            # generate moving obstacles
            propagate_obstacles(obstacles, DT)

            # warm start 右移
            U_shift = np.zeros_like(U_opt)
            U_shift[:, :-1] = U_opt[:, 1:]
            U_shift[:, -1]  = U_opt[:, -1]
            U_prev = U_shift

        traj_name = f"traj_ep{ep:03d}"
        logger.to_h5(out_path, traj_name)
        print(f"[EP {ep}] saved -> {out_path}:{traj_name}, steps={steps_per_ep}")

    print(f"\nDone. File: {out_path}")

if __name__ == "__main__":
    run_episodes(num_eps=2, steps_per_ep=80, out_path=OUT_PATH)
