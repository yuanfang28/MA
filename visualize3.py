import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---- 单集可视化（必须含 obs_id，且只弹窗不保存）----
def visualize_episode(
    h5_path: str,
    ep_name: str,
    D_SAFE=0.1,
    R_EGO=0.5,
    Y_MIN=-2,
    Y_MAX=2,
    draw_all_obstacle_circles=False
):
    with h5py.File(h5_path, "r") as f:
        if ep_name not in f:
            raise KeyError(f"Episode '{ep_name}' not found in {h5_path}.")
        g = f[ep_name]

        state = g["state"][:]              # (T,4): x,y,psi,v
        xy = state[:, :2]
        obs = g["obs_xy"][:]               # (T,K*4) 或 (T,K*5)
        if "obs_id" not in g:
            raise KeyError(f"'obs_id' not found in {ep_name}.")
        obs_id = g["obs_id"][:]            # (T,K)

        R_EGO_file = None
        if "R_EGO" in g.attrs:
            R_EGO_file = float(g.attrs["R_EGO"])
        elif "R_EGO" in g:          # 若你选择了方式B（数据集）
            R_EGO_file = float(g["R_EGO"][()])  

    if R_EGO_file is not None:
        R_EGO = R_EGO_file
    


    T, cols = obs.shape
    if cols % 5 == 0:
        K = cols // 5
        get_xy_r = lambda t, j: (obs[t, 5*j+0], obs[t, 5*j+1], obs[t, 5*j+4])
    elif cols % 4 == 0:
        K = cols // 4
        default_r = 0.20
        get_xy_r = lambda t, j: (obs[t, 4*j+0], obs[t, 4*j+1], default_r)
    else:
        raise ValueError(f"Unsupported obs_xy shape: {obs.shape}")

    fig, ax = plt.subplots(figsize=(8, 5))
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    pad = max(1.0, 0.05 * max(xmax - xmin, 1e-6))
    xmin -= pad; xmax += pad

    ax.hlines([Y_MIN, Y_MAX], xmin, xmax, linestyles="--", linewidth=1.0, label="lane bounds")
    ax.plot(xy[:, 0], xy[:, 1], "-", label="vehicle")
    for k in range(len(xy)):
        ax.add_patch(Circle((xy[k,0], xy[k,1]), R_EGO, fill=False, linewidth=1.0))
        # ax.add_patch(Circle((xy[k,0], xy[k,1]), R_EGO + D_SAFE,
        #                     fill=False, linestyle="--", linewidth=0.8))

    # 聚合同一 id 的轨迹
    tracks = {}
    for t in range(T):
        for j in range(K):
            oid = int(obs_id[t, j])
            if oid < 0:
                continue
            ox, oy, r = get_xy_r(t, j)
            tracks.setdefault(oid, []).append((t, ox, oy, r))

    for oid, pts in tracks.items():
        pts.sort(key=lambda p: p[0])
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]; rs = [p[3] for p in pts]
        ax.plot(xs, ys, "o--", markersize=3, label=f"obstacle {oid}")
        if draw_all_obstacle_circles:
            for k in range(len(xs)):
                ax.add_patch(Circle((xs[k], ys[k]), rs[k], fill=False, linewidth=1.0))
                ax.add_patch(Circle((xs[k], ys[k]), rs[k] + R_EGO + D_SAFE,
                                    fill=False, linestyle="--", linewidth=0.8))
        else:
            for ix in [0, -1]:
                ax.add_patch(Circle((xs[ix], ys[ix]), rs[ix], fill=False, linewidth=1.2))
                ax.add_patch(Circle((xs[ix], ys[ix]), rs[ix] + R_EGO + D_SAFE,
                                    fill=False, linestyle="--", linewidth=1.0))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{ep_name} : vehicle & obstacles")
    ax.axis("equal")
    ax.set_xlim([xmin, xmax])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    plt.show(block=True)  # 关闭当前窗口后才会继续

# ---- 遍历并依次显示所有 episode ----
def list_episodes(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        # 只拿 group，按名字排序
        eps = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    # 常见命名是 traj_ep000 这种，直接字典序就行
    return sorted(eps)

def visualize_all(h5_path="mpc_dataset.h5",
                  D_SAFE=0.1, R_EGO=0.5, Y_MIN=-2, Y_MAX=2,
                  draw_all_obstacle_circles=False):
    eps = list_episodes(h5_path)
    if not eps:
        raise RuntimeError("No episode in H5 file")
    print(f"Found {len(eps)} episodes:")
    for name in eps:
        print(" -", name)

    for i, ep in enumerate(eps, 1):
        print(f"\n[{i}/{len(eps)}] Showing {ep} (close the window to continue)...")
        try:
            visualize_episode(
                h5_path=h5_path,
                ep_name=ep,
                D_SAFE=D_SAFE,
                R_EGO=R_EGO,
                Y_MIN=Y_MIN,
                Y_MAX=Y_MAX,
                draw_all_obstacle_circles=draw_all_obstacle_circles
            )
        except Exception as e:
            print(f"[WARN] Skip {ep}: {e}")

if __name__ == "__main__":
    visualize_all(
        h5_path="mpc_dataset.h5",
        D_SAFE=0.1,
        R_EGO=0.5,
        Y_MIN=-2,
        Y_MAX=2,
        draw_all_obstacle_circles=False
    )
