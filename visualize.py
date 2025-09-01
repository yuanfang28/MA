import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize_episode(
    h5_path="mpc_dataset.h5",
    ep_name="traj_ep000",
    D_SAFE=0.6,     # 与生成数据时保持一致
    R_EGO=0.0,      # 自车等效半径（若考虑车辆尺寸）
    Y_MIN=-0.9,
    Y_MAX=0.9,
    draw_all_obstacle_circles=False,  # True 则每一步都画圆，图会更密
    save_path=None   # 如需保存图像，传入 "figure.png"
):
    with h5py.File(h5_path, "r") as f:
        if ep_name not in f:
            raise KeyError(f"Episode '{ep_name}' not found in {h5_path}.")
        g = f[ep_name]

        # 车辆状态与轨迹
        state = g["state"][:]          # (steps, 4) -> [x,y,psi,v]
        xy = state[:, :2]              # (steps, 2)

        # 尝试读取障碍数据
        if "obs_xy" not in g:
            raise KeyError(f"'obs_xy' not found in {ep_name}. 请确认已将障碍信息写入 HDF5。")
        obs = g["obs_xy"][:]           # (steps, K*4) 或 (steps, K*5)

    steps, cols = obs.shape
    if cols % 5 == 0:
        K = cols // 5
        fmt = 5  # 含半径
    elif cols % 4 == 0:
        K = cols // 4
        fmt = 4  # 不含半径
    else:
        raise ValueError(f"Unsupported obs_xy shape: {obs.shape}")

    # 若无半径，统一赋默认半径（可以根据你的场景修改）
    default_r = 0.20

    fig, ax = plt.subplots(figsize=(8, 5))

    # 车道边界
    xmin, xmax = xy[:,0].min(), xy[:,0].max()
    # 稍微扩展一下 x 轴范围，避免切边
    pad = max(1.0, 0.05 * (xmax - xmin + 1e-6))
    xmin -= pad; xmax += pad
    ax.hlines([Y_MIN, Y_MAX], xmin, xmax, linestyles="--", linewidth=1.0, label="lane bounds")

    # 车辆轨迹
    ax.plot(xy[:,0], xy[:,1], "-", label="vehicle")

    # 每个障碍
    for j in range(K):
        if fmt == 5:
            ox = obs[:, 5*j + 0]
            oy = obs[:, 5*j + 1]
            # vx = obs[:, 5*j + 2]; vy = obs[:, 5*j + 3]
            r  = obs[:, 5*j + 4]      # 每步半径（通常恒定）
        else:
            ox = obs[:, 4*j + 0]
            oy = obs[:, 4*j + 1]
            # vx = obs[:, 4*j + 2]; vy = obs[:, 4*j + 3]
            r  = np.full(steps, default_r)

        # 轨迹（静态障碍会是重叠的点）
        ax.plot(ox, oy, "o--", markersize=3, label=f"obstacle {j+1}")

        # 圆：默认只在第 0 步画，清晰；如需全时刻画，把下面 if 改成 for 循环
        if draw_all_obstacle_circles:
            for k in range(steps):
                circ_obs  = Circle((ox[k], oy[k]), r[k], fill=False, linewidth=1.0)
                circ_safe = Circle((ox[k], oy[k]), r[k] + R_EGO + D_SAFE, fill=False, linestyle="--", linewidth=0.8)
                ax.add_patch(circ_obs); ax.add_patch(circ_safe)
        else:
            circ_obs  = Circle((ox[0], oy[0]), r[0], fill=False, linewidth=1.5)
            circ_safe = Circle((ox[0], oy[0]), r[0] + R_EGO + D_SAFE, fill=False, linestyle="--", linewidth=1.0)
            ax.add_patch(circ_obs); ax.add_patch(circ_safe)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{ep_name} : vehicle & obstacles (radius + safety)")
    ax.axis("equal")
    ax.set_xlim([xmin, xmax])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # 示例：可按需修改 EP 名称
    visualize_episode(
        h5_path="mpc_dataset.h5",
        ep_name="traj_ep000",
        D_SAFE=0.6,
        R_EGO=0.0,
        Y_MIN=-0.9,
        Y_MAX=0.9,
        draw_all_obstacle_circles=False,
        save_path=None
    )
