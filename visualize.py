import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# ---- 单集可视化（带时间渐变色）----
def visualize_episode(
    h5_path: str,
    ep_name: str,
    D_SAFE=0.1,
    R_EGO=0.5,
    Y_MIN=-2,
    Y_MAX=2,
    draw_all_obstacle_circles=False,
    cmap_name='viridis',  # 可选: 'viridis', 'plasma', 'cool', 'rainbow'
    vehicle_circle_mode='adaptive'  # 'all', 'adaptive', 'sparse', 'endpoints'
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
        elif "R_EGO" in g:
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

    fig, ax = plt.subplots(figsize=(10, 6))
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    pad = max(1.0, 0.05 * max(xmax - xmin, 1e-6))
    xmin -= pad; xmax += pad

    # 道路边界
    ax.hlines([Y_MIN, Y_MAX], xmin, xmax, colors='gray', linestyles="--", 
              linewidth=1.5, label="lane bounds", alpha=0.6)

    # ========== 车辆轨迹（渐变色线）==========
    cmap = cm.get_cmap(cmap_name)
    
    # 创建渐变色线段
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 时间归一化到 [0, 1]
    norm = plt.Normalize(0, T-1)
    colors = cmap(norm(np.arange(T-1)))
    
    lc = LineCollection(segments, colors=colors, linewidths=2.5, label='vehicle trajectory')
    ax.add_collection(lc)
    
    # ========== 车辆圆形采样策略（可配置）==========
    if vehicle_circle_mode == 'all':
        # 模式 1: 绘制所有时间步（最密集，与障碍物散点完全同步）
        vehicle_sample_indices = np.arange(T)
    
    elif vehicle_circle_mode == 'adaptive':
        # 模式 2: 自适应采样（推荐，根据轨迹长度智能调整）
        if T <= 20:
            vehicle_sample_step = 1  # 短轨迹：每步都画
        elif T <= 50:
            vehicle_sample_step = 2  # 中等轨迹：每 2 步画一个
        elif T <= 100:
            vehicle_sample_step = 4  # 长轨迹：每 4 步画一个
        else:
            vehicle_sample_step = max(1, T // 25)  # 超长轨迹：最多画 25 个
        
        vehicle_sample_indices = list(range(0, T, vehicle_sample_step))
    
    elif vehicle_circle_mode == 'sparse':
        # 模式 3: 稀疏采样（每 10 步或 5% 的位置）
        step = max(10, T // 20)
        vehicle_sample_indices = list(range(0, T, step))
    
    elif vehicle_circle_mode == 'endpoints':
        # 模式 4: 仅起点和终点（最简洁）
        vehicle_sample_indices = [0, T-1]
    
    else:
        raise ValueError(f"Unknown vehicle_circle_mode: {vehicle_circle_mode}")
    
    # 确保起点和终点一定被包含（对于非 'all' 模式）
    if vehicle_circle_mode != 'all':
        vehicle_sample_indices = list(set(vehicle_sample_indices + [0, T-1]))
        vehicle_sample_indices.sort()
    
    # 绘制车辆圆形
    for idx in vehicle_sample_indices:
        color = cmap(norm(idx))
        circle = Circle((xy[idx, 0], xy[idx, 1]), R_EGO, 
                      fill=True, facecolor=color, edgecolor='black',
                      linewidth=1.0, alpha=0.6)
        ax.add_patch(circle)
    
    # 只在起点和终点添加文本标签（避免太杂乱）
    for idx in [0, T-1]:
        color = cmap(norm(idx))
        ax.text(xy[idx, 0], xy[idx, 1] + R_EGO + 0.15, 
               f't={idx}', ha='center', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    # ========== 障碍物轨迹（渐变色）==========
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
        ts = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        rs = np.array([p[3] for p in pts])
        
        # 障碍物轨迹渐变线
        obs_points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        obs_segments = np.concatenate([obs_points[:-1], obs_points[1:]], axis=1)
        obs_colors = cmap(norm(ts[:-1]))
        
        obs_lc = LineCollection(obs_segments, colors=obs_colors, 
                               linewidths=2.0, linestyles='--', alpha=0.8)
        ax.add_collection(obs_lc)
        
        # 障碍物标记点
        scatter = ax.scatter(xs, ys, c=ts, cmap=cmap, s=40, 
                           marker='o', edgecolors='black', linewidths=0.8,
                           label=f'obstacle {oid}', vmin=0, vmax=T-1, zorder=5)
        
        # 绘制障碍物圆形（起点和终点）
        if draw_all_obstacle_circles:
            for k in range(len(xs)):
                color = cmap(norm(ts[k]))
                ax.add_patch(Circle((xs[k], ys[k]), rs[k], 
                                  fill=False, edgecolor=color, linewidth=1.0, alpha=0.6))
                ax.add_patch(Circle((xs[k], ys[k]), rs[k] + R_EGO + D_SAFE,
                                  fill=False, edgecolor=color, linestyle=":", 
                                  linewidth=0.8, alpha=0.4))
        else:
            # 只绘制起点和终点的圆
            for ix in [0, -1]:
                color = cmap(norm(ts[ix]))
                ax.add_patch(Circle((xs[ix], ys[ix]), rs[ix], 
                                  fill=False, edgecolor=color, linewidth=1.5, alpha=0.8))
                ax.add_patch(Circle((xs[ix], ys[ix]), rs[ix] + R_EGO + D_SAFE,
                                  fill=False, edgecolor=color, linestyle=":", 
                                  linewidth=1.2, alpha=0.5))
                
                # 添加起点/终点标签
                if ix == 0:
                    ax.text(xs[ix], ys[ix] - rs[ix] - 0.3, 
                           f'Obs {oid}\nt={int(ts[ix])}', 
                           ha='center', fontsize=7, style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.6))

    # ========== 颜色条（时间轴）==========
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Time step', rotation=270, labelpad=20, fontsize=10)

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(f"{ep_name} : Vehicle & Obstacles Trajectories (Time-coded)", 
                fontsize=12, fontweight='bold')
    ax.axis("equal")
    ax.set_xlim([xmin, xmax])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.show(block=True)

# ---- 遍历并依次显示所有 episode ----
def list_episodes(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        eps = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    return sorted(eps)

def visualize_all(h5_path="mpc_dataset.h5",
                  D_SAFE=0.1, R_EGO=0.5, Y_MIN=-2, Y_MAX=2,
                  draw_all_obstacle_circles=False,
                  cmap_name='viridis',
                  vehicle_circle_mode='adaptive'):
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
                draw_all_obstacle_circles=draw_all_obstacle_circles,
                cmap_name=cmap_name,
                vehicle_circle_mode=vehicle_circle_mode
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
        draw_all_obstacle_circles=False,
        cmap_name='viridis',  # 可选: 'plasma', 'cool', 'rainbow', 'twilight','viridis'
        vehicle_circle_mode='all'  # 可选: 'all', 'adaptive', 'sparse', 'endpoints'
    )