import h5py

# 打开文件（只读模式）
with h5py.File("mpc_dataset.h5", "r") as f:
    # 列出顶层所有 group（比如 traj_ep000, traj_ep001 ...）
    print("Trajectories:", list(f.keys()))

    # 假设查看第一个轨迹
    g = f["traj_ep000"]
    print("Datasets in traj_ep000:", list(g.keys()))

    # 查看某个 dataset 的 shape 和数据
    print("state shape:", g["state"].shape)
    print("state[0]:", g["state"][0])  # 打印第一步的状态

    print("ctrl shape:", g["ctrl"].shape)
    print("ctrl[0]:", g["ctrl"][0])    # 打印第一步的控制量
