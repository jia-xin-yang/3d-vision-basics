# Day 4 练习 2：预处理流水线移植——尺度感 + 体素降采样 + 统计滤波
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ---------- Part A：读兔子，先体检尺度 ----------
# 第 7 行改成：
pcd = o3d.io.read_point_cloud("bunny_full.ply")
extent = pcd.get_axis_aligned_bounding_box().get_extent()
print(f"原始点数: {len(pcd.points)}")
print(f"bbox 尺寸(米): {extent}")

# ---------- Part B：voxel_size 参数扫描 ----------
# 故意扫三个量级，亲手看"参数带单位"意味着什么
for vs in [0.001, 0.002, 0.005]:
    ds = pcd.voxel_down_sample(voxel_size=vs)
    print(f"voxel_size={vs*1000:.0f}mm -> {len(ds.points)} 点")

V = 0.002  # 选定 2mm 作为工作分辨率
down = pcd.voxel_down_sample(voxel_size=V)
print(f"\n选定 {V}，降采样后: {len(down.points)} 点")

# ---------- Part C：统计滤波 ----------
cl, ind = down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
removed = len(down.points) - len(cl.points)
print(f"统计滤波删除: {removed} 点，剩余 {len(cl.points)} 点")

# ---------- Part D：落盘（老规矩：下游要用，存 npy，commit 前 gitignore）----------
np.save("bunny_down.npy", np.asarray(down.points))
np.save("bunny_clean.npy", np.asarray(cl.points))
print("已落盘: bunny_down.npy / bunny_clean.npy")

# ---------- Part E：降采样前后对比图 ----------
fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(121, projection='3d')
pts0 = np.asarray(pcd.points)
n_show = min(8000, len(pts0))                                          # ← 新增
idx = np.random.default_rng(42).choice(len(pts0), n_show, replace=False)  # ← 8000 换成 n_show
s0 = pts0[idx]
ax1.scatter(s0[:, 0], s0[:, 1], s0[:, 2], s=1, c=s0[:, 1], cmap='viridis')
ax1.set_title(f"original ({len(pts0)} pts, showing 8000)")

ax2 = fig.add_subplot(122, projection='3d')
pts1 = np.asarray(down.points)      # 降采样后数量可控，全画
ax2.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], s=2, c=pts1[:, 1], cmap='viridis')
ax2.set_title(f"voxel {V*1000:.0f}mm ({len(pts1)} pts)")

plt.savefig("ex2_downsample_compare.png", dpi=150, bbox_inches='tight')
print("图已保存: ex2_downsample_compare.png")