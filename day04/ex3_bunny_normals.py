# ex3_bunny_normals.py
# Day 4 练习 3：法线估计 + 法线着色可视化 + RANSAC 平面分割初体验
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Part A：读清洗后的兔子，重建成点云对象 ----------
clean = np.load("bunny_clean.npy")

# 验证 1：形状必须是 (N,3)
assert clean.ndim == 2 and clean.shape[1] == 3, f"形状不对: {clean.shape}"
assert len(clean) > 0, "空点云"
print(f"[验1] bunny_clean.npy: {clean.shape}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(clean)

# ---------- Part B：法线估计 ----------
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
)

n = np.asarray(pcd.normals)
pts = np.asarray(pcd.points)   # Vector3dVector 没有 .shape，先转 numpy
assert n.shape == pts.shape, f"法线形状 {n.shape} != 点形状 {pts.shape}"
norm_len = np.linalg.norm(n, axis=1)
assert np.allclose(norm_len, 1.0, atol=1e-6), f"法线非单位长: {norm_len.min()}~{norm_len.max()}"
print(f"[验2/3] 法线 {n.shape}，长度 {norm_len.min():.6f}~{norm_len.max():.6f}")

# ---------- Part C：方向歧义观察（只观察，不修正） ----------
up_raw = (n[:, 1] > 0).mean()
print(f"[观察] 定向前 法线 +y 比例: {up_raw:.2%}")
rgb_raw = (n + 1) / 2

# ---------- Part D：一致性定向后对比 ----------
pcd.orient_normals_consistent_tangent_plane(15)
n_fix = np.asarray(pcd.normals)
up_fix = (n_fix[:, 1] > 0).mean()
print(f"[观察] 定向后 法线 +y 比例: {up_fix:.2%}")
rgb_fix = (n_fix + 1) / 2

# ----------  ----------
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.003,
    ransac_n=3,
    num_iterations=1000,
)
a, b, c, d = plane_model
print(f"[RANSAC] 平面: {a:.3f}x+{b:.3f}y+{c:.3f}z+{d:.4f}=0")
print(f"[RANSAC] 内点 {len(inliers)}/{len(clean)} = {len(inliers)/len(clean):.2%}")

# ---------- 出图：1 行 3 面板 ----------
fig = plt.figure(figsize=(18, 5.5))

ax1 = fig.add_subplot(131, projection="3d")
ax1.scatter(clean[:,0], clean[:,1], clean[:,2], s=0.3, c=rgb_raw)
ax1.set_title(f"A: raw normals ({up_raw:.0%} +y)")
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(132, projection="3d")
ax2.scatter(clean[:,0], clean[:,1], clean[:,2], s=0.3, c=rgb_fix)
ax2.set_title(f"B: oriented ({up_fix:.0%} +y)")
ax2.view_init(elev=20, azim=45)

ax3 = fig.add_subplot(133, projection="3d")
mask = np.zeros(len(clean), dtype=bool)
mask[inliers] = True
ax3.scatter(clean[~mask,0], clean[~mask,1], clean[~mask,2], s=0.3, c="lightgray")
ax3.scatter(clean[mask,0], clean[mask,1], clean[mask,2], s=0.6, c="red")
ax3.set_title(f"C: RANSAC inliers {len(inliers)} ({len(inliers)/len(clean):.1%})")
ax3.view_init(elev=20, azim=45)

for ax in (ax1, ax2, ax3):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

plt.tight_layout()
plt.savefig("ex3_normals_ransac.png", dpi=150)
print("[出图] ex3_normals_ransac.png 已保存")
