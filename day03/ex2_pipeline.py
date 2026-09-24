import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(7)      # 换种子7：和练习1的板子区分开

# ===== 1. 读入练习1的板子，再把它弄脏 =====
pcd = o3d.io.read_point_cloud(str(OUT / "plate.pcd"))
points = np.asarray(pcd.points)
print(f"原始点数: {len(points)}")

# 污染1：200 个飞点——撒在板子上方 0.5~3mm 高空，模拟反光/灰尘/阳光干扰
flying = rng.uniform([0, 0, 0.5], [10, 10, 3.0], size=(200, 3))

# 污染2：一圈"夹具"——板子四边外 1mm 的边框，高 2mm，模拟视野里的治具
edge = np.linspace(-1, 11, 150)                  # 每条边 150 个点
z2 = np.full(150, 2.0)                           # 夹具高度恒定 2mm
fixture = np.concatenate([
    np.stack([edge, np.full_like(edge, -1.0), z2], axis=-1),   # 下边 y=-1
    np.stack([edge, np.full_like(edge, 11.0), z2], axis=-1),   # 上边 y=11
    np.stack([np.full_like(edge, -1.0), edge, z2], axis=-1),   # 左边 x=-1
    np.stack([np.full_like(edge, 11.0), edge, z2], axis=-1),   # 右边 x=11
])

dirty = np.concatenate([points, flying, fixture])
print(f"弄脏后点数: {len(dirty)}")

# ===== 2. 预处理流水线：裁剪 → 降采样 → 滤波 =====
# Step 1: ROI 裁剪——只留 x∈(2,8), y∈(2,8) 的检测区，夹具和板子边缘全切掉
roi_mask = (dirty[:, 0] > 2) & (dirty[:, 0] < 8) & \
           (dirty[:, 1] > 2) & (dirty[:, 1] < 8)
roi = dirty[roi_mask]
print(f"ROI 裁剪后点数: {len(roi)}")

# Step 2: 体素降采样——0.1mm 格子，格内取质心
pcd_roi = o3d.geometry.PointCloud()
pcd_roi.points = o3d.utility.Vector3dVector(roi)
down = pcd_roi.voxel_down_sample(voxel_size=0.1)
down_pts = np.asarray(down.points)
print(f"体素降采样后点数: {len(down_pts)}")

# Step 3: 统计滤波——k=20 个邻居，删 平均邻距 > μ+2σ 的点
pcd_down = o3d.geometry.PointCloud()
pcd_down.points = o3d.utility.Vector3dVector(down_pts)
clean, keep_idx = pcd_down.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0)
clean_pts = np.asarray(clean.points)
print(f"统计滤波后点数: {len(clean_pts)}（删掉了 {len(down_pts) - len(clean_pts)} 个野点）")
np.save(OUT / "clean_pts.npy", clean_pts)   # 中间产物落盘，下游免重跑
# ===== 3. 四张快照对比 =====
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
snapshots = [
    ("1 dirty",       dirty,     None),        # 自动范围：飞点 z 到 3.0，会把 colorbar 拉飞
    ("2 ROI",         roi,       (-0.1, 0.3)), # 后三张统一范围，才能横向对比
    ("3 downsampled", down_pts,  (-0.1, 0.3)), # 呼应 Day2 练习3 的 vmin/vmax
    ("4 cleaned",     clean_pts, (-0.1, 0.3)),
]
for ax, (title, pts, zlim) in zip(axes, snapshots):
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", s=1,
                    vmin=(zlim[0] if zlim else None),
                    vmax=(zlim[1] if zlim else None))
    ax.set_aspect("equal")
    ax.set_title(f"{title}  n={len(pts)}")
    fig.colorbar(sc, ax=ax)

plt.tight_layout()
plt.savefig(OUT / "ex2_pipeline.png", dpi=150)
print("已保存 ex2_pipeline.png")