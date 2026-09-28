# ex4_bunny_defect.py
# Day 4 收官练习：真实数据质检闭环——注入缺陷 → CAD 比对检测 → 检出率/精确率
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Part A：载入资产（依赖链终点） ----------
clean = np.load("bunny_clean.npy")
assert clean.ndim == 2 and clean.shape[1] == 3 and len(clean) > 0
print(f"[验1] bunny_clean.npy {clean.shape}")

mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
assert len(mesh.vertices) > 0 and len(mesh.triangles) > 0
print(f"[验2] 参照网格: {len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面片")

# ---------- Part B：建参照系（RaycastingScene）+ 重建法线 ----------
# 场景一份两用：Part C 拿真值法线当"仿真器"，Part D 算比对距离当"检测器"
mesh.compute_vertex_normals()   # 平滑顶点法线（重心坐标插值用）
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(clean)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(15)
normals = np.asarray(pcd.normals)   # 本练习不再用于注入，留给 Q1 对照思考

# ---------- Part C：注入压痕（沿【真值】法线向内，抛物线 punch 轮廓） ----------
# 物理建模：圆柱冲头压入——中心到边缘按抛物线变浅，边缘处深度归零
# 仿真器的信息权限：允许读网格真值（真实压痕是物理过程，不需要"估计"）
tree = cKDTree(clean)
# 注射点选址：离质心最近的表面点 = 躯干最厚处
# （薄壁区深度信号会缩水——选址本身是检测工艺的一部分，见 Q3）
target_idx = int(np.argmin(np.linalg.norm(clean - clean.mean(0), axis=1)))
center = clean[target_idx]
R = 0.010       # 压痕半径 10mm
DEPTH = 0.003   # 中心深度 3mm
nb = np.array(tree.query_ball_point(center, r=R), dtype=int)

d = np.linalg.norm(clean[nb] - center, axis=1)
falloff = 1.0 - (d / R) ** 2               # 抛物线：中心 1，R 处归零

# 真值法线：问网格"离我最近的面片是哪个、重心坐标多少"，
# 用顶点法线按重心坐标插值出【平滑表面法线】——曲面区域不能拿单个面片的法线凑数
closest = scene.compute_closest_points(o3d.core.Tensor(clean[nb], dtype=o3d.core.Dtype.Float32))
face_ids = closest["primitive_ids"].numpy()
uv = closest["primitive_uvs"].numpy()              # (N,2)，第三重心分量 = 1-u-v
tri = np.asarray(mesh.triangles)
vn = np.asarray(mesh.vertex_normals)
w0, w1 = uv[:, 0:1], uv[:, 1:2]
w2 = 1.0 - w0 - w1
n_true = w0 * vn[tri[face_ids, 0]] + w1 * vn[tri[face_ids, 1]] + w2 * vn[tri[face_ids, 2]]
n_true /= np.linalg.norm(n_true, axis=1, keepdims=True)
outward = np.sign((n_true * (clean[nb] - clean.mean(0))).sum(1, keepdims=True))
n_true = n_true * outward                  # 统一翻成"朝外"，压痕才朝里

defected = clean.copy()
defected[nb] -= n_true * (DEPTH * falloff)[:, None]

# 验证 3：注入生效——最深点位移必须 ≈ DEPTH
moved = np.linalg.norm(defected[nb] - clean[nb], axis=1)
assert np.isclose(moved.max(), DEPTH, rtol=0.01), f"最大位移 {moved.max():.5f} != {DEPTH}"
print(f"[验3] 注入 {len(nb)} 点，最大位移 {moved.max()*1000:.2f}mm")

np.save("bunny_defected.npy", defected)

# ---------- Part D：比对检测（点到网格表面距离） ----------
# 检测器的信息权限：只有"扫描件 vs CAD"的距离，不许偷看注入清单
query = o3d.core.Tensor(defected, dtype=o3d.core.Dtype.Float32)
dist = scene.compute_distance(query).numpy()   # 每点到网格的无符号距离（米）

# 仿真-检测对账：最深注入点处，到模距离必须 ≈ DEPTH（真值注入的验收证据）
assert np.isclose(dist[nb].max(), DEPTH, rtol=0.10), \
    f"仿真失真：最深注入点距离 {dist[nb].max()*1000:.2f}mm != {DEPTH*1000:.0f}mm"
print(f"[验4] 仿真对账：最深注入点到模距离 {dist[nb].max()*1000:.2f}mm ≈ 设定深度 ✓")

THRESH = 0.0015   # 1.5mm：噪声底与缺陷信号之间的空档
flagged = dist > THRESH

# ---------- Part E：用注入清单当裁判——检出率/精确率 ----------
injected = np.zeros(len(clean), dtype=bool)
injected[nb] = True
tp = int((flagged & injected).sum())
fp = int((flagged & ~injected).sum())
fn = int((~flagged & injected).sum())
recall = tp / (tp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0.0
print(f"[检测] 阈值 {THRESH*1000:.1f}mm | 检出率 {tp}/{tp+fn} = {recall:.2%} | 精确率 {tp}/{tp+fp} = {precision:.2%}")

base = dist[~injected]
print(f"[噪声底] 未注入区: p99={np.percentile(base, 99)*1000:.2f}mm  max={base.max()*1000:.2f}mm")

# ---------- 出图：1 行 3 面板 ----------
fig = plt.figure(figsize=(18, 5.5))

ax1 = fig.add_subplot(131, projection="3d")
ax1.scatter(defected[~injected, 0], defected[~injected, 1], defected[~injected, 2], s=0.3, c="lightgray")
ax1.scatter(defected[injected, 0], defected[injected, 1], defected[injected, 2], s=0.8, c="red")
ax1.set_title(f"A: injected dent ({len(nb)} pts, {DEPTH*1000:.0f}mm deep)")
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(132, projection="3d")
sc = ax2.scatter(defected[:, 0], defected[:, 1], defected[:, 2], s=0.3,
                 c=dist * 1000, cmap="hot", vmin=0, vmax=DEPTH * 1000)
ax2.set_title("B: distance to CAD mesh (mm)")
fig.colorbar(sc, ax=ax2, shrink=0.6, pad=0.1)
ax2.view_init(elev=20, azim=45)

ax3 = fig.add_subplot(133, projection="3d")
rest = ~(injected | flagged)
ax3.scatter(defected[rest, 0], defected[rest, 1], defected[rest, 2], s=0.3, c="lightgray")
ax3.scatter(defected[flagged & ~injected, 0], defected[flagged & ~injected, 1], defected[flagged & ~injected, 2], s=0.8, c="orange")
ax3.scatter(defected[~flagged & injected, 0], defected[~flagged & injected, 1], defected[~flagged & injected, 2], s=0.8, c="blue")
ax3.scatter(defected[flagged & injected, 0], defected[flagged & injected, 1], defected[flagged & injected, 2], s=0.8, c="red")
ax3.set_title(f"C: TP red / FN blue / FP orange | R={recall:.0%} P={precision:.0%}")
ax3.view_init(elev=20, azim=45)

for ax in (ax1, ax2, ax3):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")

plt.tight_layout()
plt.savefig("ex4_defect_result.png", dpi=150)
print("[出图] ex4_defect_result.png 已保存")
