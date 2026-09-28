# ex1_bunny_to_pcd.py
# Day 4 练习 1：真实数据集第一课——从三角网格到点云（斯坦福兔子）
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ---- 老规矩：切到脚本所在目录，产物相对脚本落盘 ----
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ========= 第 1 步：拿数据（下载 + 验证） =========
# open3d.data.BunnyMesh()：第一次跑自动下载到 ~/open3d_data/，之后走本地缓存
bunny_path = o3d.data.BunnyMesh().path

# 验证 1：文件必须真实存在（下载中断/路径错时在这里停，而不是读个空气继续跑）
assert os.path.exists(bunny_path), f"下载失败：找不到 {bunny_path}"
print(f"[验1] 数据文件存在: {bunny_path}")

# ========= 第 2 步：读三角网格 + 验数 =========
# 网格 = 顶点(N,3) + 面片(M,3)；点云 = 只有顶点
mesh = o3d.io.read_triangle_mesh(bunny_path)

v = np.asarray(mesh.vertices)    # 顶点坐标 (N,3)，单位是米
t = np.asarray(mesh.triangles)   # 每个面片由哪三个顶点组成 (M,3)

# 验证 2：顶点和面片都必须非空（文件损坏/格式不识别时 len=0，这里停）
assert len(v) > 0, "顶点数为 0：读出来是空的"
assert len(t) > 0, "面片数为 0：这是纯点云文件，不是网格"
print(f"[验2] 顶点 {len(v)} 个，面片 {len(t)} 个")

# watertight = 每条边恰好被两个面共享（封闭无洞）
# 真实扫描重建常有微洞，所以只打印观察，不做断言
print(f"[观察] watertight(水密): {mesh.is_watertight()}")

print(f"[数据] 顶点 bbox min: {v.min(axis=0).round(4)}")
print(f"[数据] 顶点 bbox max: {v.max(axis=0).round(4)}")

# ========= 第 3 步：网格 → 点云（面积加权均匀采样） =========
# 在表面上撒 NUM 个点：大面多撒、小面少撒
# 注意：撒出来的点严格落在三角形内部/顶点上，不在外面
NUM = 200
pcd = mesh.sample_points_uniformly(number_of_points=NUM)
pts = np.asarray(pcd.points)

# 验证 3：点数必须等于请求数
assert len(pts) == NUM, f"采样点数 {len(pts)} != 请求 {NUM}"

# 验证 4（数学不变量）：面上任意点 = 三个顶点的加权平均（权重>=0 且和为 1）
# 加权平均必然落在顶点的包围盒内 => 采样点绝不可能跑出 bbox
eps = 1e-6
ok = (pts >= v.min(axis=0) - eps).all() and (pts <= v.max(axis=0) + eps).all()
assert ok, "采样点跑出顶点 bbox：数据有问题，停"
print(f"[验3/4] 采样 {len(pts)} 点，全部落在顶点 bbox 内 ✓")

# ========= 第 4 步：落盘（中间产物双格式） =========
np.save("bunny_pts.npy", pts)               # npy：float64 原值，阶段二喂模型
o3d.io.write_point_cloud("bunny.pcd", pcd)  # pcd：float32 落盘，Open3D 生态用
print("[落盘] bunny_pts.npy / bunny.pcd 已写入")

# ========= 第 5 步：出图对比（顶点 vs 采样点） =========
fig = plt.figure(figsize=(12, 5.5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(v[:, 0], v[:, 1], v[:, 2], s=0.2, c="tab:blue")
ax1.set_title(f"mesh vertices: {len(v)}")
ax1.view_init(elev=20, azim=45)

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2, c="tab:orange")
ax2.set_title(f"sampled points: {len(pts)}")
ax2.view_init(elev=20, azim=45)

for ax in (ax1, ax2):
    ax.set_box_aspect((1, 1, 1))   # 三轴等比例，不然兔子被拉扁
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

plt.tight_layout()
plt.savefig("ex1_mesh_vs_cloud.png", dpi=150)
print("[出图] ex1_mesh_vs_cloud.png 已保存")