import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent          # 所有产物存到脚本所在目录（day03/）
rng = np.random.default_rng(42)      # 固定随机种子：每次运行结果一模一样，方便对照

# ===== 1. 伪造金属板点云 =====
# 模拟线扫：200×200 个采样点，物理尺寸 10mm × 10mm → 点距 0.05mm
N = 200
x = np.linspace(0, 10, N)            # x: 沿激光线方向，单位 mm
y = np.linspace(0, 10, N)            # y: 传送带运动方向
xx, yy = np.meshgrid(x, y)           # 变成两个 (200,200) 网格：每个格子的物理坐标
zz = np.zeros_like(xx)               # 理想平面：高度全 0

# 真实线扫仪有噪声，加 ±0.005mm 高斯噪声（典型量级），别加太大
zz += rng.normal(0, 0.005, size=xx.shape)

# 缺陷1：一道细长划痕（凹陷），宽 0.3mm 深 0.08mm，横在板子中间
scratch = (np.abs(yy - 5.0) < 0.15) & (xx > 3.0) & (xx < 7.0)
zz[scratch] -= 0.08

# 缺陷2：一个圆形毛刺（凸起），半径 0.5mm 高 0.2mm
bump = (xx - 8.0) ** 2 + (yy - 2.0) ** 2 < 0.5 ** 2
zz[bump] += 0.2

# 拼成点云的标准身材 (N*N, 3)
points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
print("点云形状:", points.shape)     # 预期 (40000, 3)
print("dtype:", points.dtype)        # 预期 float64

# ===== 2. 存成四种格式 =====
np.save(OUT / "plate.npy", points)   # NumPy 原生：数组原样 dump（float64 全保留）

pcd = o3d.geometry.PointCloud()      # Open3D 的点云对象
pcd.points = o3d.utility.Vector3dVector(points)   # 把 (N,3) 数组"挂"进对象
o3d.io.write_point_cloud(str(OUT / "plate.pcd"), pcd)   # 默认 ASCII
o3d.io.write_point_cloud(str(OUT / "plate.ply"), pcd)
o3d.io.write_point_cloud(str(OUT / "plate.xyz"), pcd)

# ===== 3. 读回来，验证一致性 =====
def load_any(path):
    """统一入口：npy 用 numpy 读，其余用 Open3D 读"""
    if str(path).endswith(".npy"):
        return np.load(path)
    pcd_back = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd_back.points)

for name in ["plate.npy", "plate.xyz", "plate.pcd", "plate.ply"]:
    path = OUT / name
    q = load_any(path)
    size_kb = path.stat().st_size / 1024
    # 浮点比较不能用 ==，要用 allclose（等下 Q1 会问你为什么）
    same = np.allclose(q, points, atol=1e-4)
    print(f"{name:12s} 大小={size_kb:8.1f} KB  点数={q.shape[0]:6d}  一致: {same}")

# ===== 4. 画图（WSL 无显示，必须 savefig，禁止 plt.show()）====
fig = plt.figure(figsize=(12, 5))

# 左图：俯视（质检工位最常看的视角），颜色=高度
ax1 = fig.add_subplot(1, 2, 1)
sc = ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis", s=1)
ax1.set_aspect("equal")              # x/y 都是 mm，必须 1:1，否则划痕变形
ax1.set_title("top view (color = height z)")
fig.colorbar(sc, ax=ax1, label="z / mm")

# 右图：3D 视角（4 万点画太密，抽 1/20 看形状）
step = 20
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(points[::step, 0], points[::step, 1], points[::step, 2],
            c=points[::step, 2], cmap="viridis", s=1)
ax2.set_title("3D view (every 20th point)")

plt.tight_layout()
plt.savefig(OUT / "ex1_pointcloud_overview.png", dpi=150)
print("已保存 ex1_pointcloud_overview.png")