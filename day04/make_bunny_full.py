# make_bunny_full.py
# Day 4 补充工具：网格顶点全量落盘成 bunny_full.ply（ex2 的输入，丢了可再生）
# 背景：ex1 只采样 200 点，全量点云原本靠手动生成，复跑链断在这里——本脚本补上。
import os
import open3d as o3d

# ---- 老规矩：切到脚本所在目录，产物相对脚本落盘 ----
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 与 ex1 同一份数据源：首跑自动下载到 ~/open3d_data/，之后走本地缓存
mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)

# 验证：顶点和面片都必须非空（文件损坏/格式不识别时 len=0，这里停）
assert len(mesh.vertices) > 0, "顶点数为 0：读出来是空的"
assert len(mesh.triangles) > 0, "面片数为 0：这是纯点云文件，不是网格"

# 全量点云 = 网格的全部顶点（35947 点），不采样、不随机，输出确定性可复现
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices

o3d.io.write_point_cloud("bunny_full.ply", pcd)
print(f"[落盘] bunny_full.ply 已写入，{len(pcd.points)} 点（= 网格顶点数）")
