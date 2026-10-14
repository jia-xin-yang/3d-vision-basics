# ex01_fpfh_first_contact.py
# 阶段 2 任务 1：FPFH 初接触 —— 从单张 good 图到 33 维几何指纹
#
# 阶段 1 结论在本题全部上岗：
#   ① 坏点是 (0,0,0)，np.isnan 滤不掉 → (xyz==0).all(axis=2)
#   ② 背景是大平面（法线与视轴夹角 ~83°，每类姿态固定）→ 物体用
#      "到背景平面的距离"隔离（平面剥离 = 阶段 1 给阶段 2 投的票）
#   ③ 3D 网格各向异性且密（x 步 0.05mm）→ 先 voxel 降采样再算特征
#
# 流水线：读图 → 滤坏点 → 边缘环带拟合背景平面 → 距平面剥离背景
#         → voxel 1mm 降采样 → 估法线 → FPFH → 特征质量局部性检验

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path('/mnt/e/datasets/mvtec3d')
RING = 30            # 边缘环带宽（像素），ex05 验证过的背景采样法
VOXEL = 0.001        # 降采样体素 1mm
NORMAL_R = 0.005     # 法线邻域半径 5mm
NORMAL_K = 30        # 法线邻域最多点数
FPFH_R = 0.010       # FPFH 邻域半径 10mm（惯例：约 2× 法线半径）
FPFH_K = 100
OBJ_THRESH = 0.010   # 距平面 > 10mm 判为物体（bagel 凸出 +70mm，余量充足）
SEED = 42


def load_valid(cls, split, defect, idx):
    """读 xyz tiff，滤 (0,0,0) 坏点，返回有效点 N×3（米）。"""
    xyz = cv2.imread(str(ROOT / cls / split / defect / 'xyz' / f'{idx:03d}.tiff'),
                     cv2.IMREAD_UNCHANGED)
    valid = ~(xyz == 0).all(axis=2)
    return xyz[valid].astype(np.float64)


def plane_from_ring(xyz_img):
    """对整幅 xyz 图取边缘环带有效点，拟合背景平面。
    返回 (a, b, c)：z = a*x + b*y + c，以及单位法线 n。"""
    valid = ~(xyz_img == 0).all(axis=2)
    H, W = valid.shape
    ring = np.zeros_like(valid)
    ring[:RING, :] = ring[-RING:, :] = True
    ring[:, :RING] = ring[:, -RING:] = True
    border = valid & ring
    p = xyz_img[border].astype(np.float64)
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    (a, b, c), *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)
    return (a, b, c), n, border.sum()


def main():
    np.random.seed(SEED)
    cls, idx = 'bagel', 0

    # ---- 1. 读图 + 滤坏点 ----
    xyz_img = cv2.imread(str(ROOT / cls / 'train' / 'good' / 'xyz' / f'{idx:03d}.tiff'),
                         cv2.IMREAD_UNCHANGED)
    H, W = xyz_img.shape[:2]
    valid = ~(xyz_img == 0).all(axis=2)
    pts = xyz_img[valid].astype(np.float64)
    print(f'[1] 有效点 {len(pts)} / {H * W}（坏点 {100 * (1 - valid.mean()):.1f}%）')

    # ---- 2. 边缘环带拟合背景平面 + 剥离 ----
    (a, b, c), n, n_border = plane_from_ring(xyz_img)
    # 有符号距离 = (z - (a x + b y + c)) / √(a²+b²+1)，分母即法线模长
    denom = np.sqrt(a * a + b * b + 1.0)
    signed = (pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) / denom
    obj = pts[signed > OBJ_THRESH]
    print(f'[2] 背景平面 z={a:.2f}x+{b:.2f}y+{c:.3f}（边缘点 {n_border}）')
    print(f'    距平面 >{OBJ_THRESH * 1000:.0f}mm 判物体：{len(obj)} 点 '
          f'（占有效点 {100 * len(obj) / len(pts):.1f}%）')

    # ---- 3. voxel 降采样 ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj)
    pcd_ds = pcd.voxel_down_sample(voxel_size=VOXEL)
    q = np.asarray(pcd_ds.points)
    print(f'[3] voxel {VOXEL * 1000:.0f}mm 降采样：{len(obj)} → {len(q)}')

    # ---- 4. 法线估计（FPFH 的原料）----
    pcd_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_R, max_nn=NORMAL_K))
    pcd_ds.orient_normals_consistent_tangent_plane(100)  # 方向一致性（慢但稳）
    normals = np.asarray(pcd_ds.normals)
    print(f'[4] 法线 {normals.shape}，|n| 均值 {np.linalg.norm(normals, axis=1).mean():.4f}')

    # ---- 5. FPFH ----
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_ds, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_R, max_nn=FPFH_K))
    feat = np.asarray(fpfh.data).T          # Open3D 给 (33, N)，转置成 (N, 33)
    nan_cnt = int(np.isnan(feat).sum())
    print(f'[5] FPFH {feat.shape}（33 维/点），值域 [{np.nanmin(feat):.3f}, '
          f'{np.nanmax(feat):.3f}]，NaN {nan_cnt} 个')

    # ---- 6. 特征质量局部性检验 ----
    # 设计：局部几何描述子 → 空间近邻的特征距离应显著小于随机点对。
    # 统计量：① 近邻对 vs 随机对的特征距离均值比 ② (空间距, 特征距) 的 Spearman 秩相关
    tree = cKDTree(q)
    m = min(2000, len(q))
    sel = np.random.choice(len(q), m, replace=False)
    _, nns = tree.query(q[sel], k=6)        # 含自己，k=6 → 5 个真近邻
    d_near, f_near = [], []
    for i, nn in zip(sel, nns):
        for j in nn[1:]:
            d_near.append(np.linalg.norm(q[i] - q[j]))
            f_near.append(np.linalg.norm(feat[i] - feat[j]))
    d_rnd, f_rnd = [], []
    for _ in range(len(d_near)):
        i, j = np.random.choice(len(q), 2, replace=False)
        d_rnd.append(np.linalg.norm(q[i] - q[j]))
        f_rnd.append(np.linalg.norm(feat[i] - feat[j]))
    d_near, f_near = np.array(d_near), np.array(f_near)
    d_rnd, f_rnd = np.array(d_rnd), np.array(f_rnd)
    rho = spearmanr(np.concatenate([d_near, d_rnd]),
                    np.concatenate([f_near, f_rnd])).statistic
    print(f'[6] 局部性检验：近邻对特征距 {f_near.mean():.3f} vs 随机对 '
          f'{f_rnd.mean():.3f}（比 {f_rnd.mean() / max(f_near.mean(), 1e-9):.2f}×）；'
          f'Spearman ρ = {rho:.3f}')

    # ---- 7. 四联张验收图 ----
    rgb = cv2.imread(str(ROOT / cls / 'train' / 'good' / 'rgb' / f'{idx:03d}.png'),
                     cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    zmap = np.ma.masked_where(~valid, xyz_img[..., 2])
    fig = plt.figure(figsize=(15, 4.2))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(rgb); ax.set_title('rgb reference'); ax.axis('off')
    ax = fig.add_subplot(1, 4, 2)
    im = ax.imshow(zmap, cmap='viridis')
    ax.set_title('depth z (valid masked)'); ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    sub = np.random.choice(len(q), min(12000, len(q)), replace=False)
    sc = ax.scatter(q[sub, 0], q[sub, 1], q[sub, 2], c=q[sub, 2], cmap='viridis',
                    s=1, alpha=0.5)
    ax.set_title(f'object cloud ({len(q)} pts)'); ax.axis('off')
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
    ax = fig.add_subplot(1, 4, 4, projection='3d')
    nv = normals.copy()
    nv[nv[:, 2] < 0] *= -1                 # 朝镜头翻转，颜色可读
    # c 传 N×3 RGB 数组时 matplotlib 自动忽略 cmap，故不传（避免警告刷屏）
    ax.scatter(q[sub, 0], q[sub, 1], q[sub, 2], c=np.abs(nv[sub]), s=1, alpha=0.5)
    ax.set_title('normals |nx|,|ny|,|nz|→RGB'); ax.axis('off')
    fig.tight_layout()
    fig.savefig('ex01_fpfh_first_contact.png', dpi=150)
    print('[7] 四联张已存 ex01_fpfh_first_contact.png')


if __name__ == '__main__':
    main()
