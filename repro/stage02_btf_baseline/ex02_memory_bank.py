# ex02_memory_bank.py
# 阶段 2 任务 2：建记忆库（bagel 类，前 K 张 train/good）
#
# 记忆库 = 异常检测的"正常原型仓库"：把训练集正常样本的每个点
# 的 FPFH 指纹全部存进一个大矩阵，检索时拿测试点的指纹来比。
#
# 与 ex01 的两点刻意不同：
#   ① 法线朝向改用"朝相机原点"：ex01 的 tangent_plane 法只保证
#      图内一致，跨图可能整体翻转——建库后特征是跨图比对的，
#      朝向约定必须全局确定（点云在相机坐标系，相机在原点）。
#   ② 产物是 bank.npy（~60MB，已入 .gitignore：可再生 > 可入库）。

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from ex01_fpfh_first_contact import plane_from_ring   # 复用 ex01 验证过的平面拟合

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLS = 'bagel'
K = 30                    # 建库用前 30 张 train/good（244 张里的子集）
VOXEL, NORMAL_R, NORMAL_K = 0.001, 0.005, 30
FPFH_R, FPFH_K = 0.010, 100
OBJ_THRESH = 0.010
CAM = np.array([0.0, 0.0, 0.0])   # 点云是相机坐标系，光心在原点


def extract_fpfh(xyz_img):
    """单张 xyz 图 → (降采样点云 q, FPFH 特征 feat)。流水线与 ex01 相同。"""
    valid = ~(xyz_img == 0).all(axis=2)
    pts = xyz_img[valid].astype(np.float64)
    (a, b, c), _, _ = plane_from_ring(xyz_img)
    denom = np.sqrt(a * a + b * b + 1.0)
    signed = (pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) / denom
    obj = pts[signed > OBJ_THRESH]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_R, max_nn=NORMAL_K))
    # 关键：朝向全局确定（朝相机），保证跨图特征可比
    pcd.orient_normals_towards_camera_location(CAM)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_R, max_nn=FPFH_K))
    q = np.asarray(pcd.points)
    feat = np.asarray(fpfh.data).T.astype(np.float32)   # (N, 33)，float32 省一半盘
    return q, feat


def main():
    xyz_dir = ROOT / CLS / 'train' / 'good' / 'xyz'
    files = sorted(xyz_dir.glob('*.tiff'))[:K]
    print(f'建库：{CLS} train/good 前 {len(files)} 张（共 {len(list(xyz_dir.glob("*.tiff")))} 张）')

    rows, chunks = [], []
    t_all = time.time()
    for i, fp in enumerate(files):
        t0 = time.time()
        xyz_img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        q, feat = extract_fpfh(xyz_img)
        dt = time.time() - t0
        rows.append((fp.stem, len(q), dt))
        chunks.append(feat)
        print(f'  [{i + 1:>2}/{len(files)}] {fp.stem}: {len(q):>6} 点, {dt:.1f}s')
    bank = np.vstack(chunks)

    np.save('ex02_memory_bank_bagel.npy', bank)
    with open('ex02_bank_build_log.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'n_points', 'seconds'])
        w.writerows(rows)

    mb = bank.nbytes / 1024 / 1024
    norms = np.linalg.norm(bank, axis=1)
    print(f'\n记忆库: shape={bank.shape} dtype={bank.dtype} {mb:.1f}MB')
    print(f'  NaN {int(np.isnan(bank).sum())} 个；指纹 L2 范数均值 {norms.mean():.2f} '
          f'± {norms.std():.2f}')
    print(f'  总耗时 {time.time() - t_all:.0f}s；bank.npy 已被 .gitignore 覆盖（可再生）')


if __name__ == '__main__':
    main()
