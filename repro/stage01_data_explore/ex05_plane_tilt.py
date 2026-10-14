# ex05_plane_tilt.py
# 阶段 1 任务 5：背景平面倾角 · 多样本再核（任务 3 挂账清零）
#
# 挂账回顾（任务 3 Q1）：
#   用户肉眼估 bagel 背景倾角 >45°，教师全图重拟合得 41.9°，
#   但全图拟合内点率仅 8.2% —— 因为物体占了约 40% 像素，
#   最小二乘被物体点拉偏，单平面模型在此场景"先天不合"。
#
# 本任务改进：不跟物体抢数据，改采图像边缘环带点（背景的地盘），
# 逐类 3 张 train/good 样本，核两件事：
#   ① 边缘环带拟合 vs 全图拟合的内点率差距（验证"采样策略>蛮力模型"）
#   ② 同 class 多样本倾角是否一致（不一致 → 每帧姿态不同，
#      阶段 2 建记忆库前必须考虑对齐/归一化）

from pathlib import Path

import cv2
import numpy as np
import csv

ROOT = Path('/mnt/e/datasets/mvtec3d')
N_SAMPLES = 3       # 每类取前 3 张 train/good
RING = 30           # 边缘环带宽度（像素）
RES_THRESH = 1e-3   # 内点阈值：残差 < 1mm


def fit_plane(x, y, z):
    """最小二乘拟合 z = a*x + b*y + c（系数含单位，斜率无量纲）。"""
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ coef
    return coef, res


def tilt_deg(coef):
    """平面法线 n=(-a,-b,1) 与竖直方向的夹角（度）。0°=完全水平。"""
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    return float(np.degrees(np.arccos(abs(n[2]))))


def sample_stats(cls, idx):
    """单张样本：边缘环带拟合 + 全图拟合对照，返回各项统计。"""
    xyz = cv2.imread(str(ROOT / cls / 'train' / 'good' / 'xyz' / f'{idx:03d}.tiff'),
                     cv2.IMREAD_UNCHANGED)
    valid = ~(xyz == 0).all(axis=2)          # 任务 2 的遗产：(0,0,0) 是坏点
    H, W = valid.shape

    ring = np.zeros_like(valid)              # 边缘环带掩码
    ring[:RING, :] = ring[-RING:, :] = True
    ring[:, :RING] = ring[:, -RING:] = True
    border = valid & ring                    # 环带 ∩ 有效点 = 背景候选

    p = xyz[border]                          # N×3，米制坐标
    coef_b, res_b = fit_plane(p[:, 0], p[:, 1], p[:, 2])
    q = xyz[valid]                           # 全图有效点（任务 3 的老做法）
    coef_w, res_w = fit_plane(q[:, 0], q[:, 1], q[:, 2])

    return dict(tilt=tilt_deg(coef_b),
                border_inlier=float((np.abs(res_b) < RES_THRESH).mean()),
                whole_inlier=float((np.abs(res_w) < RES_THRESH).mean()),
                n_border=int(border.sum()), n_valid=int(valid.sum()))


def main():
    rows = []
    print(f'{"class":>12} {"sample":>6} {"边缘点":>8} {"有效点":>8} '
          f'{"倾角":>7} {"边缘内点":>8} {"全图内点":>8}')
    for cls_dir in sorted(ROOT.iterdir()):
        if not (cls_dir / 'train' / 'good' / 'xyz').is_dir():
            continue                                  # 语义过滤：只认真类目录
        for i in range(N_SAMPLES):
            s = sample_stats(cls_dir.name, i)
            rows.append((cls_dir.name, i, s['n_border'], s['n_valid'],
                         s['tilt'], s['border_inlier'] * 100, s['whole_inlier'] * 100))
            print(f"{cls_dir.name:>12} good/{i:03d} {s['n_border']:>8} {s['n_valid']:>8} "
                  f"{s['tilt']:>6.2f}° {s['border_inlier'] * 100:>7.1f}% {s['whole_inlier'] * 100:>7.1f}%")

    print('\n=== 按类聚合：同 class 3 样本倾角一致性 ===')
    for cls in sorted({r[0] for r in rows}):
        ts = [r[4] for r in rows if r[0] == cls]
        print(f'{cls:>12}: 均值 {np.mean(ts):6.2f}°   极差 {max(ts) - min(ts):5.2f}°')

    with open('ex05_plane_tilt.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'sample', 'n_border', 'n_valid',
                    'tilt_deg', 'border_inlier_pct', 'whole_inlier_pct'])
        w.writerows(rows)
    print('\nCSV 已写入 ex05_plane_tilt.csv')


if __name__ == '__main__':
    main()
