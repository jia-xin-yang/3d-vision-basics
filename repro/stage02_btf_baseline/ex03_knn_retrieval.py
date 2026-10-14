# ex03_knn_retrieval.py
# 阶段 2 任务 3：kNN 检索 —— 测试点在记忆库里找最近邻居
#
# 核心思想：异常分 = 与"最近的正常原型"的不相似度。
#   正常点 → 库里必有长得像的 → kNN 距离小
#   缺陷点 → 库里没有这种形状 → kNN 距离大
#
# 图级聚合给两个统计量，故意制造对比：
#   score_mean   = 全点距离均值（多数派稀释陷阱的再现现场）
#   score_top1   = 最远 1% 点的距离均值（少数派保护）

import csv
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ex02_memory_bank import extract_fpfh   # 复用建库流水线（一致性 = Q3）

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLS = 'bagel'
TOP_PCT = 0.01                              # 图级聚合：最远 1% 点

# 查询集：2 张正常 + 4 种缺陷各 1 张（bagel 的 4 种缺陷全到齐）
QUERIES = [
    ('good', 0), ('good', 1),
    ('crack', 0), ('contamination', 0), ('hole', 0), ('combined', 0),
]


def main():
    t0 = time.time()
    bank = np.load('ex02_memory_bank_bagel.npy')
    print(f'记忆库 {bank.shape} 载入 {time.time() - t0:.1f}s')
    t0 = time.time()
    tree = cKDTree(bank)                     # KD 树：33 维空间的索引（Q3 的数感）
    print(f'KD 树建立 {time.time() - t0:.1f}s')

    rows = []
    print(f'\n{"query":>18} {"label":>8} {"点数":>7} {"score_mean":>10} {"score_top1%":>11}')
    for defect, idx in QUERIES:
        xyz_img = cv2.imread(str(ROOT / CLS / 'test' / defect / 'xyz' / f'{idx:03d}.tiff'),
                             cv2.IMREAD_UNCHANGED)
        q, feat = extract_fpfh(xyz_img)
        d, _ = tree.query(feat, k=1)         # 逐点最近邻距离
        m = max(1, int(len(d) * TOP_PCT))
        top1 = float(np.mean(np.sort(d)[-m:]))
        s_mean = float(d.mean())
        rows.append((f'{defect}/{idx:03d}', defect, len(q), s_mean, top1))
        print(f'{defect + "/" + f"{idx:03d}":>18} {defect:>8} {len(q):>7} '
              f'{s_mean:>10.2f} {top1:>11.2f}')

    with open('ex03_query_scores.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['query', 'label', 'n_points', 'score_mean', 'score_top1pct'])
        w.writerows(rows)

    # 分离检查：good 的最大分 vs 缺陷的最小分
    good = [r for r in rows if r[1] == 'good']
    bad = [r for r in rows if r[1] != 'good']
    for stat, col in [('score_mean', 3), ('score_top1%', 4)]:
        g_max = max(r[col] for r in good)
        b_min = min(r[col] for r in bad)
        verdict = '✓ 完全分离' if b_min > g_max else ('✗ 未分离' if b_min < g_max else '— 贴线')
        print(f'[{stat}] good 最大 {g_max:.2f} vs 缺陷最小 {b_min:.2f}  →  {verdict}')


if __name__ == '__main__':
    main()
