# ex04_eval_auroc.py
# 阶段 2 任务 4：全测试集评测 —— 两个统计量的 AUROC 对决
#
# 任务 3 留下悬案：score_mean 与 score_top1% 谁该当图级异常分？
# 6 张小样本判不了 → 用 bagel 全部 110 张测试图（22 good + 88 缺陷）裁决。
#
# AUROC 不用 sklearn，手写秩实现（Mann-Whitney U）：
#   AUROC = P(随机缺陷分 > 随机正常分) —— 只认排序，不认分数绝对值
#   这正是阶段 1 任务 4 Q2 那个"智障模型 AUROC=0.5"的正面兑现。

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from ex02_memory_bank import extract_fpfh

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLS = 'bagel'
TOP_PCT = 0.01


def auroc(scores, labels):
    """labels: 1=缺陷, 0=正常。秩法（并列取平均秩），越接近 1 越好。"""
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(labels)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def main():
    t0 = time.time()
    bank = np.load('ex02_memory_bank_bagel.npy')
    tree = cKDTree(bank)
    print(f'库 {bank.shape}，KD 树 {time.time() - t0:.1f}s')

    # 汇总测试集：good + 全部缺陷目录
    test_dir = ROOT / CLS / 'test'
    jobs = [(fp, 'good') for fp in sorted((test_dir / 'good' / 'xyz').glob('*.tiff'))]
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name != 'good':
            jobs += [(fp, d.name) for fp in sorted((d / 'xyz').glob('*.tiff'))]
    print(f'测试集 {len(jobs)} 张（good {sum(1 for _, l in jobs if l == "good")}，'
          f'缺陷 {sum(1 for _, l in jobs if l != "good")}）')

    rows = []
    for i, (fp, label) in enumerate(jobs):
        xyz_img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        q, feat = extract_fpfh(xyz_img)
        d, _ = tree.query(feat, k=1)
        m = max(1, int(len(d) * TOP_PCT))
        rows.append((f'{label}/{fp.stem}', label, float(d.mean()),
                     float(np.mean(np.sort(d)[-m:]))))
        if (i + 1) % 20 == 0:
            print(f'  [{i + 1}/{len(jobs)}] ...')

    with open('ex04_test_scores.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label', 'score_mean', 'score_top1pct'])
        w.writerows(rows)

    labels = [1 if r[1] != 'good' else 0 for r in rows]
    a_mean = auroc([r[2] for r in rows], labels)
    a_top1 = auroc([r[3] for r in rows], labels)
    print(f'\nAUROC 对决：score_mean = {a_mean:.4f}   score_top1% = {a_top1:.4f}')
    winner = 'score_mean' if a_mean > a_top1 else 'score_top1%'
    print(f'胜者：{winner}（Δ = {abs(a_mean - a_top1):.4f}）')
    print(f'总耗时 {time.time() - t0:.0f}s')

    # 分数分布图：两个统计量各一 panel，看 good/缺陷的分布分离
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, name in [(axes[0], 2, 'score_mean'), (axes[1], 3, 'score_top1%')]:
        g = [r[col] for r in rows if r[1] == 'good']
        b = [r[col] for r in rows if r[1] != 'good']
        ax.hist(g, bins=20, alpha=0.6, label=f'good (n={len(g)})', density=True)
        ax.hist(b, bins=20, alpha=0.6, label=f'defect (n={len(b)})', density=True)
        ax.set_title(f'{name}  AUROC={auroc([r[col] for r in rows], labels):.3f}')
        ax.legend()
    fig.tight_layout()
    fig.savefig('ex04_score_distributions.png', dpi=150)
    print('分布图已存 ex04_score_distributions.png')


if __name__ == '__main__':
    main()
