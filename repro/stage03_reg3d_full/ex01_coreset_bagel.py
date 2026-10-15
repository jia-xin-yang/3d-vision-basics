# ex01_coreset_bagel.py
# 阶段 3 任务 1：Coreset 核心集 —— 用 1% 的库，保多少检索力？
#
# 阶段 2 的记忆库是全量的：bagel 一类 479344 点 × 33 维 float32 ≈ 60MB。
# Reg3D-AD / PatchCore 一脉的正式做法是建库后再做 Coreset 采样：
# 从全量库里挑一个"小而代表性"的子集当记忆库（PatchCore 论文预算 ≈1%）。
#
# 为什么敢扔 99%？因为 voxel 1mm 之后相邻点的 FPFH 高度相似
# （阶段 2 任务 1 局部性检验：近邻对特征距只有随机对的 1/5.67），
# 库里大部分点是"重复信息"——留着只增加查询成本，不增加判别力。
#
# 实验设计（控制变量，阶段 1 任务 4 的老纪律）：
#   只换"库"这一个变量，评分仍用阶段 2 的 score_top1%，AUROC 口径与 ex04 完全相同，
#   这样 AUROC 的任何差异都只能归因于 coreset 本身。
#   每个预算跑两种采样：贪心 k-center vs 随机采样 → 直接对照"为什么要贪心"。
#   预算：0.1% / 0.5% / 1% / 2% / 5% / 10%，参照线 = ex04 全量库 0.9923。
#
# 核心算法：贪心 k-center（PatchCore 用的就是这套）
#   1. 随机选一个种子点进 coreset
#   2. 维护每个点到"已选 coreset"的最近平方距离 min_dist
#   3. 每轮选 min_dist 最大的点（= 当前最没被覆盖的点）入 coreset
#   4. 用新中心更新 min_dist，直到预算用完
#   理论保证：k-center 贪心是 2-近似（覆盖半径 ≤ 最优解的 2 倍）
#
# 实现提速（呼应 Day 2 挑战 B 的向量化思想）：
#   平方距离展开 |x-c|^2 = |x|^2 - 2x·c + |c|^2，每轮只做一次"矩阵 × 向量"，
#   免开方、免逐点 Python 循环；min_dist 单调不增，用 np.minimum 原地更新。

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import rankdata

# 复用阶段 2 已验收的 FPFH 提取流水线（跨目录导入：把 stage02 加进模块搜索路径）
STAGE02 = Path(__file__).resolve().parent.parent / 'stage02_btf_baseline'
sys.path.insert(0, str(STAGE02))
from ex02_memory_bank import extract_fpfh   # noqa: E402

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLS = 'bagel'
BANK = STAGE02 / 'ex02_memory_bank_bagel.npy'
TOP_PCT = 0.01               # 与 ex04 完全一致：图级分数 = 逐点最近邻距离的 top1% 均值
REF_AUROC_FULL = 0.9923      # ex04 全量库实测（本任务参照线）
SEED = 0                     # 固定种子：跨次运行结果可复现（老规矩）
BUDGETS = [0.1, 0.5, 1, 2, 5, 10]   # coreset 预算（占全量库的百分比）


def auroc(scores, labels):
    """手写秩法 AUROC（Mann-Whitney U，与 ex04 逐字一致）：只认排序，不认绝对值。"""
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(labels)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def load_test_features():
    """把 bagel 全部 110 张测试图的特征预计算一次（各预算配置共用）。
    流水线与 ex04 相同，只是提前算好，免得每个预算都重复提一遍 FPFH。"""
    test_dir = ROOT / CLS / 'test'
    jobs = [(fp, 'good') for fp in sorted((test_dir / 'good' / 'xyz').glob('*.tiff'))]
    for d in sorted(test_dir.iterdir()):
        if d.is_dir() and d.name != 'good':
            jobs += [(fp, d.name) for fp in sorted((d / 'xyz').glob('*.tiff'))]
    out = []
    for i, (fp, label) in enumerate(jobs):
        xyz_img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        _, feat = extract_fpfh(xyz_img)
        out.append((f'{label}/{fp.stem}', label, feat))
        if (i + 1) % 20 == 0:
            print(f'  特征预计算 [{i + 1}/{len(jobs)}] ...')
    return out


def greedy_kcenter(X, budget, seed=SEED):
    """手写贪心 k-center。返回 (选中点下标数组, 最终覆盖半径)。

    每轮迭代（全向量化，免 Python 逐点循环）：
      d2       = |x|^2 - 2 x·c + |c|^2   # 新中心 c 到所有点的平方距离
      min_dist = min(min_dist, d2)        # 每个点"离已选集合最近"的平方距离
      选 min_dist 最大的点作为下一个中心
    平方距离与真实距离排序一致（开方是单调函数），argmax 不受影响；
    float32 相对误差 ~1e-7，对"选最远点"这种粗决策绰绰有余。
    """
    n = len(X)
    budget = min(budget, n)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))            # 种子：随机一个点（PatchCore 同款做法）

    sel = np.empty(budget, dtype=np.int64)
    sel[0] = first
    x_sq = np.einsum('ij,ij->i', X, X)      # |x|^2 只算一次，之后每轮复用
    # 初始 min_dist = 到种子点的平方距离
    min_dist = x_sq - 2.0 * (X @ X[first]) + x_sq[first]
    for i in range(1, budget):
        nxt = int(np.argmax(min_dist))      # 当前"最没被覆盖"的点
        sel[i] = nxt
        d2 = x_sq - 2.0 * (X @ X[nxt]) + x_sq[nxt]
        np.minimum(min_dist, d2, out=min_dist)  # 覆盖集合只增不减 → 逐点取 min
    cover_radius = float(np.sqrt(min_dist.max()))  # 最坏点到 coreset 的距离
    return sel, cover_radius


def eval_bank(bank_subset, test_data):
    """给定一个库子集，复刻 ex04 的评分：逐点 kNN 距离 → top1% 均值 → AUROC。
    返回 (auroc, 查询总秒数)。"""
    tree = cKDTree(bank_subset)
    t0 = time.time()
    scores, labels = [], []
    for _, label, feat in test_data:
        d, _ = tree.query(feat, k=1)
        m = max(1, int(len(d) * TOP_PCT))
        scores.append(float(np.mean(np.sort(d)[-m:])))
        labels.append(1 if label != 'good' else 0)
    return auroc(scores, labels), time.time() - t0


def main():
    bank = np.load(BANK)
    n = len(bank)
    print(f'全量库 {bank.shape}（{bank.nbytes / 1024 / 1024:.1f}MB），'
          f'参照线 AUROC = {REF_AUROC_FULL}')

    print('预计算 110 张测试图特征（一次性，各预算共用）...')
    test_data = load_test_features()

    rows = []
    for pct in BUDGETS:
        budget = max(1, int(n * pct / 100))
        # ① 贪心 k-center
        t0 = time.time()
        sel, radius = greedy_kcenter(bank, budget)
        cs_sec = time.time() - t0
        a_knn, q_sec = eval_bank(bank[sel], test_data)
        rows.append((pct, 'kcenter', budget, radius, cs_sec, q_sec, a_knn))
        print(f'  k-center {pct:>4}% ({budget:>6} 点): 覆盖半径 {radius:6.2f}，'
              f'AUROC {a_knn:.4f}（贪心 {cs_sec:.0f}s + 查询 {q_sec:.0f}s）')
        # ② 随机采样（同预算对照组：同一批测试、同一评分，只换采样方法）
        rng = np.random.default_rng(SEED)
        rnd = rng.choice(n, size=budget, replace=False)
        a_rnd, q2 = eval_bank(bank[rnd], test_data)
        rows.append((pct, 'random', budget, float('nan'), float('nan'), q2, a_rnd))
        print(f'  随机     {pct:>4}% ({budget:>6} 点): AUROC {a_rnd:.4f}'
              f'（查询 {q2:.0f}s）')

    with open('ex01_coreset_curve.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['budget_pct', 'method', 'n_coreset', 'cover_radius',
                    'coreset_sec', 'query_sec', 'auroc_top1pct'])
        w.writerows(rows)

    # 双图：AUROC vs 预算（左，对照贪心/随机/全量参照线）；覆盖半径 vs 预算（右）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ks = [r for r in rows if r[1] == 'kcenter']
    rd = [r for r in rows if r[1] == 'random']
    axes[0].plot([r[0] for r in ks], [r[6] for r in ks], 'o-', label='k-center (greedy)')
    axes[0].plot([r[0] for r in rd], [r[6] for r in rd], 's--', label='random')
    axes[0].axhline(REF_AUROC_FULL, ls=':', c='gray',
                    label=f'full bank = {REF_AUROC_FULL}')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('coreset budget (% of full bank)')
    axes[0].set_ylabel('image AUROC (score_top1%)')
    axes[0].set_title('AUROC vs budget')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot([r[0] for r in ks], [r[3] for r in ks], 'o-')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('coreset budget (% of full bank)')
    axes[1].set_ylabel('covering radius (FPFH L2)')
    axes[1].set_title('k-center coverage')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('ex01_coreset_curve.png', dpi=150)
    print('图已存 ex01_coreset_curve.png')


if __name__ == '__main__':
    main()
