# ex02_max_scoring_bagel.py
# 阶段 3 任务 2：图级打分三选一 —— 式 1-2 裸 max vs 式 3 加权 vs score_top1%
#
# 阶段 2 的图级分数是 score_top1%（最可疑 ~150 点取均值），Reg3D-AD 论文式 1-2
# 用的是 max：逐点最近邻距离里的最大值，"一个工件只要有 1 个点足够离群，整件异常"。
# 式 3 再给裸 max 乘一个权重 w（邻近密度校准），缓解"库噪音点/覆盖盲区单点爆表"。
#
# 实验设计（延续任务 1 的控制变量纪律）：
#   2 种库（全量 479344 点 / 0.5% coreset 2396 点）× 3 种打分 = 6 格全交叉。
#   同一个 KD 树查询结果喂给三种打分，打分本身零额外查询成本。
#   三问里你要用这张 6 格表裁决任务 1 Q3② 的预测（max 对库覆盖更敏感？）。
#
# 式 3 复刻（与详解文档逐字对应）：
#   s = w · s*，  s* = max_i d_i（最疑点 m* 的最近邻距离）
#   w = 1 - exp(d1) / Σ_{i=1..b} exp(di)
#   di = 最疑点特征到库中第 i 近邻的距离（d1 为最近），b = 邻域大小
#   实现把 m* 特征再查一次库取 k=b 即可（m* 是测试点，库里没有它自己）。

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from ex01_coreset_bagel import (   # 同目录复用任务 1 已验收的零件
    BANK, TOP_PCT, REF_AUROC_FULL, SEED,
    auroc, greedy_kcenter, load_test_features,
)

CORESET_PCT = 0.5        # 任务 1 结论：0.5% coreset 已打平全量库
B_NEIGH = 5              # 式 3 邻域大小 b（超参数，PatchCore 一脉常用小整数）


def score_three_ways(feat, d, tree):
    """同一组逐点最近邻距离 d，喂三种图级打分，返回三元组。

    top1%：阶段 2 的冠军（最可疑 m 个点取均值，自带稀释缓冲）
    max  ：式 1-2 裸 max（零缓冲，整张图押在最极端一个点上）
    wmax ：式 3 加权 max（邻近密度校准：只跟个别库点远 → 压分；跟整片远 → 保分）
    """
    m = max(1, int(len(d) * TOP_PCT))
    top1 = float(np.mean(np.sort(d)[-m:]))

    mx = float(d.max())

    # 式 3：m* = 最疑点；取它在库里的 b 个近邻，按公式算 w
    mstar = int(np.argmax(d))
    dd, _ = tree.query(feat[mstar], k=B_NEIGH)   # dd[0]=d1（最近），升序
    w = 1.0 - np.exp(dd[0]) / np.exp(dd).sum()
    wmax = float(w * mx)
    return top1, mx, wmax


def main():
    bank = np.load(BANK)
    n = len(bank)
    print(f'全量库 {bank.shape}；coreset {CORESET_PCT}%（任务 1 验证档）；'
          f'b={B_NEIGH}；参照线 AUROC={REF_AUROC_FULL}')

    print('预计算 110 张测试图特征（一次性）...')
    test_data = load_test_features()

    # 0.5% coreset：贪心 k-center，同任务 1 参数（种子固定可复现）
    t0 = time.time()
    sel, radius = greedy_kcenter(bank, max(1, int(n * CORESET_PCT / 100)))
    print(f'coreset {len(sel)} 点，覆盖半径 {radius:.2f}，贪心 {time.time() - t0:.0f}s')

    banks = [('full', bank), (f'coreset_{CORESET_PCT}%', bank[sel])]
    methods = ['top1pct', 'max', 'wmax']

    per_image = {}   # (bank_name, method) -> [score, ...]，与 test_data 对齐
    summary = []
    for bank_name, subset in banks:
        tree = cKDTree(subset)
        cols = {mthd: [] for mthd in methods}
        t0 = time.time()
        for i, (_, label, feat) in enumerate(test_data):
            d, _ = tree.query(feat, k=1)          # 一次查询，三种打分共用
            top1, mx, wmax = score_three_ways(feat, d, tree)
            cols['top1pct'].append(top1)
            cols['max'].append(mx)
            cols['wmax'].append(wmax)
            if (i + 1) % 20 == 0:
                print(f'  [{bank_name}][{i + 1}/{len(test_data)}] ...')
        labels = [1 if lab != 'good' else 0 for _, lab, _ in test_data]
        for mthd in methods:
            a = auroc(cols[mthd], labels)
            per_image[(bank_name, mthd)] = cols[mthd]
            summary.append((bank_name, mthd, a))
            print(f'  >> {bank_name:>15s} | {mthd:>7s} | AUROC {a:.4f}')
        print(f'  （{bank_name} 查询 {time.time() - t0:.0f}s）')

    # 逐图分数落盘：6 列分数 + 标签，供事后分析（哪张图被哪种打分救/害）
    with open('ex02_image_scores.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        header = ['image', 'label']
        for bank_name, _ in banks:
            header += [f'{bank_name}_{m}' for m in methods]
        w.writerow(header)
        for i, (name, label, _) in enumerate(test_data):
            row = [name, label]
            for bank_name, _ in banks:
                row += [f'{per_image[(bank_name, m)][i]:.4f}' for m in methods]
            w.writerow(row)

    with open('ex02_scoring_summary.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['bank', 'scoring', 'auroc'])
        w.writerows(summary)

    # 分组柱状图：2 库 × 3 打分，画参照线
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(banks))
    width = 0.25
    for j, mthd in enumerate(methods):
        vals = [a for bn, m, a in summary if m == mthd]
        ax.bar(x + (j - 1) * width, vals, width, label=mthd)
    ax.axhline(REF_AUROC_FULL, ls=':', c='gray', label=f'full+top1% = {REF_AUROC_FULL}')
    ax.set_xticks(x)
    ax.set_xticklabels([bn for bn, _ in banks])
    ax.set_ylim(0.8, 1.0)   # 下界留足：max 系 AUROC 可能跌破 0.9，别剪没了
    ax.set_ylabel('image AUROC')
    ax.set_title(f'scoring x bank (b={B_NEIGH})')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('ex02_scoring_compare.png', dpi=150)
    print('图已存 ex02_scoring_compare.png')


if __name__ == '__main__':
    main()
