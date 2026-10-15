# ex03_ten_class_run.py
# 阶段 3 任务 3：10 类全量评测 —— 手写迷你 Reg3D-AD 的完整口径
#
# 阶段 2 + 任务 1/2 的全部结论在 bagel 一类上做出，本任务把它们推到 10 类：
#   流水线/类：plane 剥离 → voxel 1mm → 法线(5mm/30,朝相机) → FPFH(10mm/100)
#   建库：train/good 前 30 张（与 bagel 同口径 K=30）
#   采样：0.5% 贪心 k-center coreset（任务 1 冠军档）
#   打分：score_top1%（任务 2 冠军）与裸 max（任务 2 实证崩盘，作对照列保留）
#
# 全交叉：10 类 × {全量库, coreset} × {top1%, max} = 40 格 AUROC。
#   全量库列同时回填任务 1 遗留问题："0.5% coreset 无损"是否在 10 类普遍成立。
#   裸 max 列收集噪声尾巴崩盘在 10 类上的分布（观察题素材）。
#
# 产物：
#   ex03_class_results.csv   每类一行：规模/半径/4 格 AUROC/耗时
#   ex03_image_scores.csv    每图一行：2 库 × 2 打分 共 4 列分数（约 1200 行）
#   ex03_class_auroc.png     分组柱状图

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

STAGE02 = Path(__file__).resolve().parent.parent / 'stage02_btf_baseline'
sys.path.insert(0, str(STAGE02))
from ex02_memory_bank import extract_fpfh   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex01_coreset_bagel import greedy_kcenter, auroc   # noqa: E402

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLASSES = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
           'foam', 'peach', 'potato', 'rope', 'tire']
K = 30               # 建库 train/good 张数：与 bagel 同口径
CORESET_PCT = 0.5    # 任务 1 冠军档
TOP_PCT = 0.01       # score_top1% 的 1%


def collect_jobs(split_dir):
    """汇总某 split 的 (tiff 路径, 标签) 列表：good + 全部缺陷目录。"""
    jobs = [(fp, 'good') for fp in sorted((split_dir / 'good' / 'xyz').glob('*.tiff'))]
    if split_dir.name == 'test':
        for d in sorted(split_dir.iterdir()):
            if d.is_dir() and d.name != 'good':
                jobs += [(fp, d.name) for fp in sorted((d / 'xyz').glob('*.tiff'))]
    return jobs


def extract_all(jobs, tag):
    """逐张提 FPFH。返回 (成功列表, 跳过列表)。
    个别图平面剥离后物体点数为 0（如细杆类某些角度），估法线会抛异常——
    跳过并记录，不动流水线本身（与 bagel 冠军口径逐字一致）。"""
    out, skipped = [], []
    for i, (fp, label) in enumerate(jobs):
        xyz_img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
        try:
            _, feat = extract_fpfh(xyz_img)
            if len(feat) == 0:
                raise ValueError('empty feature set')
        except Exception as e:   # Open3D 对空点云会抛 RuntimeError
            skipped.append((tag, f'{label}/{fp.stem}', str(e)[:60]))
            print(f'    !! 跳过 {tag}/{label}/{fp.stem}: {str(e)[:50]}')
            continue
        out.append((f'{label}/{fp.stem}', label, feat))
        if (i + 1) % 40 == 0:
            print(f'    [{tag}][{i + 1}/{len(jobs)}] ...')
    return out, skipped


def main():
    class_rows = []
    img_rows = []
    all_skipped = []
    for cls in CLASSES:
        t0 = time.time()
        # ① 建库：train/good 前 K 张
        train_jobs = collect_jobs(ROOT / cls / 'train')[:K]
        train_feats, sk = extract_all(train_jobs, f'{cls}/train')
        all_skipped += sk
        if not train_feats:
            print(f'  {cls}: 全部 train 图被跳过，放弃本类')
            continue
        bank = np.vstack([f for _, _, f in train_feats])
        # ② 0.5% 贪心 coreset
        sel, radius = greedy_kcenter(bank, max(1, int(len(bank) * CORESET_PCT / 100)))
        core = bank[sel]
        # ③ 两棵树：全量 / coreset
        tree_full = cKDTree(bank)
        tree_core = cKDTree(core)
        # ④ 测试集评分
        test_jobs = collect_jobs(ROOT / cls / 'test')
        test_feats, sk = extract_all(test_jobs, f'{cls}/test')
        all_skipped += sk
        s_tf, s_tm, s_cf, s_cm, labels = [], [], [], [], []
        for j, (name, label, feat) in enumerate(test_feats):
            # workers=-1：多核并行精确搜索，结果与单核逐位一致，速度数倍提升
            d_f, _ = tree_full.query(feat, k=1, workers=-1)
            d_c, _ = tree_core.query(feat, k=1, workers=-1)
            m = max(1, int(len(d_f) * TOP_PCT))
            s_tf.append(float(np.mean(np.sort(d_f)[-m:])))
            s_tm.append(float(d_f.max()))
            s_cf.append(float(np.mean(np.sort(d_c)[-m:])))
            s_cm.append(float(d_c.max()))
            labels.append(1 if label != 'good' else 0)
            img_rows.append((cls, name, label, s_tf[-1], s_tm[-1],
                             s_cf[-1], s_cm[-1]))
            if (j + 1) % 10 == 0:
                print(f'    [{cls}/test 评分][{j + 1}/{len(test_feats)}] ...')
        a_tf = auroc(s_tf, labels)
        a_tm = auroc(s_tm, labels)
        a_cf = auroc(s_cf, labels)
        a_cm = auroc(s_cm, labels)
        dt = time.time() - t0
        class_rows.append((cls, len(train_jobs), len(bank), len(core),
                           radius, len(test_jobs),
                           a_tf, a_cf, a_tm, a_cm, dt))
        print(f'  {cls:>12s}: bank {len(bank):>7d} → coreset {len(core):>5d} '
              f'(r={radius:5.1f}), test {len(test_jobs):>3d} 张, '
              f'top1% full={a_tf:.4f} core={a_cf:.4f} | '
              f'max full={a_tm:.4f} core={a_cm:.4f}  [{dt:.0f}s]')

    with open('ex03_skipped_images.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['split', 'image', 'reason'])
        w.writerows(all_skipped)
    print(f'跳过图像共 {len(all_skipped)} 张，清单见 ex03_skipped_images.csv')

    with open('ex03_class_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'n_train', 'bank_pts', 'coreset_pts', 'cover_radius',
                    'n_test', 'auroc_full_top1pct', 'auroc_coreset_top1pct',
                    'auroc_full_max', 'auroc_coreset_max', 'seconds'])
        w.writerows(class_rows)
        m = np.mean([[r[6], r[7], r[8], r[9]] for r in class_rows], axis=0)
        w.writerow(['MEAN'] + [''] * 5 + [f'{v:.4f}' for v in m] + [''])

    with open('ex03_image_scores.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'image', 'label', 'full_top1pct', 'full_max',
                    'coreset_top1pct', 'coreset_max'])
        w.writerows(img_rows)

    # 分组柱状图：每类 4 根（full/core × top1/max），带数值标签
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(class_rows))
    w = 0.2
    series = [(6, 'full+top1%', '#1f77b4'), (7, 'coreset+top1%', '#2ca02c'),
              (8, 'full+max', '#d62728'), (9, 'coreset+max', '#ff7f0e')]
    for j, (idx, name, color) in enumerate(series):
        xs = [i + (j - 1.5) * w for i in x]
        ys = [r[idx] for r in class_rows]
        ax.bar(xs, ys, w, label=name, color=color)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.008, f'{yi:.2f}', ha='center', fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in class_rows], rotation=20)
    ax.set_ylim(0.4, 1.1)   # 下界留足：dowel coreset 档 0.43，别剪没了
    ax.set_ylabel('image AUROC')
    ax.set_title('10 classes x {full, coreset 0.5%} x {top1%, max}')
    ax.legend(ncol=2)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('ex03_class_auroc.png', dpi=150)
    print('图已存 ex03_class_auroc.png')


if __name__ == '__main__':
    main()
