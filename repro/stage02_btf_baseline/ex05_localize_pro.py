# ex05_localize_pro.py
# 阶段 2 任务 5（收官）：逐点定位 + 像素级评测 + validation 阈值标定
#
# 任务 4 Q3 欠的两笔债，本任务一起还：
#   ① "缺陷在哪"：逐点 kNN 距离画回原图（定位图），像素级指标验收
#   ② "线画在哪"：validation/good（22 张）标定判定阈值，test 出混淆矩阵
#
# 三部分：
#   A. 全测试集逐像素打分：物体像素 → 最近体素质心的 kNN 距离
#      → 每图像素级 AUROC + PRO（等权逐连通区召回，anomalib 惯例）
#   B. 定位图 2×3：rgb + GT 轮廓(绿) + 分数热力图
#   C. validation 标定阈值 → test 混淆矩阵（recall / precision / F1）

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

from ex01_fpfh_first_contact import plane_from_ring
from ex02_memory_bank import extract_fpfh

ROOT = Path('/mnt/e/datasets/mvtec3d')
CLS = 'bagel'
OBJ_THRESH = 0.010
TOP_PCT = 0.01
FPR_RANGE = 0.3          # PRO 在 FPR∈[0, 0.3] 上积分（MVTec 协议）

# 定位图选样：4 种缺陷各 1 张 + crack/combined 各加 1 张
PICKS = [('crack', 0), ('contamination', 0), ('hole', 0),
         ('combined', 0), ('crack', 1), ('combined', 1)]


def auroc(scores, labels):
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    labels = np.asarray(labels)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def compute_pro(gt_mask, scored_mask, score_img):
    """PRO：逐 GT 连通区算召回 → 等权平均 → 在 FPR∈[0,0.3] 积分。
    等权 = 大小缺陷区同票，逼模型把每个区找全（区别于像素召回的按面积多数派）。"""
    pos = gt_mask & scored_mask
    neg = ~gt_mask & scored_mask
    n_comp, comp_lab = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    if n_comp <= 1 or pos.sum() == 0:
        return float('nan')
    sp = score_img[scored_mask]
    po = pos[scored_mask]
    ne = neg[scored_mask]
    comp_pos = comp_lab[pos]                    # 每个缺陷像素的连通区编号（scored 顺序）
    comp_size = np.bincount(comp_pos)           # 各区面积（索引 0 = 背景，恒 0）
    comp_all = np.full(sp.shape, -1)            # 扩展回 scored 顺序：非缺陷 = -1
    comp_all[po] = comp_pos
    order = np.argsort(-sp)                     # 全部已打分像素，分数降序
    tp_cum = np.cumsum(po[order])
    fp_cum = np.cumsum(ne[order])
    comp_ranked = comp_all[order]               # 排名序列对应的连通区编号
    pos_rank = np.where(comp_ranked >= 0)[0]    # 缺陷像素在排名里的位置
    idx_grid = np.linspace(0, len(sp) - 1, 100).astype(int)
    recalls, fprs = [], []
    for i in idx_grid:
        sel = pos_rank <= i                     # 阈值线切到的缺陷像素
        hit = np.bincount(comp_ranked[pos_rank[sel]], minlength=len(comp_size))
        recalls.append(float(np.mean(hit[1:] / np.maximum(comp_size[1:], 1))))
        fprs.append(float(fp_cum[i] / max(ne.sum(), 1)))
    fprs, recalls = np.array(fprs), np.array(recalls)
    ok = fprs <= FPR_RANGE
    if ok.sum() < 2:
        return float('nan')
    return float(np.trapz(recalls[ok], fprs[ok]) / FPR_RANGE)


def pixel_score_img(xyz_img, q, d):
    """体素点分数 → 像素网格：每个物体像素取最近体素质心的分数。"""
    valid = ~(xyz_img == 0).all(axis=2)
    (a, b, c), _, _ = plane_from_ring(xyz_img)
    denom = np.sqrt(a * a + b * b + 1.0)
    pts = xyz_img[valid].astype(np.float64)
    signed = (pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) / denom
    obj_flat = signed > OBJ_THRESH
    _, nn = cKDTree(q).query(pts[obj_flat], k=1)
    score_flat = np.full(len(pts), np.nan)
    score_flat[obj_flat] = d[nn]
    score_img = np.full(xyz_img.shape[:2], np.nan)
    score_img[valid] = score_flat
    return score_img, ~np.isnan(score_img)


def eval_one(fp, split, label, bank_tree):
    """单张图：特征提取 + 逐点分数 + 逐像素映射 + 像素级指标。"""
    xyz_img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
    q, feat = extract_fpfh(xyz_img)
    d, _ = bank_tree.query(feat, k=1)
    m = max(1, int(len(d) * TOP_PCT))
    img_score = float(np.mean(np.sort(d)[-m:]))
    score_img, scored = pixel_score_img(xyz_img, q, d)
    gt = None
    if split == 'test' and label != 'good':
        gt = cv2.imread(str(ROOT / CLS / 'test' / label / 'gt' / f'{fp.stem}.png'),
                        cv2.IMREAD_UNCHANGED)
        p_auroc = auroc(score_img[scored], (gt > 0)[scored].astype(int))
        pro = compute_pro(gt > 0, scored, score_img)
    else:
        p_auroc, pro = float('nan'), float('nan')
    return dict(name=f'{label}/{fp.stem}', label=label, img_score=img_score,
                pixel_auroc=p_auroc, pro=pro, score_img=score_img, gt=gt)


def jobs_of(split):
    base = ROOT / CLS / split
    jobs = [(fp, 'good') for fp in sorted((base / 'good' / 'xyz').glob('*.tiff'))]
    if split == 'test':
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name != 'good':
                jobs += [(fp, d.name) for fp in sorted((d / 'xyz').glob('*.tiff'))]
    return jobs


def main():
    bank = np.load('ex02_memory_bank_bagel.npy')
    bank_tree = cKDTree(bank)

    print('=== A. 全测试集逐像素评测（110 张）===')
    rows, t0 = [], time.time()
    for i, (fp, label) in enumerate(jobs_of('test')):
        keep_img = any(fp.stem == f'{idx:03d}' and label == d for d, idx in PICKS)
        r = eval_one(fp, 'test', label, bank_tree)
        if not keep_img:
            r['score_img'] = None                       # 大图不留，省内存
        rows.append(r)
        if (i + 1) % 20 == 0:
            print(f'  [{i + 1}/110] {time.time() - t0:.0f}s')

    defect = [r for r in rows if r['label'] != 'good']
    good = [r for r in rows if r['label'] == 'good']
    labels = [1 if r['label'] != 'good' else 0 for r in rows]
    print(f'图像级 AUROC (top1%): {auroc([r["img_score"] for r in rows], labels):.4f}  ← 任务 4 对照')
    print(f'像素级 AUROC (88 张缺陷图宏观平均): {np.nanmean([r["pixel_auroc"] for r in defect]):.4f}')
    print(f'PRO (FPR≤0.3, 等权逐区): {np.nanmean([r["pro"] for r in defect]):.4f}')

    with open('ex05_pixel_eval.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['image', 'label', 'img_score_top1pct', 'pixel_auroc', 'pro'])
        for r in rows:
            w.writerow([r['name'], r['label'], f'{r["img_score"]:.3f}',
                        f'{r["pixel_auroc"]:.4f}', f'{r["pro"]:.4f}'])

    print('\n=== B. 定位图 2×3（绿线=GT 轮廓，热力=逐点分数）===')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (defect_name, idx) in zip(axes.flat, PICKS):
        r = next(x for x in rows if x['name'] == f'{defect_name}/{idx:03d}')
        rgb = cv2.imread(str(ROOT / CLS / 'test' / defect_name / 'rgb' /
                             f'{idx:03d}.png'), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb, alpha=0.45)
        im = ax.imshow(np.ma.masked_invalid(r['score_img']), cmap='hot', alpha=0.75)
        ax.contour(r['gt'] > 0, levels=[0.5], colors='lime', linewidths=1.2)
        ax.set_title(f'{defect_name}/{idx:03d}  PA={r["pixel_auroc"]:.3f}  '
                     f'PRO={r["pro"]:.3f}', fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig('ex05_localization_panel.png', dpi=140)
    print('定位图已存 ex05_localization_panel.png')

    print('\n=== C. validation 标定阈值 → test 混淆矩阵 ===')
    val_rows = [eval_one(fp, 'validation', 'good', bank_tree)
                for fp, _ in jobs_of('validation')]
    val_scores = [r['img_score'] for r in val_rows]
    thr = max(val_scores)
    print(f'val/good {len(val_scores)} 张，top1% 分数 [{min(val_scores):.2f}, {thr:.2f}]')
    print(f'阈值 = val 最大分 {thr:.2f} → 期望 FPR ≈ 1/{len(val_scores)} '
          f'= {100 / len(val_scores):.1f}%（22 张 = 标定分辨率极限）')
    tp = sum(1 for r in defect if r['img_score'] > thr)
    fn = len(defect) - tp
    fp_ = sum(1 for r in good if r['img_score'] > thr)
    tn = len(good) - fp_
    recall = tp / (tp + fn)
    precision = tp / max(tp + fp_, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f'test 混淆矩阵: TP={tp} FP={fp_} FN={fn} TN={tn}')
    print(f'recall={recall:.3f}  precision={precision:.3f}  F1={f1:.3f}')


if __name__ == '__main__':
    main()
