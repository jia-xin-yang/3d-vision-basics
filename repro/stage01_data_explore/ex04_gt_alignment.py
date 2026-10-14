# ex04_gt_alignment.py
# 阶段 1 任务 4：GT 掩码对齐与统计
# 回答三个问题：
#   ① GT 灰度编码的语义（class_ids.json）+ combined 目录之谜
#   ② GT 掩码和 rgb/xyz 网格是否真对齐（轮廓叠加亲眼验证）
#   ③ 缺陷像素占比多少（"罕见事件"是异常检测方法论的根源）

from pathlib import Path
import numpy as np
import cv2
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/e/datasets/mvtec3d')
OUT_CSV = Path('ex04_defect_ratio.csv')


def load_gt(cls, defect, idx):
    p = ROOT / cls / 'test' / defect / 'gt' / f'{idx:03d}.png'
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def report_one(cls, defect, idx):
    """单张 GT 的编码解读：唯一灰度值 → 缺陷名（class_ids.json）→ 占比。"""
    gt = load_gt(cls, defect, idx)
    name_map = json.loads((ROOT / cls / 'class_ids.json').read_text())
    uniq = np.unique(gt)
    n_defect = int((gt > 0).sum())
    parts = []
    for v in uniq:
        label = name_map.get(str(int(v)), f'未登记:{int(v)}')
        parts.append(f'{int(v)}={label}')
    print(f'{cls}/test/{defect}/{idx:03d}.png  unique[{", ".join(parts)}]')
    print(f'    缺陷像素 {n_defect} / {gt.size} = {n_defect / gt.size * 100:.2f}%')
    return gt


def overlay_figure(cls, defect, idx):
    """三联张：rgb 照片 / 深度图 / rgb+GT轮廓叠加 —— 对齐与否一眼可见。"""
    base = ROOT / cls / 'test' / defect
    rgb = cv2.imread(str(base / 'rgb' / f'{idx:03d}.png'), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    xyz = cv2.imread(str(base / 'xyz' / f'{idx:03d}.tiff'), cv2.IMREAD_UNCHANGED)
    gt = load_gt(cls, defect, idx)
    zero = (xyz == 0).all(axis=2)
    z = np.ma.masked_where(zero, xyz[..., 2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title('rgb photo')
    g1 = axes[1].imshow(z, cmap='viridis')
    axes[1].set_title('depth z')
    fig.colorbar(g1, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb)
    # GT 轮廓线叠在 rgb 上：轮廓应"咬住"照片上肉眼可见的缺陷
    axes[2].contour(gt > 0, levels=[0.5], colors='red', linewidths=1.2)
    axes[2].set_title('rgb + GT contour')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{cls}/test/{defect}/{idx:03d}')
    fig.tight_layout()
    out = f'ex04_{cls}_{defect}_overlay.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'    叠加图已存: {out}')


def defect_ratio_all():
    """全类缺陷占比统计：每类遍历其全部 defect 的 gt（948 张，秒级）。"""
    rows = []
    for cls_dir in sorted(ROOT.iterdir()):
        if not cls_dir.is_dir() or not (cls_dir / 'test').is_dir():
            continue
        if not (cls_dir / 'train').is_dir():      # 语义过滤，跳过 MVTEC_3D-AD 残留
            continue
        ratios = []
        for defect_dir in sorted((cls_dir / 'test').iterdir()):
            if defect_dir.name == 'good':
                continue
            for gt_p in sorted(defect_dir.glob('gt/*.png')):
                gt = cv2.imread(str(gt_p), cv2.IMREAD_UNCHANGED)
                if gt is None:
                    continue
                ratios.append(float((gt > 0).mean()))
        if ratios:
            rows.append((cls_dir.name, len(ratios),
                         float(np.mean(ratios)) * 100, float(np.max(ratios)) * 100))
            print(f"{cls_dir.name:>12}: {len(ratios):>3} 张缺陷 GT  "
                  f"平均缺陷占比 {np.mean(ratios) * 100:5.2f}%  最大 {np.max(ratios) * 100:5.2f}%")
    all_mean = np.mean([r[2] for r in rows])
    print(f"\n10 类平均缺陷占比: {all_mean:.2f}%")
    with OUT_CSV.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'n_defect_gt', 'mean_defect_ratio_pct', 'max_defect_ratio_pct'])
        for r in rows:
            w.writerow(r)
    print(f'CSV 已写入: {OUT_CSV.resolve()}')


def main():
    print('=== ① 编码解读 ===')
    report_one('bagel', 'crack', 0)        # 单缺陷：预期 unique = [0, 254]
    report_one('bagel', 'combined', 0)     # combined：任务 1 的钩子，看 unique 里有几种灰度

    print('\n=== ② 对齐验证（轮廓叠加）===')
    overlay_figure('bagel', 'crack', 0)

    print('\n=== ③ 全类缺陷占比 ===')
    defect_ratio_all()


if __name__ == '__main__':
    main()
