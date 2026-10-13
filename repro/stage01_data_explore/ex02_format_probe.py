# ex02_format_probe.py
# 阶段 1 任务 2：单张 xyz.tiff 格式探查
# 验证格式假设（shape/dtype/单位/NaN）与组织假设（规则网格）
# 方法核心：对账 —— 用实测数字比对论文/设备规格（点间距 0.37mm）

from pathlib import Path
import numpy as np
import cv2

ROOT = Path('/mnt/e/datasets/mvtec3d')


def probe_one(path: Path) -> dict:
    """读一张 xyz tiff，产出格式报告的全部字段。"""
    # IMREAD_UNCHANGED: 原样读取，禁止 cv2 默认转 8-bit（0.00037m 会被抹成 0）
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'读不到文件: {path}')

    H, W = img.shape[:2]
    # 一个像素只要有任一通道是 NaN，就算无效点
    nan_mask = np.isnan(img).any(axis=2)
    n_valid = int((~nan_mask).sum())

    def spacing(a, b, ma, mb):
        """两组相邻像素的 3D 间距中位数（只统计两侧都有效的对）。"""
        both = ~(ma | mb)               # 两侧都有效才算一对
        if both.sum() == 0:
            return float('nan')
        d = np.linalg.norm((a - b)[both], axis=1)
        return float(np.median(d))

    # 水平相邻（同一行，列+1）与垂直相邻（同一列，行+1）各算一遍
    d_h = spacing(img[:, 1:, :], img[:, :-1, :], nan_mask[:, 1:], nan_mask[:, :-1])
    d_v = spacing(img[1:, :, :], img[:-1, :, :], nan_mask[1:, :], nan_mask[:-1, :])

    return {
        'file': str(path).split('mvtec3d/')[1],
        'shape': img.shape, 'dtype': str(img.dtype),
        'n_nan': int(nan_mask.sum()), 'n_valid': n_valid,
        # 每通道数值范围 → 判断单位（米：值在零点几；毫米：值在几百）
        'range': [(round(float(img[..., c].min()), 4), round(float(img[..., c].max()), 4))
                  for c in range(img.shape[2])],
        'spacing_h_mm': round(d_h * 1000, 3),   # 米 → 毫米，好和 0.37mm 对账
        'spacing_v_mm': round(d_v * 1000, 3),
    }


def main():
    # ---- 第一部分：一张样本的完整格式报告 ----
    sample = ROOT / 'bagel/train/good/xyz/000.tiff'
    r = probe_one(sample)
    print('=== 单张格式报告 ===')
    for k, v in r.items():
        print(f'{k:>14}: {v}')

    # ---- 第二部分：跨类抽查分辨率/NaN —— 各类的网格一样大吗？----
    print('\n=== 跨类抽查（各类取 train/good 第一张）===')
    for cls in ['bagel', 'tire', 'foam', 'rope', 'carrot']:
        p = ROOT / cls / 'train/good/xyz/000.tiff'
        r = probe_one(p)
        print(f"{cls:>12}: shape={r['shape']} nan={r['n_nan']}")

    # ---- 第三部分：GT 对齐抽查（任务 4 的预告）----
    # 缺陷样本的掩码尺寸，和 xyz 网格尺寸一致吗？
    gt = cv2.imread(str(ROOT / 'bagel/test/crack/gt/000.png'), cv2.IMREAD_UNCHANGED)
    xyz = cv2.imread(str(ROOT / 'bagel/test/crack/xyz/000.tiff'), cv2.IMREAD_UNCHANGED)
    print('\n=== GT 对齐抽查 ===')
    print(f'xyz: {xyz.shape}  gt: {gt.shape}  gt.dtype: {gt.dtype}')
    print(f'gt 唯一灰度值: {np.unique(gt)}   ← 对照 class_ids.json 解读')

    # ---- 第四部分：点间距对账 ----
    print('\n=== 点间距对账 ===')
    print('论文/设备规格: Zivid One+ 点间距 ~0.37mm')
    r = probe_one(sample)
    print(f"实测 bagel: 水平 {r['spacing_h_mm']}mm  垂直 {r['spacing_v_mm']}mm")


if __name__ == '__main__':
    main()
