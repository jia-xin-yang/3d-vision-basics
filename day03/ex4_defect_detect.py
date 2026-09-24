import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
T_LOW, T_HIGH = -0.05, 0.08   # 划痕/凸起判定阈值 (mm)
AREA_NG = 0.1                 # 面积超过 0.1 mm^2 即判 NG

def detect(H, res, x0, y0, label):
    """双阈值缺陷检测 + 统计。返回 (scratch_mask, bump_mask, stats)。"""
    valid = ~np.isnan(H)
    missing = 1.0 - valid.mean()
    # NaN 跟任何数比较都得 False，所以 (H < T) 天然跳过空格；
    # 这里仍显式 & valid，把"排除缺失"的意图写进代码，防止后人误改
    scratch = (H < T_LOW) & valid
    bump = (H > T_HIGH) & valid
    print(f"--- {label} ---")
    print(f"缺失率: {missing * 100:.1f}%  (空格 {int((~valid).sum())} / {H.size})")
    stats = {}
    for name, mask, take_min in [("划痕", scratch, True), ("凸起", bump, False)]:
        n = int(mask.sum())
        if n == 0:
            print(f"{name}: 未检出")
            stats[name] = (0, 0.0, np.nan)
            continue
        area = n * res * res
        zs = H[mask]
        extreme = zs.min() if take_min else zs.max()
        iy, ix = np.argwhere(mask).mean(axis=0)      # 质心（像素坐标）
        xc, yc = x0 + ix * res, y0 + iy * res        # 换算回 mm（左上角约定，半像素误差不影响结论）
        word = "最深" if take_min else "最高"
        print(f"{name}: {n} 像素, 面积 {area:.3f} mm^2, {word} {extreme:.3f} mm, 质心 ({xc:.2f}, {yc:.2f}) mm")
        stats[name] = (n, area, extreme)
    return scratch, bump, stats

# ===== Part A: 全板（有组织点云 reshape，无缺失）=====
raw = np.load(OUT / "plate.npy").reshape(200, 200, 3)
H_A = raw[:, :, 2]
res_A = 10.0 / 199                 # 实际点距 mm/像素（0~10mm 均分 199 个间隔）
sA, bA, stA = detect(H_A, res_A, 0.0, 0.0, "Part A 全板 200×200")

# ===== Part B: ROI 工作版（散点 mean binning，有缺失）=====
H_B = np.load(OUT / "heightmap.npy")          # 练习3 的 60×60 产物，NaN=空格
res_B = 0.1
x0 = y0 = 2.01                                 # clean 点云的最小坐标（练习3 算过）
sB, bB, stB = detect(H_B, res_B, x0, y0, "Part B ROI 60×60")

# ===== 判定：任一划痕/凸起面积超 AREA_NG 即 NG =====
worst = max(stA["划痕"][1], stA["凸起"][1], stB["划痕"][1], stB["凸起"][1])
print(f"\n最大缺陷面积: {worst:.3f} mm^2  (NG 线 {AREA_NG} mm^2)")
print("整板判定:", "NG（不合格）" if worst > AREA_NG else "PASS")

# ===== 可视化：高度图 + 缺陷叠加 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, H, s, b, t in [(axes[0], H_A, sA, bA, "A: full plate (organized)"),
                       (axes[1], H_B, sB, bB, "B: ROI (scattered->binning)")]:
    im = ax.imshow(H, cmap="viridis", vmin=-0.1, vmax=0.3)
    mark = np.zeros(H.shape + (4,))   # RGBA 叠加层
    mark[s] = (1, 0, 0, 0.9)          # 划痕 = 红
    mark[b] = (1, 1, 0, 0.9)          # 凸起 = 黄
    ax.imshow(mark)
    ax.set_title(t)
    fig.colorbar(im, ax=ax, label="z / mm")
plt.tight_layout()
plt.savefig(OUT / "ex4_defect.png", dpi=150)
print("已保存 ex4_defect.png")