import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent

# ===== Part A: 有组织点云 → 高度图（reshape 即可）====
raw = np.load(OUT / "plate.npy")          # (400, 3)
grid = raw.reshape(200, 200, 3)           # 点本来就是按 200×200 网格逐行生成的
height_org = grid[:, :, 2]                # 高度图 = z 通道
print("Part A 高度图形状:", height_org.shape)   # (200, 200)，无空洞

# ===== Part B: 散点 → 高度图（mean bining）====
clean = np.load(OUT / "clean_pts.npy")    # 练习2 的流水线产出
print("清洗后点数:", len(clean))

res = 0.1                                  # 分辨率 mm/像素——和降采样后的点距匹配
x_min, y_min = clean[:, 0].min(), clean[:, 1].min()
nx = int(np.floor((clean[:, 0].max() - x_min) / res)) + 1
ny = int(np.floor((clean[:, 1].max() - y_min) / res)) + 1

ix = ((clean[:, 0] - x_min) / res).astype(int)   # 每个点属于哪列
iy = ((clean[:, 1] - y_min) / res).astype(int)   # 每个点属于哪行

# bincount 数每格点数 + 累每格 z 总和（压成一维再 reshape 回二维）
idx = iy * nx + ix
cnt = np.bincount(idx, minlength=nx * ny).reshape(ny, nx)
zsum = np.bincount(idx, weights=clean[:, 2], minlength=nx * ny).reshape(ny, nx)

height = np.full((ny, nx), np.nan)        # 先全填 NaN
has = cnt > 0
height[has] = zsum[has] / cnt[has]        # 有点的格子取均值
print(f"Part B 高度图: {ny}×{nx} 像素, 空格占比 {np.mean(~has) * 100:.1f}%")

np.save(OUT / "heightmap.npy", height.astype(np.float32))   # NaN 会原样保留

# ===== Part C: 并排对比 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, h, t in [(axes[0], height_org, "A: organized -> reshape"),
                 (axes[1], height,     "B: scattered -> mean bining")]:
    sc = ax.imshow(h, cmap="viridis", vmin=-0.1, vmax=0.3)   # 同一色域才能对比
    ax.set_title(t)
    fig.colorbar(sc, ax=ax, label="z / mm")
plt.tight_layout()
plt.savefig(OUT / "ex3_heightmap.png", dpi=150)
print("已保存 ex3_heightmap.png")