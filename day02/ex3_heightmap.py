# day02/ex3_heightmap.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---- 1. 读图 + 转成 float，模拟真实高度图的数据形态 ----
img = cv2.imread('/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg',
                 cv2.IMREAD_GRAYSCALE)
hmap = img.astype(np.float32)      # uint8 → float32：真实高度图都是浮点毫米值，先习惯数据形态
print("尺寸:", hmap.shape, "dtype:", hmap.dtype, "范围:", hmap.min(), "-", hmap.max())

# ---- 2. 四种 colormap 对比 + colorbar ----
cmaps = ['gray', 'viridis', 'jet', 'turbo']
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for ax, cm in zip(axes, cmaps):
    im = ax.imshow(hmap, cmap=cm)                  # cmap 只影响渲染配色，不改数据
    ax.set_title(f'cmap: {cm}')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046)        # 每个子图配一根颜色刻度条
plt.savefig('ex3_cmaps.png', dpi=100, bbox_inches='tight')
print("已保存 ex3_cmaps.png")

# ---- 3. Q2 实验：vmin/vmax 只改显示，不改数据 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im0 = axes[0].imshow(hmap, cmap='viridis', vmin=0, vmax=255)
axes[0].set_title('vmax=255 (全范围)')
fig.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(hmap, cmap='viridis', vmin=0, vmax=100)
axes[1].set_title('vmax=100 (压缩显示范围)')
fig.colorbar(im1, ax=axes[1], fraction=0.046)
for ax in axes:
    ax.axis('off')
plt.savefig('ex3_vmin_vmax.png', dpi=100, bbox_inches='tight')
print("保存后数据范围仍是:", hmap.min(), "-", hmap.max(), "（数组没动过）")

# ---- 4. 挑战（选做）：3D 曲面视角看高度图 ----
step = 8                                   # 每 8 像素取 1 个 → 60×80 网格（480×640 全画=30万点会卡死）
small = hmap[::step, ::step]
ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]]   # 生成网格坐标
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, small, cmap='viridis', linewidth=0, antialiased=False)
ax.set_xlabel('x (pixel)'); ax.set_ylabel('y (pixel)'); ax.set_zlabel('height')
ax.set_title('2.5D surface view (downsampled 8x)')
plt.savefig('ex3_surface.png', dpi=100, bbox_inches='tight')
print("已保存 ex3_surface.png")