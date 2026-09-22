# day02/ex2_edge.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---- 1. 读图（路径从 ex1 抄，还是那张灰度图）----
img = cv2.imread('/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg', cv2.IMREAD_GRAYSCALE)
print("尺寸:", img.shape, "dtype:", img.dtype)

# ---- 2. 高斯模糊 ----
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# ---- 3. Canny 边缘检测 ----
edges = cv2.Canny(blurred, 50, 150)

# ---- 4. 三拼图：原图 | 模糊图 | 边缘图 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, im, t in zip(axes, [img, blurred, edges],
                     ['original', 'gaussian blurred', 'canny edges']):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')                        # 关掉坐标刻度，图更干净
plt.savefig('ex2_result.png', dpi=100, bbox_inches='tight')
print("已保存 ex2_result.png")

# ---- 5. 边缘像素统计 ----
print("边缘像素:", np.count_nonzero(edges), "/", edges.size)

# ---- 6. Q2 实验：去掉高斯，直接对原图 Canny ----
edges_no_blur = cv2.Canny(img, 50, 150)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, im, t in zip(axes, [edges, edges_no_blur],
                     ['with gaussian', 'without gaussian']):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')
plt.savefig('ex2_q2_compare.png', dpi=100, bbox_inches='tight')
print("无模糊边缘像素:", np.count_nonzero(edges_no_blur), "/", edges_no_blur.size)