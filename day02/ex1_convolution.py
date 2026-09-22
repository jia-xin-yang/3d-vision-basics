import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读灰度图
img = cv2.imread('/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg',
                 cv2.IMREAD_GRAYSCALE)
assert img is not None, "图片没读到，检查路径"
print("尺寸:", img.shape, "dtype:", img.dtype)

# 2. 均值核
kernel = np.ones((3, 3), dtype=np.float32) / 9.0

# 3. 转 float + 边缘补一圈零
pad = np.pad(img.astype(np.float32), pad_width=1, mode='constant')

# 4. 手写卷积 —— 核心部分，自己写
H, W = img.shape
out = np.zeros((H, W), dtype=np.float32)
for i in range(H):
    for j in range(W):
       window = pad[i:i+3, j:j+3]
       out[i, j] = np.sum(window * kernel)

# 5. 转回 uint8
out_u8 = np.clip(out, 0, 255).astype(np.uint8)

# 6. 和 cv2 对比（注意 borderType 指定 BORDER_CONSTANT，对应你的补零）
ref = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
print("完全一致:", np.allclose(out_u8, ref, atol=1))
diff = np.abs(out_u8.astype(np.int32) - ref.astype(np.int32))

# 7. 存四联对比图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, im, t in zip(axes, [img, out_u8, ref, diff],
                     ['original', 'my conv', 'cv2.filter2D', 'diff']):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
plt.savefig('ex1_result.png', dpi=100, bbox_inches='tight')
print("已保存 ex1_result.png")
print("diff max =", diff.max(), "  超差(>2)像素 =", int((diff > 2).sum()), "/", diff.size)