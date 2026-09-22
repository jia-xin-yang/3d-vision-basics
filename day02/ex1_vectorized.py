# day02/ex1_vectorized.py
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# ===== 以下和 ex1 完全相同，直接抄过来 =====
img = cv2.imread('/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg', cv2.IMREAD_GRAYSCALE)   # 用 ex1 里那张图
print("尺寸:", img.shape, "dtype:", img.dtype)

kernel = np.ones((3, 3), dtype=np.float32) / 9.0
pad = np.pad(img.astype(np.float32), pad_width=1, mode='constant')
H, W = img.shape

# ===== 循环版：留作计时基准和对比基准 =====
t0 = time.perf_counter()                                # 计时起点
out_loop = np.zeros((H, W), dtype=np.float32)
for i in range(H):
    for j in range(W):
        window = pad[i:i+3, j:j+3]
        out_loop[i, j] = np.sum(window * kernel)
t_loop = time.perf_counter() - t0                       # 循环版耗时

# ===== 向量化版：两步替代双重循环 =====
t0 = time.perf_counter()
windows = sliding_window_view(pad, (3, 3))              # ① 取窗口（视图，不复制）
out_vec = np.einsum('ijkl,kl->ij', windows, kernel)     # ② 乘加归约（一行）
t_vec = time.perf_counter() - t0                        # 向量化耗时

# ===== 验证三连 =====
print("windows shape:", windows.shape)                  # 期望 (H, W, 3, 3)
print("base 是 pad 吗:", windows.base is pad)           # 期望 True → 没复制内存
print("浮点级一致:", np.allclose(out_vec, out_loop))    # 期望 True
print(f"循环 {t_loop*1000:.1f} ms | 向量化 {t_vec*1000:.2f} ms | 加速 {t_loop/t_vec:.0f}×")

# ===== 和 cv2 对答案（同 ex1 流程，拷过来即可）=====
out_u8 = np.clip(out_vec, 0, 255).astype(np.uint8)
ref = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
diff = np.abs(out_u8.astype(np.int32) - ref.astype(np.int32))
print("diff max =", diff.max(), " 超差(>2) =", int((diff > 2).sum()))
print("numpy 版本:", np.__version__)
print("shares_memory:", np.shares_memory(windows, pad))   # 权威判据：是否共享同一块内存
print("数据指针相同:", windows.ctypes.data == pad.ctypes.data)
print("base 的类型:", type(windows.base))