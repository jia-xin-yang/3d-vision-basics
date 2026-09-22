# 3d-vision-basics

Learning repo: industrial 3D vision inspection + edge AI + embodied perception. Stage 1: 3D vision basics.

## 目录结构

- `day02/` — 数字图像基础练习（OpenCV / NumPy / Matplotlib）
  - `ex1_convolution.py` 手写卷积 + 向量化加速版 `ex1_vectorized.py`
  - `ex2_edge.py` Canny 边缘提取 + 滞后阈值对照实验
  - `ex3_heightmap.py` 高度图可视化（伪彩色 / vmin-vmax / 3D 曲面）
  - `ex4_fileio.py` 图像文件 IO（JPEG/PNG/npy 对比 + YOLO 标注画框）
  - `ex*_*.png` 各练习结果图（交付图，入库）
  - `ex4_output/` 练习 4 运行产物（可再生，已 gitignore）

运行环境：WSL2 Ubuntu + conda 环境 `industrial3d`（Python 3.10, OpenCV, NumPy, Matplotlib）。
