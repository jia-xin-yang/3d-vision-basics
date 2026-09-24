# 3d-vision-basics

Learning repo: industrial 3D vision inspection + edge AI + embodied perception.
Stage 1: 3D vision basics —— 从 2D 图像基础到 3D 点云质检最小闭环。

带练进度：Day 1 环境搭建 ✅ → Day 2 数字图像基础 ✅ → Day 3 点云质检流水线 ✅ → Day 4 真实数据集（待开工）

## 目录结构

### `day02/` — 数字图像基础（OpenCV / NumPy / Matplotlib）

- `ex1_convolution.py` 手写卷积 + 诊断行；`ex1_vectorized.py` 向量化加速版（sliding_window_view + einsum，~147×）
- `ex2_edge.py` Canny 边缘提取 + 高低阈值对照实验
- `ex3_heightmap.py` 高度图可视化（伪彩色 / vmin-vmax 对照 / 3D 曲面）
- `ex4_fileio.py` 图像文件 IO（JPEG/PNG/npy 保真对比 + 标注画框）
- `ex5_homogeneous.py` 附加题：2D 齐次坐标刚体变换（直通 3D 4×4 外参）
- `ex*_*.png` 各练习交付图（入库）；`ex4_output/` 运行产物（可再生，已 gitignore）

### `day03/` — 点云质检全流程（Open3D + NumPy）

- `ex1_formats.py` 点云本质 + 4 种格式读写（npy/pcd/ply/xyz）；生成合成金属板 `plate.*`（seed 42，确定性可复现）
- `make_tiny_pcd.py` 生成 tiny.pcd，用于 pcd header 逐行解剖
- `ex2_pipeline.py` 预处理流水线：ROI 裁剪 → 体素降采样(0.1mm) → 统计滤波(k=20)，落盘 `clean_pts.npy`（40800→14475→3810→3735）
- `ex3_heightmap.py` 点云→高度图双路线：有组织 reshape vs 散点 mean binning（60×60，空格=NaN）
- `ex4_defect_detect.py` 双阈值缺陷检测（划痕 z<-0.05 / 凸起 z>0.08），像素→mm² 换算 + NG 判定
- `ex*_*.png` 交付图（入库）
- 数据产物（`plate.*` / `clean_pts.npy` / `heightmap.npy`，脚本可再生）已 gitignore

## Day 3 复跑顺序（依赖链，缺产物就从第一步补）

```bash
conda activate industrial3d
cd day03
python ex1_formats.py        # 生成 plate.{npy,pcd,ply,xyz}
python ex2_pipeline.py       # plate.pcd → clean_pts.npy
python ex3_heightmap.py      # clean_pts.npy → heightmap.npy
python ex4_defect_detect.py  # plate.npy + heightmap.npy → ex4_defect.png
```

## 运行环境

WSL2 Ubuntu + conda 环境 `industrial3d`（Python 3.10 / PyTorch 2.1.2+cu121 / Open3D 0.18.0 / OpenCV 4.8.1 / NumPy 1.24.3 / Matplotlib 3.7.2）。

WSL 无显示器：所有图一律 `plt.savefig()`，用 `explorer.exe xxx.png` 查看。
