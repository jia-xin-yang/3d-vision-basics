# ex03_visualize.py
# 阶段 1 任务 3：可视化 —— 亲眼看到 2.5D
# WSL 无显示器：matplotlib 一律 savefig 存 PNG，用 explorer.exe 在 Windows 侧查看
# 图内文字一律英文（WSL matplotlib 默认无中文字体，中文会变方框）

from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')            # 强制非交互后端：只写文件、不弹窗（无显示器环境必须）
import matplotlib.pyplot as plt

ROOT = Path('/mnt/e/datasets/mvtec3d')


def load(cls: str, split: str, defect: str, idx: int):
    base = ROOT / cls / split / defect
    xyz = cv2.imread(str(base / 'xyz' / f'{idx:03d}.tiff'), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(base / 'rgb' / f'{idx:03d}.png'), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)   # OpenCV 读出的是 BGR，转回 RGB 才是人眼看到的颜色
    return xyz, rgb


def main():
    cls, split, defect, idx = 'bagel', 'train', 'good', 0
    xyz, rgb = load(cls, split, defect, idx)

    # ex02 的核心发现在这里直接生效：坏点是 (0,0,0) 不是 NaN
    zero = (xyz == 0).all(axis=2)
    z = xyz[..., 2]
    # masked array：被 mask 的点在 imshow 里自动显示为透明（叠在黑色背景上=黑洞）
    z_m = np.ma.masked_where(zero, z)

    # ---- 图 1：四联张（rgb 照片 / 深度灰度 / 深度伪彩 / 有效点掩码）----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title('rgb photo')
    g1 = axes[1].imshow(z_m, cmap='gray')
    axes[1].set_title('depth z (gray)')
    fig.colorbar(g1, ax=axes[1], fraction=0.046)
    g2 = axes[2].imshow(z_m, cmap='viridis')
    axes[2].set_title('depth z (viridis)')
    fig.colorbar(g2, ax=axes[2], fraction=0.046)
    axes[3].imshow(~zero, cmap='gray')
    axes[3].set_title('valid mask (white=point)')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{cls}/{split}/{defect}/{idx:03d}   zero={int(zero.sum())} '
                 f'({zero.mean() * 100:.1f}%)')
    fig.tight_layout()
    fig.savefig('ex03_bagel_good_panel.png', dpi=150)
    plt.close(fig)

    # ---- 图 2：俯视投影（鸟瞰点云）—— 把有效点摊到 x-y 平面，颜色标高度 z ----
    pts = xyz[~zero]                      # (N, 3) 只保留有效点
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pts[:, 0] * 1000, pts[:, 1] * 1000,
                    c=pts[:, 2] * 1000, s=0.3, cmap='viridis')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')                # x/y 同比例，不然面包圈会被拉成椭圆
    ax.set_title(f'top-down view  N={pts.shape[0]}')
    fig.colorbar(sc, ax=ax, label='z (mm)')
    fig.tight_layout()
    fig.savefig('ex03_bagel_good_topdown.png', dpi=150)
    plt.close(fig)

    print('已生成: ex03_bagel_good_panel.png / ex03_bagel_good_topdown.png')
    print('查看: explorer.exe ex03_bagel_good_panel.png')


if __name__ == '__main__':
    main()
