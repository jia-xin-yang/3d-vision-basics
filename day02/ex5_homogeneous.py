# day02/ex5_homogeneous.py
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG = '/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg'
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
H, W = img.shape

# ---- 1. 构造 2D 刚体变换矩阵 T = [R | t; 0 1] ----
theta = np.deg2rad(30)                    # 弧度制：np 的 sin/cos 只吃弧度，不吃角度
c, s = np.cos(theta), np.sin(theta)
t = np.array([50.0, 20.0])                # 平移向量 (tx, ty)

T = np.array([[c, -s, t[0]],
              [s,  c, t[1]],
              [0., 0., 1. ]])
print("T =\n", T)

# ---- 2. 变换一个点（先用纸笔手算，再跑代码对答案）----
p = np.array([100.0, 50.0, 1.0])          # 齐次形式 (x, y, 1)
p2 = T @ p                                 # @ 是矩阵乘
print("p =", p[:2], "-> p' =", p2[:2])
# 手算：x' = c*100 - s*50 + 50；y' = s*100 + c*50 + 20

# ---- 3. 图像四角点批量过一遍 ----
corners = np.array([[0, 0, 1],
                    [W, 0, 1],
                    [W, H, 1],
                    [0, H, 1]], dtype=float)   # 每行一个齐次点
moved = (T @ corners.T).T                      # corners.T 每列一个点 → 变换 → 转回
print("角点变换前:\n", corners[:, :2])
print("角点变换后:\n", moved[:, :2])

# ---- 4. 四角点连线可视化 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, pts, title in [(axes[0], corners, 'before'), (axes[1], moved, 'after T')]:
    closed = np.vstack([pts, pts[0]])          # 复制首点到末尾，首尾相连
    ax.plot(closed[:, 0], closed[:, 1], 'o-')
    ax.set_aspect('equal')                     # x/y 等比例，不然正方形变长方形
    ax.invert_yaxis()                          # 图像坐标系 y 轴向下，转回来才像图
    ax.set_title(title)
plt.savefig('ex5_corners.png', dpi=100, bbox_inches='tight')

# ---- 5. 整图变换：cv2.warpAffine（内部就是同一个齐次矩阵）----
M = T[:2, :]                                   # warpAffine 只要 2×3，就是 T 的前两行
warped = cv2.warpAffine(img, M, (W, H))        # 输出尺寸不变，转出去的部分被裁掉

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, im, tt in [(axes[0], img, 'original'),
                   (axes[1], warped, 'warpAffine: rot30 + shift(50,20)')]:
    ax.imshow(im, cmap='gray'); ax.set_title(tt); ax.axis('off')
plt.savefig('ex5_warp.png', dpi=100, bbox_inches='tight')
print("已保存 ex5_corners.png / ex5_warp.png")

# ---- 6. Q2 实验：复合变换的顺序 ----
T_rot = np.array([[c, -s, 0.], [s, c, 0.], [0, 0, 1.]])        # 只旋转
T_tr  = np.array([[1., 0., t[0]], [0., 1., t[1]], [0, 0, 1.]])  # 只平移
q = np.array([100.0, 50.0, 1.0])
print("先转后移:", (T_tr @ T_rot @ q)[:2])     # 从右往左读：先 T_rot 再 T_tr
print("先移后转:", (T_rot @ T_tr @ q)[:2])
print("两次结果相同?", np.allclose(T_tr @ T_rot @ q, T_rot @ T_tr @ q))