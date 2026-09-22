# day02/ex4_fileio.py
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 运行产物统一落到 ex4_output/（该目录已 gitignore，跑一遍脚本就会重新生成）
os.makedirs('ex4_output', exist_ok=True)
os.chdir('ex4_output')

IMG = '/home/jiaxin/datasets/objectDetectionDatasets/demo/voc/JPEGImages/train_29635.jpg'
LBL = '/home/jiaxin/datasets/objectDetectionDatasets/demo/yolov3/custom/labels/train_29635.txt'

img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)

# ---- 1. 三种存法：JPEG(有损) / PNG(无损) / npy(数组原样) ----
cv2.imwrite('out_q10.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 10])  # 低质量，故意压出 artifacts
cv2.imwrite('out_q95.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 高质量
cv2.imwrite('out.png', img)                                       # 无损

hmap = img.astype(np.float32) * 0.05   # 伪装成真实高度图：0~12.75mm 的毫米值
np.save('hmap.npy', hmap)              # npy 一步存，dtype/shape/数值全保真

for f in ['out_q10.jpg', 'out_q95.jpg', 'out.png', 'hmap.npy']:
    print(f, os.path.getsize(f), 'bytes')

# ---- 2. 读回来，量化验尸 ----
back_jpg = cv2.imread('out_q95.jpg', cv2.IMREAD_GRAYSCALE)
back_png = cv2.imread('out.png', cv2.IMREAD_GRAYSCALE)
back_npy = np.load('hmap.npy')

print("PNG 与原图逐位相同:", np.array_equal(back_png, img))
print("JPG q95 与原图最大误差:", np.abs(back_jpg.astype(int) - img.astype(int)).max())
print("npy 读回:", back_npy.dtype, back_npy.shape, "数据相同:", np.array_equal(back_npy, hmap))

# ---- 3. 错误示范：高度图硬存 PNG（看数据怎么死的）----
cv2.imwrite('hmap_wrong.png', hmap)     # float 被静默截断成 uint8，不报错
back_wrong = cv2.imread('hmap_wrong.png', cv2.IMREAD_GRAYSCALE)
print("高度图硬存 PNG 读回范围:", back_wrong.min(), "-", back_wrong.max(), "（mm 值已不可恢复）")

# ---- 4. 读 YOLO 标注，把框画回图上 ----
H, W = img.shape
vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    # 画彩框需要三通道；cv2 用 BGR 顺序
with open(LBL) as f:
    for line in f:
        cls, xc, yc, w, h = line.split()       # 字符串拆开，5 个一组
        cls, xc, yc, w, h = int(cls), float(xc), float(yc), float(w), float(h)
        x1 = int((xc - w / 2) * W); y1 = int((yc - h / 2) * H)   # 归一化 → 像素
        x2 = int((xc + w / 2) * W); y2 = int((yc + h / 2) * H)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)   # 绿框，线宽 2
        cv2.putText(vis, f'class {cls}', (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))  # cv2 是 BGR，matplotlib 要 RGB——经典坑！
plt.title('YOLO boxes'); plt.axis('off')
plt.savefig('ex4_boxes.png', dpi=100, bbox_inches='tight')
print("已保存 ex4_boxes.png")