from pdf2image import convert_from_path
import cv2
import numpy as np
import os

pdf_path = ".\\origin\\璀璨宝石.pdf"
output_root = ".\\output"

# ---------------------------
# 第一步：PDF → PNG
# ---------------------------
pages = convert_from_path(pdf_path, dpi=300)  # 转图片
os.makedirs(output_root, exist_ok=True)

for idx, page in enumerate(pages):
    page_path = f"{output_root}\\page_{idx}.png"
    page.save(page_path)
    print(f"[OK] 已保存：{page_path}")

# ---------------------------
# 第二步：对每页进行自动网格线裁切
# ---------------------------
for idx, _ in enumerate(pages):

    # 创建当前页文件夹
    page_folder = f"{output_root}\\page_{idx}"
    os.makedirs(page_folder, exist_ok=True)

    img_path = f"{output_root}\\page_{idx}.png"
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 膨胀一下增强细网格线
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=300,
        maxLineGap=20
    )

    if lines is None:
        print(f"[WARN] page_{idx} 未检测到线。")
        continue

    horizontal = []
    vertical = []

    # 分类水平 / 垂直线
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y1 - y2) < 20:   # 水平线
            horizontal.append(y1)
        if abs(x1 - x2) < 20:   # 垂直线
            vertical.append(x1)

    # 去重排序
    horizontal = sorted(list(set(horizontal)))
    vertical = sorted(list(set(vertical)))

    print(f"[page_{idx}] 检测到横线：{horizontal}")
    print(f"[page_{idx}] 检测到竖线：{vertical}")

    # 若线数太少，可能识别不完整
    if len(horizontal) < 2 or len(vertical) < 2:
        print(f"[WARN] page_{idx} 网格线不足，无法裁切")
        continue

    # ---------------------------
    # 裁切网格
    # ---------------------------
    card_id = 0
    for i in range(len(horizontal) - 1):
        for j in range(len(vertical) - 1):
            y1, y2 = horizontal[i], horizontal[i+1]
            x1, x2 = vertical[j], vertical[j+1]

            # 裁切
            crop = img[y1:y2, x1:x2]

            # 保存
            save_path = f"{page_folder}\\card_{card_id}.png"
            cv2.imwrite(save_path, crop)
            print(f"[OK] 保存：{save_path}")

            card_id += 1

print("全部页处理完毕。")
