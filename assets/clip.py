import os
import cv2
import numpy as np

def crop_and_round_corners(img, pad=5, radius=20):
    """裁掉 pad，并制作圆角，返回 BGRA PNG 图像。"""

    h, w = img.shape[:2]

    # ---- 硬裁四周 pad ----
    img = img[pad:h-pad, pad:w-pad]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # 增加 alpha 通道

    h, w = img.shape[:2]

    # ---- 创建 mask ----
    mask = np.zeros((h, w), dtype=np.uint8)

    # 中间矩形（无圆角部分）
    mask[radius:h-radius, :] = 255
    mask[:, radius:w-radius] = 255

    # 圆角绘制
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (w-radius-1, radius), radius, 255, -1)
    cv2.circle(mask, (radius, h-radius-1), radius, 255, -1)
    cv2.circle(mask, (w-radius-1, h-radius-1), radius, 255, -1)

    # 应用 mask 作为 alpha
    img[:, :, 3] = mask

    return img


def process_folder(input_dir, output_dir, pad=5, radius=20):
    """批处理文件夹中的所有图片。"""

    os.makedirs(output_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg")

    for fname in os.listdir(input_dir):
        if fname.lower().endswith(exts):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname.rsplit(".",1)[0] + ".png")  # 输出统一 PNG

            img = cv2.imread(in_path, cv2.IMREAD_COLOR)
            if img is None:
                print("无法读取:", in_path)
                continue

            processed = crop_and_round_corners(img, pad=pad, radius=radius)
            cv2.imwrite(out_path, processed)

            print("处理完成:", out_path)


# ======================
# 使用示例
# ======================

input_dir = "assets\\nobles"     # 输入文件夹
output_dir = "output"   # 输出文件夹
process_folder(input_dir, output_dir, pad=5, radius=50)
