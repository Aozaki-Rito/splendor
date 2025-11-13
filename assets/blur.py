import cv2

def blur_overlay(input_path, output_path, blur_strength=25, alpha=0.4):
    """
    用模糊图作为蒙版叠加在原图上:
    blur_strength = 模糊核大小
    alpha = 模糊层占比（0~1）
    """

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(input_path)

    # 模糊图层
    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)

    # alpha 混合
    out = cv2.addWeighted(blurred, alpha, img, 1 - alpha, 0)

    cv2.imwrite(output_path, out)
    print("Saved:", output_path)


# 使用示例
blur_overlay("assets\\splendor.jpeg", "assets\\card_blur_overlay.png",
             blur_strength=391, alpha=0.5)
