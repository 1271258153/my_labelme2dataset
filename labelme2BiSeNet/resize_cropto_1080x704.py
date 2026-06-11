# resize_cropto_1080x704.py
# 将 dataset_for_Flir 中混分辨率图片统一为 1080×704（等比缩放 + 中心裁剪）

import cv2
import os

TARGET_W = 1080
TARGET_H = 704

GT_PATH = './dataset_for_Flir/gt_png'
LABEL_PATH = './dataset_for_Flir/label_png'


def resize_crop_to_1080x704(image, is_label=False):
    h, w = image.shape[:2]
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR

    scale = max(TARGET_W / w, TARGET_H / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    x0 = (new_w - TARGET_W) // 2
    y0 = (new_h - TARGET_H) // 2
    return resized[y0:y0 + TARGET_H, x0:x0 + TARGET_W]


def main(path, is_label=False):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                image_name = os.path.join(root, file)
                image = cv2.imread(image_name, -1)
                if image is None:
                    print(f'跳过无法读取: {image_name}')
                    continue
                out = resize_crop_to_1080x704(image, is_label=is_label)
                os.remove(image_name)
                cv2.imwrite(image_name, out)
                print(f'{file}: {image.shape[1]}x{image.shape[0]} -> {TARGET_W}x{TARGET_H}')


if __name__ == '__main__':
    main(GT_PATH, is_label=False)
    main(LABEL_PATH, is_label=True)
