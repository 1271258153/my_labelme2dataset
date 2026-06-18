# augment_dataset.py
# 使用 Albumentations 对训练集进行数据增强
# 只增强 dataset/gt_png/train 和 dataset/label_png/train
# 每张图生成 10 个增强版本

import albumentations as A
import cv2
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ========== 配置 ==========
GT_DIR = 'dataset/gt_png/train'
LABEL_DIR = 'dataset/label_png/train'
OUTPUT_GT_DIR = 'expand_dataset/gt_png'
OUTPUT_LABEL_DIR = 'expand_dataset/label_png'
NUM_AUGMENTS = 10
NUM_WORKERS = min(cpu_count(), 8)
TARGET_SIZE = (640, 640)

# ========== 增强 Pipeline（严格按要求配置，禁止 Hue/Saturation） ==========
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),
    A.GaussNoise(
        var_limit=(10.0, 50.0),
        p=0.3,
    ),
    A.MotionBlur(
        blur_limit=5,
        p=0.2,
    ),
])


def verify_filenames(gt_dir, label_dir):
    """验证原图和标签文件名是否完全一致"""
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

    gt_set = set(gt_files)
    label_set = set(label_files)

    if gt_set != label_set:
        only_in_gt = gt_set - label_set
        only_in_label = label_set - gt_set
        if only_in_gt:
            print(f'[错误] 以下文件只在 gt_png 中存在: {only_in_gt}')
        if only_in_label:
            print(f'[错误] 以下文件只在 label_png 中存在: {only_in_label}')
        raise ValueError('原图和标签文件名不一致，请检查数据集！')

    print(f'✓ 文件名验证通过，共 {len(gt_files)} 对文件')
    return gt_files


def augment_single(args):
    """增强单张图像（用于多进程）"""
    fname, gt_dir, label_dir, output_gt_dir, output_label_dir, aug_idx, seed = args

    np.random.seed(seed)

    gt_path = os.path.join(gt_dir, fname)
    label_path = os.path.join(label_dir, fname)

    image = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if image is None or label is None:
        return f'[警告] 无法读取: {fname}'

    # 确保尺寸一致
    if image.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        image = cv2.resize(image, TARGET_SIZE)
    if label.shape[:2] != (TARGET_SIZE[1], TARGET_SIZE[0]):
        label = cv2.resize(label, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # 应用增强
    augmented = transform(image=image, mask=label)
    aug_image = augmented['image']
    aug_label = augmented['mask']

    # 确保标签值不越界
    aug_label = np.clip(aug_label, 0, 255)

    # 生成输出文件名
    stem = Path(fname).stem
    ext = Path(fname).suffix
    aug_name = f'{stem}_aug{aug_idx}{ext}'

    cv2.imwrite(os.path.join(output_gt_dir, aug_name), aug_image)
    cv2.imwrite(os.path.join(output_label_dir, aug_name), aug_label)

    return None


def copy_originals(gt_dir, label_dir, output_gt_dir, output_label_dir, file_list):
    """将原始文件复制到输出目录"""
    for fname in file_list:
        gt_src = os.path.join(gt_dir, fname)
        gt_dst = os.path.join(output_gt_dir, fname)
        label_src = os.path.join(label_dir, fname)
        label_dst = os.path.join(output_label_dir, fname)

        if not os.path.exists(gt_dst):
            img = cv2.imread(gt_src, cv2.IMREAD_COLOR)
            cv2.imwrite(gt_dst, img)
        if not os.path.exists(label_dst):
            lbl = cv2.imread(label_src, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(label_dst, lbl)


def main():
    print('=' * 60)
    print('Albumentations 训练集数据增强')
    print('=' * 60)
    print(f'原图目录:   {GT_DIR}')
    print(f'标签目录:   {LABEL_DIR}')
    print(f'输出原图:   {OUTPUT_GT_DIR}')
    print(f'输出标签:   {OUTPUT_LABEL_DIR}')
    print(f'增强倍数:   {NUM_AUGMENTS}')
    print(f'并行进程:   {NUM_WORKERS}')
    print('=' * 60)

    # 1. 验证文件名
    file_list = verify_filenames(GT_DIR, LABEL_DIR)
    total = len(file_list)

    # 2. 创建输出目录
    os.makedirs(OUTPUT_GT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # 3. 复制原始图像
    print(f'\n复制 {total} 张原始图像...')
    copy_originals(GT_DIR, LABEL_DIR, OUTPUT_GT_DIR, OUTPUT_LABEL_DIR, file_list)

    # 4. 构建增强任务列表
    tasks = []
    for fname in file_list:
        for aug_idx in range(1, NUM_AUGMENTS + 1):
            seed = hash(fname) % (2**31) + aug_idx
            tasks.append((
                fname, GT_DIR, LABEL_DIR,
                OUTPUT_GT_DIR, OUTPUT_LABEL_DIR,
                aug_idx, seed
            ))

    print(f'\n开始增强，共 {len(tasks)} 个任务...')

    # 5. 多进程执行增强
    warnings = []
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(augment_single, tasks),
            total=len(tasks),
            desc='增强进度',
            ncols=80,
        ):
            if result is not None:
                warnings.append(result)

    # 6. 输出结果
    print(f'\n{"=" * 60}')
    print(f'增强完成！')
    print(f'  原始图像: {total} 张')
    print(f'  增强图像: {len(tasks)} 张')
    print(f'  总计:     {total + len(tasks)} 张')
    if warnings:
        print(f'\n警告 ({len(warnings)}):')
        for w in warnings:
            print(f'  {w}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()