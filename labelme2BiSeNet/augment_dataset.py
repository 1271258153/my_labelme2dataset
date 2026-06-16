# augment_dataset.py
# 使用ImgAug对数据集进行同步增强（图像和标签一起增强）
# 适用于语义分割任务

import imgaug.augmenters as iaa
import imgaug.augmenters.size as iaa_size
import cv2
import os
import numpy as np
from pathlib import Path
import random

class DatasetAugmentor:
    """数据集增强器，同步处理图像和标签"""
    
    def __init__(self, target_size=(640, 640)):
        """
        初始化增强器
        
        Args:
            target_size: 目标尺寸 (width, height)
        """
        self.target_size = target_size
        
        # 定义几何变换序列（同步应用于图像和标签）
        self.geometric_transform = iaa.Sequential([
            # 随机水平翻转（50%概率）
            iaa.Fliplr(0.5),
            
            # 随机垂直翻转（30%概率）
            iaa.Flipud(0.3),
            
            # 随机旋转（-30到30度）
            iaa.Affine(
                rotate=(-30, 30),
                mode='reflect',  # 使用反射填充
                order=0,  # 最近邻插值（对标签重要）
            ),
            
            # 随机缩放（0.8到1.2倍）
            iaa.Affine(
                scale=(0.8, 1.2),
                mode='reflect',
                order=0,
            ),
            
            # 随机裁剪到目标尺寸
            iaa_size.CenterCropToFixedSize(
                width=self.target_size[0],
                height=self.target_size[1]
            ),
        ])
        
        # 定义色彩变换序列（只应用于图像）
        self.color_transform = iaa.Sequential([
            # 随机亮度调整
            iaa.Multiply((0.7, 1.3)),
            
            # 随机对比度调整
            iaa.contrast.LinearContrast((0.7, 1.3)),
            
            # 随机饱和度调整（通过乘法增强/减弱饱和度）
            iaa.MultiplySaturation((0.7, 1.3)),
            
            # 随机色调调整
            iaa.AddToHue((-20, 20)),
            
            # 随机添加高斯噪声
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        ])
        
        # 定义随机擦除序列（同步应用于图像和标签）
        self.erasing_transform = iaa.Sequential([
            # 随机擦除区域（使用Cutout）
            iaa.Cutout(
                nb_iterations=1,
                position='uniform',
                size=0.2,
                squared=True,
                fill_mode='constant',
                cval=0  # 擦除区域填充0（背景类）
            ),
        ])
    
    def augment_pair(self, image, label, augment_geometry=True, 
                     augment_color=True, augment_erasing=True):
        """
        同步增强图像和标签
        
        Args:
            image: 原始图像 (H, W, 3)
            label: 标签图像 (H, W)
            augment_geometry: 是否应用几何变换
            augment_color: 是否应用色彩变换
            augment_erasing: 是否应用随机擦除
        
        Returns:
            augmented_image, augmented_label
        """
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 确保标签是单通道格式
        if len(label.shape) == 3:
            label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        
        # 应用几何变换（同步变换）
        if augment_geometry:
            # ImgAug需要同时传入图像和标签
            aug_det = self.geometric_transform.to_deterministic()
            image = aug_det.augment_image(image)
            label = aug_det.augment_image(label)
        
        # 应用色彩变换（只变换图像）
        if augment_color:
            image = self.color_transform.augment_image(image)
        
        # 应用随机擦除（同步变换）
        if augment_erasing:
            # 对于随机擦除，我们需要将标签中的擦除区域设为背景类（0）
            aug_det = self.erasing_transform.to_deterministic()
            
            # 先增强图像
            image = aug_det.augment_image(image)
            
            # 对标签应用相同的增强，但确保擦除区域为0
            label_aug = aug_det.augment_image(label)
            # 将标签中的0值保持为0（背景类）
            label = label_aug
        
        # 确保输出尺寸正确
        h, w = self.target_size[1], self.target_size[0]
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        if label.shape[:2] != (h, w):
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 校验标签值范围（必须在 [0, 9] 内）
        label_max = label.max()
        if label_max > 9:
            print(f"  警告: 标签值超出范围 max={label_max}，已裁剪")
            label = np.clip(label, 0, 9)
        
        return image, label
    
    def augment_dataset(self, gt_dir, label_dir, output_gt_dir, output_label_dir, 
                       num_augments=5, seed=42):
        """
        对整个数据集进行增强
        
        Args:
            gt_dir: 原始图像目录
            label_dir: 标签图像目录
            output_gt_dir: 输出原始图像目录
            output_label_dir: 输出标签图像目录
            num_augments: 每张图像生成的增强版本数量
            seed: 随机种子
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建输出目录
        os.makedirs(output_gt_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # 获取所有图像文件
        gt_files = list(Path(gt_dir).glob("*.png"))
        print(f"找到 {len(gt_files)} 张原始图像")
        
        for i, gt_path in enumerate(gt_files):
            # 找到对应的标签文件
            label_path = Path(label_dir) / gt_path.name
            
            if not label_path.exists():
                print(f"警告: 找不到对应的标签文件 {label_path}")
                continue
            
            # 读取图像和标签
            # cv2.imread 默认 BGR，转换为 RGB 供 imgaug 使用
            image = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
            label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            
            if image is None or label is None:
                print(f"警告: 无法读取文件 {gt_path.name}")
                continue
            
            print(f"[{i+1}/{len(gt_files)}] 处理: {gt_path.name}")
            
            # 保存原始图像（如果不存在）
            output_gt_path = Path(output_gt_dir) / gt_path.name
            output_label_path = Path(output_label_dir) / label_path.name
            
            if not output_gt_path.exists():
                cv2.imwrite(str(output_gt_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not output_label_path.exists():
                cv2.imwrite(str(output_label_path), label)
            
            # 生成增强版本
            for j in range(num_augments):
                # 随机选择增强参数
                augment_geometry = random.random() > 0.3  # 70%概率应用几何变换
                augment_color = random.random() > 0.2     # 80%概率应用色彩变换
                augment_erasing = random.random() > 0.7   # 30%概率应用随机擦除
                
                # 应用增强
                aug_image, aug_label = self.augment_pair(
                    image, label, 
                    augment_geometry=augment_geometry,
                    augment_color=augment_color,
                    augment_erasing=augment_erasing
                )
                
                # 生成增强文件名
                base_name = gt_path.stem
                aug_name = f"{base_name}_aug{j+1}.png"
                
                # 保存增强后的图像和标签（图像转回BGR供cv2保存）
                cv2.imwrite(str(Path(output_gt_dir) / aug_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(Path(output_label_dir) / aug_name), aug_label)
            
            print(f"  生成 {num_augments} 个增强版本")
        
        print("数据增强完成！")

def verify_augmentation(gt_dir, label_dir, num_samples=5):
    """验证增强结果的正确性"""
    print("\n验证增强结果...")
    
    gt_files = list(Path(gt_dir).glob("*_aug*.png"))
    if not gt_files:
        print("未找到增强文件")
        return
    
    for i, gt_path in enumerate(gt_files[:num_samples]):
        label_path = Path(label_dir) / gt_path.name
        
        if not label_path.exists():
            print(f"警告: 找不到对应的标签文件 {label_path}")
            continue
        
        # 读取图像和标签
        image = cv2.imread(str(gt_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or label is None:
            print(f"警告: 无法读取文件 {gt_path.name}")
            continue
        
        # 检查尺寸是否一致
        if image.shape[:2] != label.shape[:2]:
            print(f"错误: 尺寸不匹配 {gt_path.name}")
            print(f"  图像尺寸: {image.shape[:2]}")
            print(f"  标签尺寸: {label.shape[:2]}")
            continue
        
        # 检查标签值范围
        unique_labels = np.unique(label)
        valid_labels = set(range(10))  # 0-9共10类
        
        if not all(label in valid_labels for label in unique_labels):
            print(f"错误: 标签值超出范围 {gt_path.name}")
            print(f"  发现的标签值: {unique_labels}")
            continue
        
        print(f"✓ {gt_path.name}: 尺寸 {image.shape[:2]}, 标签值 {unique_labels}")

def main():
    """主函数"""
    # 数据集路径
    base_dir = Path("./dataset")
    gt_dir = base_dir / "gt_png"
    label_dir = base_dir / "label_png"
    
    # 输出目录（expand_dataset 下）
    expand_dir = Path("./expand_dataset")
    output_gt_dir = expand_dir / "gt_png"
    output_label_dir = expand_dir / "label_png"
    
    # 每张图像生成的增强版本数量
    num_augments = 5
    
    print("=" * 60)
    print("ImgAug 数据集增强工具")
    print("=" * 60)
    print(f"原始图像目录: {gt_dir}")
    print(f"原始标签目录: {label_dir}")
    print(f"输出图像目录: {output_gt_dir}")
    print(f"输出标签目录: {output_label_dir}")
    print(f"每张图像生成: {num_augments} 个增强版本")
    print("=" * 60)
    
    # 检查输入目录
    if not gt_dir.exists() or not label_dir.exists():
        print("错误: 输入目录不存在")
        return
    
    # 创建增强器
    augmentor = DatasetAugmentor(target_size=(640, 640))
    
    # 执行增强
    augmentor.augment_dataset(
        gt_dir=str(gt_dir),
        label_dir=str(label_dir),
        output_gt_dir=str(output_gt_dir),
        output_label_dir=str(output_label_dir),
        num_augments=num_augments,
        seed=42
    )
    
    # 验证增强结果
    verify_augmentation(str(output_gt_dir), str(output_label_dir))
    
    print("\n" + "=" * 60)
    print("增强完成！")
    print("增强后的数据集位于:")
    print(f"  原始图像: {output_gt_dir}")
    print(f"  标签图像: {output_label_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()