# resize_cropto_640x640.py
# 将数据集图片统一裁剪为 640×640 分辨率（等比缩放 + 中心裁剪）
# 适用于 PIDNet 训练和 RK3588 部署
# 默认处理 ./dataset_for_Flir 目录下的图像

import cv2
import os
import argparse
import sys
from pathlib import Path

# 默认目标分辨率
DEFAULT_TARGET_W = 640
DEFAULT_TARGET_H = 640


def resize_crop_to_target(image, target_w, target_h, is_label=False):
    """
    将图像等比缩放并中心裁剪到目标分辨率
    
    Args:
        image: 输入图像
        target_w: 目标宽度
        target_h: 目标高度
        is_label: 是否为标签图像（使用最近邻插值）
    
    Returns:
        裁剪后的图像
    """
    h, w = image.shape[:2]
    
    # 根据图像类型选择插值方式
    # 标签图像使用最近邻插值保持类别值准确
    # 普通图像使用双线性插值保持视觉质量
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    
    # 计算缩放比例，确保目标分辨率完全被覆盖
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    
    # 执行缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    
    # 计算中心裁剪的起始位置
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    
    # 返回裁剪后的图像
    return resized[y0:y0 + target_h, x0:x0 + target_w]


def process_directory(input_path, output_path, target_w, target_h, is_label=False, dry_run=False):
    """
    处理目录中的所有PNG图像
    
    Args:
        input_path: 输入目录路径
        output_path: 输出目录路径
        target_w: 目标宽度
        target_h: 目标高度
        is_label: 是否为标签图像
        dry_run: 是否只显示预览不实际处理
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在 - {input_path}")
        return
    
    # 创建输出目录（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_files = 0
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    # 获取所有PNG文件
    png_files = list(input_path.glob("**/*.png"))
    total_files = len(png_files)
    
    if total_files == 0:
        print(f"警告: 在 {input_path} 中未找到PNG文件")
        return
    
    print(f"找到 {total_files} 个PNG文件")
    print(f"目标分辨率: {target_w}×{target_h}")
    print(f"处理模式: {'标签图像' if is_label else '普通图像'}")
    print("-" * 50)
    
    for i, image_path in enumerate(png_files, 1):
        # 计算相对路径，保持目录结构
        relative_path = image_path.relative_to(input_path)
        output_file_path = output_path / relative_path
        
        # 确保输出目录存在
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 读取图像
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"[{i}/{total_files}] 跳过无法读取: {relative_path}")
                skipped_files += 1
                continue
            
            # 获取原始尺寸
            orig_h, orig_w = image.shape[:2]
            
            if dry_run:
                # 干跑模式：只显示预览
                print(f"[{i}/{total_files}] 预览: {relative_path}")
                print(f"  原始尺寸: {orig_w}×{orig_h}")
                print(f"  目标尺寸: {target_w}×{target_h}")
                processed_files += 1
                continue
            
            # 执行裁剪
            processed_image = resize_crop_to_target(image, target_w, target_h, is_label)
            
            # 保存处理后的图像
            cv2.imwrite(str(output_file_path), processed_image)
            
            print(f"[{i}/{total_files}] 已处理: {relative_path}")
            print(f"  {orig_w}×{orig_h} -> {target_w}×{target_h}")
            
            processed_files += 1
            
        except Exception as e:
            print(f"[{i}/{total_files}] 处理错误: {relative_path}")
            print(f"  错误信息: {e}")
            error_files += 1
    
    # 打印统计信息
    print("-" * 50)
    print("处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"跳过文件: {skipped_files}")
    print(f"错误文件: {error_files}")


def main():
    """
    主函数：解析命令行参数并执行处理
    """
    parser = argparse.ArgumentParser(
        description="将数据集图片裁剪到指定分辨率（默认640×640）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（默认就地修改，与旧脚本行为一致）
  python resize_cropto_640x640.py
  
  # 处理标签图像（默认就地修改）
  python resize_cropto_640x640.py --label
  
  # 不就地修改，创建新目录
  python resize_cropto_640x640.py --no-inplace
  
  # 指定输入输出目录
  python resize_cropto_640x640.py --input ./my_dataset/gt_png --output ./resized/gt_png
  
  # 自定义分辨率
  python resize_cropto_640x640.py --width 512 --height 512
  
  # 预览模式（不实际处理）
  python resize_cropto_640x640.py --dry-run
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=None,
        help="输入目录路径（默认: ./dataset_for_Flir/gt_png 或 ./dataset_for_Flir/label_png）"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="输出目录路径（默认: 与输入路径相同，就地修改）"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        default=DEFAULT_TARGET_W,
        help=f"目标宽度（默认: {DEFAULT_TARGET_W}）"
    )
    
    parser.add_argument(
        "--height", 
        type=int, 
        default=DEFAULT_TARGET_H,
        help=f"目标高度（默认: {DEFAULT_TARGET_H}）"
    )
    
    parser.add_argument(
        "--label", 
        action="store_true",
        help="处理标签图像（使用最近邻插值）"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="预览模式，只显示将处理的文件，不实际修改"
    )
    
    parser.add_argument(
        "--inplace", 
        action="store_true",
        default=True,  # 默认就地修改，与旧脚本行为一致
        help="就地修改原文件（覆盖原文件）（默认行为）"
    )
    
    parser.add_argument(
        "--no-inplace", 
        action="store_true",
        help="不就地修改，创建新目录"
    )
    
    args = parser.parse_args()
    
    # 验证目标分辨率
    if args.width <= 0 or args.height <= 0:
        print("错误: 目标分辨率必须大于0")
        sys.exit(1)
    
    # 设置默认路径（与旧脚本 resize_cropto_1080x704.py 保持一致）
    if args.input is None:
        if args.label:
            input_path = "./dataset_for_Flir/label_png"
        else:
            input_path = "./dataset_for_Flir/gt_png"
    else:
        input_path = args.input
    
    # 处理 --no-inplace 参数
    if args.no_inplace:
        args.inplace = False
    
    # 设置输出路径
    if args.output is None:
        if args.inplace:
            # 就地修改：输出路径与输入路径相同（默认行为，与旧脚本一致）
            output_path = input_path
        else:
            # 非就地修改：在输入路径同级创建新目录
            input_parent = Path(input_path).parent
            input_name = Path(input_path).name
            output_path = str(input_parent / f"{input_name}_resized_{args.width}x{args.height}")
    else:
        output_path = args.output
    
    # 显示处理信息
    print("=" * 50)
    print("数据集分辨率裁剪工具（默认行为与旧脚本一致）")
    print("=" * 50)
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"目标分辨率: {args.width}×{args.height}")
    print(f"图像类型: {'标签图像' if args.label else '普通图像'}")
    print(f"处理模式: {'就地修改（覆盖原文件）' if args.inplace else '创建新目录'}")
    print(f"运行模式: {'预览' if args.dry_run else '实际处理'}")
    print("=" * 50)
    
    # 确认操作
    if not args.dry_run and not args.inplace:
        response = input("是否继续处理？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            print("操作已取消")
            return
    
    # 就地修改时的警告
    if not args.dry_run and args.inplace:
        print(f"警告：将就地修改原文件，覆盖 {input_path} 中的图像")
        response = input("是否继续？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            print("操作已取消")
            return
    
    # 执行处理
    process_directory(
        input_path=input_path,
        output_path=output_path,
        target_w=args.width,
        target_h=args.height,
        is_label=args.label,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()