# split_dataset.py

'''
对 get_dataset.py 生成的 dataset/gt_png 和 dataset/label_png 进行划分
按 train:test:val = 6:2:2 移动到对应子目录下

划分后的目录结构:
  dataset/
    gt_png/train/   gt_png/test/   gt_png/val/
    label_png/train/ label_png/test/ label_png/val/
'''

import os
import random
import shutil

# ========== 配置 ==========
dataset_dir = 'dataset'
src_image_dir = os.path.join(dataset_dir, 'gt_png')
src_label_dir = os.path.join(dataset_dir, 'label_png')

train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

random_seed = 42

# ========== 创建目标目录 ==========
splits = ['train', 'test', 'val']
dst_dirs = {}
for split in splits:
    dst_dirs[split] = {
        'image': os.path.join(src_image_dir, split),
        'label': os.path.join(src_label_dir, split),
    }
    for d in dst_dirs[split].values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

# ========== 收集文件列表（仅取 gt_png 根目录下的 png 文件） ==========
all_files = sorted([
    f for f in os.listdir(src_image_dir)
    if f.endswith('.png') and os.path.isfile(os.path.join(src_image_dir, f))
])

total_size = len(all_files)
print(f'总样本数: {total_size}')

# ========== 随机打乱并划分 ==========
random.seed(random_seed)
random.shuffle(all_files)

train_size = int(total_size * train_ratio)
test_size = int(total_size * test_ratio)

train_files = all_files[:train_size]
test_files = all_files[train_size:train_size + test_size]
val_files = all_files[train_size + test_size:]

print(f'train: {len(train_files)}, test: {len(test_files)}, val: {len(val_files)}')

# ========== 移动文件到对应目录 ==========
split_map = {
    'train': train_files,
    'test': test_files,
    'val': val_files,
}

for split_name, file_list in split_map.items():
    for fname in file_list:
        # 移动 gt_png
        src_img = os.path.join(src_image_dir, fname)
        dst_img = os.path.join(dst_dirs[split_name]['image'], fname)
        shutil.move(src_img, dst_img)

        # 移动 label_png
        src_lbl = os.path.join(src_label_dir, fname)
        dst_lbl = os.path.join(dst_dirs[split_name]['label'], fname)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f'[警告] label 文件缺失: {src_lbl}')

# ========== 生成 txt 文件 ==========
def write_txt(split_name, txt_path):
    gt_dir = dst_dirs[split_name]['image']
    label_dir = dst_dirs[split_name]['label']
    gt_files = sorted(os.listdir(gt_dir))
    with open(txt_path, 'w') as f:
        for g in gt_files:
            label_file = g
            if os.path.exists(os.path.join(label_dir, label_file)):
                f.write(f'gt_png/{split_name}/{g},label_png/{split_name}/{label_file}\n')

write_txt('train', 'train.txt')
write_txt('test', 'test.txt')
write_txt('val', 'val.txt')

print('\n划分完成！')
print(f'目录结构: dataset/gt_png/train, dataset/gt_png/test, dataset/gt_png/val')
print(f'生成文件: train.txt ({len(train_files)}条), test.txt ({len(test_files)}条), val.txt ({len(val_files)}条)')
