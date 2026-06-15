# labelme2dataset

将 Labelme 标注数据转换为 BiSeNet 训练格式。

## 操作步骤

### 1. 准备数据
我的数据集大小：`1280*720`

将数据集中的 `.jpg` 图片和对应的 `.json` 标注文件放入 `labelme2BiSeNet/infrared_iamges` 目录（图片与 JSON 需同名配对）。

### 2. 转换数据集

进入 `labelme2BiSeNet` 目录，按需修改 `json_to_dataset.py` 中的数据集路径，然后执行：

```bash
cd labelme2BiSeNet
python json_to_dataset.py
```

> 执行完成后会在当前目录下生成 `output` 文件夹，其中包含 `img.png`、`label.png` 等文件。

### 3. 生成类别文件

执行 `create_class.py`，根据标注自动生成 `class_name.txt`（含背景类 `_background_`）：

```bash
python create_class.py
```

生成了10类
|_background_|背景（无目标区域）|
|-|-|
|BL_Device|BL设备（基础层设备）|
|CC_Server|CC服务器（内容缓存服务器）|
|DP_Server|DP服务器（数据处理服务器）|
|KDVideo_Device|KD视频设备（监控摄像头）|
|KVM_Switcher|KVM切换器（多台服务器的键鼠显示器切换设备）|
|SP_Cloud|SP云设备（云存储平台）|
|VPN_Gateway|VPN网关|
|WEB_Firewall|Web防火墙|
|YP_Server|YP服务器（业务应用服务器）|


### 4. 生成类别文件
执行`get_png.py`，会获得`jpg_png`文件，jpg文件存放了原图，png存放了与之对应的24位灰度图（肉眼看都是黑色的，因为类别都是按照像素值划分的，看上去都是黑的其实像素值都是0，1什么的）
```bash
python get_png.py
```

> 执行完成后会在当前目录下生成 `jpg_png` 文件夹，其中包含 `img.jpg`、`img.png` 等文件。

### 5. 生成8位灰度图
`get_png.py`最后生成的是24位灰度图，需要执行`get_dataset.py`
```bash
python get_dataset.py
```

> 执行完之后，得到`dataset`文件，里面存放了成对的训练图片
> - `gt_png`：gt_png 保存的是原图，格式从 jpg 变成 png
> - `label_png`：对 png 图做 RGB→Gray 得到的 8bit 单通道灰度图

### 6. 剪裁图片大小至640*640，会直接替换掉dataset下面的图片
```bash
python resize_cropto_640x640.py --input ./dataset/gt_png
python resize_cropto_640x640.py --input ./dataset/label_png --label
```

- 完成上述步骤之后，执行train_val.py以及train_val_txt.py文件，即以下代码，将数据集划分成训练集和验证集，并且得到txt文件
```bash
python train_val.py
python train_val_txt.py
```
 > 得到我们需要的文件，即 **`dataset`**、**`train.txt`** 以及 **`val.txt`**，至此数据集的准备工作已经全部完成。

 ***
**如果要添加的新的数据集，需要修改路径和拓展名**
```bash
cd labelme2BiSeNet
# 建议先清掉旧中间结果，避免和 infrared_iamges 混在一起
# rm -rf output jpg_png dataset class_name.txt
# 1. 改路径后执行
python json_to_dataset.py
python create_class.py      # base 改为 infrad_images_Flir
python get_png.py           # 路径 + .JPG 兼容
python get_dataset.py       # .JPG 兼容

# 第 1–5 步会按原始分辨率生成 dataset/gt_png 和 dataset/label_png
python resize_cropto_640x640.py # 将新增的数据集裁剪到640*640

# 修改 train_val.py 的数据集路径，给新的数据集进行划分
python train_val.py

# 划分完train 和 val 之后，把新的图片全部拷贝到dataset/gt_png(和label_png)/train 和 val 中
cp dataset_for_Flir/gt_png/train/* dataset/gt_png/train/
cp dataset_for_Flir/gt_png/val/* dataset/gt_png/val/
cp dataset_for_Flir/label_png/train/* dataset/label_png/train/
cp dataset_for_Flir/label_png/val/* dataset/label_png/val/

# 最后再执行 train_val_txt.py
python train_val_txt.py
```
***
