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

### 4. 生成类别文件
执行`get_png.py`，会获得`jpg_png`文件，jpg文件存放了原图，png存放了与之对应的24位灰度图（肉眼看都是黑色的，因为类别都是按照像素值划分的，看上去都是黑的其实像素值都是0，1什么的）
```bash
python get_png.py
```

> 执行完成后会在当前目录下生成 `jpg_png` 文件夹，其中包含 `img.jpg`、`img.png` 等文件。

### 5. 生成24位灰度图
`get_png.py`最后生成的是8位灰度图，需要执行`get_dataset.py`
```bash
python get_dataset.py
```

> 执行完之后，得到`dataset`文件，里面存放了成对的训练图片

### 6. 剪裁图片大小至1080×704
分别执行 `resize.py` 和 `crop.py` 文件
```bash
python resize.py
python crop.py
```

- 完成上述步骤之后，执行train_val.py以及train_val_txt.py文件，即以下代码，将数据集划分成训练集和验证集，并且得到txt文件
```bash
python train_val.py
python train_val_txt.py
```
 > 得到我们需要的文件，即 **`dataset`**、**`train.txt`** 以及 **`val.txt`**，至此数据集的准备工作已经全部完成。