# get_png.py
# 将jpg图片转换为png图片，并保存到jpg_png/jpg目录下
# 将png图片转换为8位灰度图，并保存到jpg_png/png目录下

import os
from PIL import Image, ImageOps
import numpy as np


def main():
    # 读取原文件夹
    count = os.listdir("infrared_iamges")        # 注意修改为自己的地址！！！
    for i in range(0, len(count)):
        # 如果里的文件以jpg结尾,jpg和JPG兼容
        # 则寻找它对应的png
        if count[i].endswith("jpg") or count[i].endswith("JPG"):
            path = os.path.join("infrared_iamges", count[i]) # 注意修改为自己的地址！！！
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)              # 解决图片旋转问题
            if not os.path.exists('jpg_png/jpg'):
                os.makedirs('jpg_png/jpg')
            img.save(os.path.join("jpg_png/jpg", count[i]))
            # 找到对应的png
            path = "output/" + count[i].split(".")[0] + "_json/label.png"
            img = Image.open(path)
            # 找到全局的类
            class_txt = open("class_name.txt", "r")
            class_name = class_txt.read().splitlines()
            # ["_background_","a","b"]
            # 打开json文件里面存在的类，称其为局部类
            with open("output/" + count[i].split(".")[0] + "_json/label_names.txt", "r") as f:
                names = f.read().splitlines()
                # ["_background_","b"]
                new = Image.new("RGB", [np.shape(img)[1], np.shape(img)[0]])
                # print('new:',new)
                for name in names:
                    index_json = names.index(name)
                    index_all = class_name.index(name)
                    # 将局部类转换成为全局类
                    new = new + np.expand_dims(index_all * (np.array(img) == index_json), -1)
            new = Image.fromarray(np.uint8(new))
            print('new:',new)
            if not os.path.exists('jpg_png/png'):
                os.makedirs('jpg_png/png')
            png_name = os.path.splitext(count[i])[0] + ".png"
            new.save(os.path.join("jpg_png/png", png_name))
            print(np.max(new), np.min(new))

if __name__ == '__main__':
    main()
