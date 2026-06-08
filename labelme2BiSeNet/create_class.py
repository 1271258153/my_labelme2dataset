# create_class.py: 根据自己的数据集创建class_name.txt

import json
from pathlib import Path

base = Path("infrared_iamges")
if not base.exists():
    base = Path("infrared_images")
labels = set()
for f in sorted(base.glob("*.json")):
    data = json.load(open(f, encoding="utf-8"))
    for s in data.get("shapes", []):
        labels.add(s["label"])
# 背景放第一行，其余按字母序（与 json_to_dataset.py 一致）
others = sorted(x for x in labels if x != "_background_")
ordered = ["_background_"] + others
out = Path("class_name.txt")
out.write_text("\n".join(ordered) + "\n", encoding="utf-8")
print(f"已写入 {out}，共 {len(ordered)} 个类：")
for i, name in enumerate(ordered):
    print(f"  {i}: {name}")