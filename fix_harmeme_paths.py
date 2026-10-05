# fix_harmeme_paths.py
"""
一次性修復:把 HarMeme 的 train_vqa.json / val_vqa.json 裡的 image 欄位
加上 "images/" 前綴,並存成新檔案(不覆蓋原檔,保留備份)。
"""
import json
import os

ROOT = "./dataset/HarMeme/annotations"
FILES = ["train_vqa.json", "val_vqa.json"]

for fname in FILES:
    path = os.path.join(ROOT, fname)
    if not os.path.exists(path):
        print(f"skip (not found): {path}")
        continue

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    fixed = 0
    for item in data:
        if not item["image"].startswith("images/"):
            item["image"] = f"images/{item['image']}"
            fixed += 1

    out_path = os.path.join(ROOT, fname)  # 直接覆寫,因為前綴補一次後之後重跑不會再加第二次
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{fname}: fixed {fixed} entries (total {len(data)})")