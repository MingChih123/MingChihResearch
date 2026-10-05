# check_harmeme.py
import json
import os
from collections import Counter

root = "./dataset/HarMeme"
data_file = "annotations/val_vqa.json"

with open(os.path.join(root, data_file), encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
print("Label distribution:", Counter(d["answer"] for d in data))
print()
print("First 3 entries, checking if image file actually exists:")
for d in data[:3]:
    resolved_path = os.path.join(root, d["image"])
    exists = os.path.exists(resolved_path)
    print(f"  image field: {d['image']}")
    print(f"  resolved path: {resolved_path}")
    print(f"  file exists: {exists}")
    print(f"  question: {d['question'][:80]}...")
    print(f"  answer: {d['answer']}")
    print()