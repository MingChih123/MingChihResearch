# make_extra_vqa.py
"""
把新加的三個資料集轉成和 FB dev_vqa.json 一樣的格式(image / question / answer),
每個資料集寫一個 test_vqa.json。只用官方的 test 切分(我們不訓練、不調參,test 只拿來報結果)。

  MAMI     (厭女迷因,SemEval-2022 Task 5)  ./dataset/MAMI/test.tsv + Images/test_images/
           label 1 = 厭女 → Yes
  PrideMM  (LGBTQ+ 相關的仇恨迷因)         ./dataset/PrideMM/PrideMM.csv(split == test)+ Images/
           hate 1 = 仇恨 → Yes
  HarmP    (政治類有害迷因,MOMENTA)        ./dataset/HarmP/Harm-P/data/datasets/memes/defaults/annotations/test.jsonl
           "not harmful" → No,其餘(somewhat / very harmful)→ Yes(和 HarMeme 同一套規則)

問題句子用 defense.py 的 EXTRA_TEMPLATES[資料集][0](q0),迷因文字裡的雙引號換成單引號
(不然 extract_caption 會把 caption 截斷)。

用法(CMD,在專案根目錄):
  python make_extra_vqa.py                 三個都轉
  python make_extra_vqa.py --only MAMI     只轉一個(MAMI / PrideMM / HarmP)
"""
import argparse
import csv
import json
import os
from collections import Counter

from config import DATASET_ROOT_MAP
from defense import EXTRA_TEMPLATES


def clean_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    if text.upper() in ("NA", "NAN", "NONE"):
        return ""
    return text.replace('"', "'")


def read_table(path):
    """CSV or TSV with a header row (delimiter detected from the header line)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        head = f.readline()
        f.seek(0)
        delim = "\t" if head.count("\t") > head.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


def pick(row, *names):
    """First matching column, case-insensitive (releases differ, e.g. 'text' vs 'Text Transcription')."""
    low = {k.strip().lower(): v for k, v in row.items() if k}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} in columns {list(row)}")


def yes_no(value):
    v = str(value).strip()
    if v in ("1", "1.0"):
        return "Yes"
    if v in ("0", "0.0"):
        return "No"
    return None  # missing / unexpected label: skipped


def convert_mami(root):
    items = []
    for r in read_table(os.path.join(root, "test.tsv")):
        ans = yes_no(pick(r, "label", "misogynous"))
        if ans is None:
            continue
        items.append(("Images/test_images/" + pick(r, "file_name", "file name").strip(),
                      clean_text(pick(r, "text", "Text Transcription")), ans))
    return items


def convert_pridemm(root):
    items = []
    for r in read_table(os.path.join(root, "PrideMM.csv")):
        if pick(r, "split").strip().lower() != "test":
            continue
        ans = yes_no(pick(r, "hate"))
        if ans is None:
            continue
        items.append(("Images/" + pick(r, "name").strip(), clean_text(pick(r, "text")), ans))
    return items


def convert_harmp(root):
    items = []
    with open(os.path.join(root, "annotations", "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lab = r.get("labels", r.get("label"))
            if isinstance(lab, list):
                lab = lab[0] if lab else None
            if lab is None:
                continue
            ans = "No" if str(lab).strip().lower() in ("not harmful", "not-harmful", "0") else "Yes"
            items.append(("images/" + os.path.basename(r.get("image", r.get("img", ""))),
                          clean_text(r.get("text", r.get("caption"))), ans))
    return items


CONVERTERS = {"MAMI": convert_mami, "PrideMM": convert_pridemm, "HarmP": convert_harmp}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(CONVERTERS), default=None)
    args = parser.parse_args()
    for name, fn in CONVERTERS.items():
        if args.only and name != args.only:
            continue
        root = DATASET_ROOT_MAP[name]
        print(f"===== {name} ({root}) =====")
        try:
            items = fn(root)
        except FileNotFoundError as e:
            print(f"  [SKIP] file not found: {e.filename}\n")
            continue
        q0 = EXTRA_TEMPLATES[name][0]
        data = [{"image": img, "question": q0.format(caption=cap), "answer": ans} for img, cap, ans in items]
        missing = [d["image"] for d in data if not os.path.exists(os.path.join(root, d["image"]))]
        out = os.path.join(root, "test_vqa.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        empty = sum(cap == "" for _, cap, _ in items)
        print(f"  wrote {out}: {len(data)} memes, answers {dict(Counter(d['answer'] for d in data))}")
        print(f"  empty captions: {empty}; images missing on disk: {len(missing)}"
              + (f" (e.g. {missing[:3]})" if missing else ""))
        if data:
            print(f"  example: {data[0]['image']} -> {data[0]['answer']}\n    {data[0]['question'][:160]}")
        print()


if __name__ == "__main__":
    main()
