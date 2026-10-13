# make_harmeme_test_vqa.py
"""
把 HarMeme 的 annotations/test.jsonl 轉成和 val_vqa.json 一樣格式的 annotations/test_vqa.json,
另外輸出 annotations/valtest_vqa.json(val + test 合併)。我們在 HarMeme 上不調任何參數,
所以 val 和 test 都可以當評估資料,合併後樣本比較多。

安全檢查:先用同樣的規則轉換 val.jsonl,和現有的 val_vqa.json 逐筆比對(答案、caption),
一致率要接近 100% 才代表規則和當初的 trans.py 相同。不一致時會印出例子,請把結果貼給 Claude。

標籤規則:"not harmful" → No;其餘("somewhat harmful"、"very harmful" 等)→ Yes。
問題句子:直接沿用 val_vqa.json 第一筆的問題模板,只換 caption。

用法(CMD,在專案根目錄):
  python make_harmeme_test_vqa.py
"""
import json
import os
from collections import Counter

ROOT = "./dataset/HarMeme/annotations"
CAPTION_MARK = 'caption "'


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_of(rec):
    """Return 'Yes' / 'No' from the various label formats HarMeme releases use."""
    lab = rec.get("labels", rec.get("label"))
    if isinstance(lab, list):
        lab = lab[0] if lab else None
    if isinstance(lab, (int, float)):
        return "Yes" if int(lab) == 1 else "No"
    if isinstance(lab, str):
        return "No" if lab.strip().lower() in ("not harmful", "not-harmful", "0") else "Yes"
    raise ValueError(f"cannot read label from record keys {list(rec)}")


def caption_of(rec):
    text = rec.get("text", rec.get("caption"))
    if text is None:
        raise ValueError(f"cannot find text field in record keys {list(rec)}")
    # A double quote inside the caption would cut extract_caption() short
    return text.replace('"', "'")


def image_of(rec):
    img = rec.get("image", rec.get("img"))
    if img is None:
        raise ValueError(f"cannot find image field in record keys {list(rec)}")
    img = os.path.basename(img)
    return f"images/{img}"


def extract_caption(question):
    return question.split(CAPTION_MARK)[1].split('"')[0]


def main():
    with open(os.path.join(ROOT, "val_vqa.json"), encoding="utf-8") as f:
        val_vqa = json.load(f)
    q0 = val_vqa[0]["question"]
    cap0 = extract_caption(q0)
    head, tail = q0.split(CAPTION_MARK + cap0, 1)
    template = head + CAPTION_MARK + "{caption}" + tail
    print(f"Question template (from val_vqa.json):\n  {template}\n")

    def convert(records):
        return [{"image": image_of(r), "question": template.format(caption=caption_of(r)), "answer": label_of(r)}
                for r in records]

    # ---- check against the existing val_vqa.json ----
    val_conv = convert(read_jsonl(os.path.join(ROOT, "val.jsonl")))
    by_img = {os.path.basename(d["image"]): d for d in val_vqa}
    matched = [(c, by_img.get(os.path.basename(c["image"]))) for c in val_conv]
    found = [(c, o) for c, o in matched if o is not None]
    same_ans = sum(c["answer"] == o["answer"] for c, o in found)
    same_cap = sum(extract_caption(c["question"]).strip() == extract_caption(o["question"]).strip() for c, o in found)
    print(f"val check: {len(found)}/{len(val_conv)} images found in val_vqa.json")
    print(f"  same answer : {same_ans}/{len(found)}")
    print(f"  same caption: {same_cap}/{len(found)}")
    bad = [(c, o) for c, o in found if c["answer"] != o["answer"]][:3]
    for c, o in bad:
        print(f"  answer differs: {c['image']} converted={c['answer']} existing={o['answer']}")
    bad_cap = [(c, o) for c, o in found
               if extract_caption(c["question"]).strip() != extract_caption(o["question"]).strip()][:2]
    for c, o in bad_cap:
        print(f"  caption differs: {c['image']}\n    converted: {extract_caption(c['question'])[:80]!r}"
              f"\n    existing : {extract_caption(o['question'])[:80]!r}")
    if len(found) == 0 or same_ans < len(found):
        print("\n[STOP] Labels do not match the existing val_vqa.json; test_vqa.json was NOT written. "
              "Please send this output (and trans.py) to Claude.")
        return

    # ---- convert test ----
    test_conv = convert(read_jsonl(os.path.join(ROOT, "test.jsonl")))
    out = os.path.join(ROOT, "test_vqa.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(test_conv, f, ensure_ascii=False, indent=2)
    missing = sum(not os.path.exists(os.path.join("./dataset/HarMeme", d["image"])) for d in test_conv)
    print(f"\nWrote {out}: {len(test_conv)} items, answers {dict(Counter(d['answer'] for d in test_conv))}")
    print(f"Images missing on disk: {missing}")

    # Nothing is tuned on HarMeme, so val + test together can serve as one larger evaluation set
    val_imgs = {os.path.basename(d["image"]) for d in val_vqa}
    overlap = sum(os.path.basename(d["image"]) in val_imgs for d in test_conv)
    combined = val_vqa + [d for d in test_conv if os.path.basename(d["image"]) not in val_imgs]
    out2 = os.path.join(ROOT, "valtest_vqa.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out2}: {len(combined)} items (val + test, {overlap} overlapping images skipped), "
          f"answers {dict(Counter(d['answer'] for d in combined))}")
    if same_cap < len(found):
        print("Note: some captions differ slightly from val_vqa.json (e.g. whitespace); answers all match, "
              "so the file is usable, but mention this to Claude.")


if __name__ == "__main__":
    main()
