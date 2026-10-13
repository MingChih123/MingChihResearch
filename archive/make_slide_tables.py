# make_slide_tables.py
"""
合併 run_clean_eval.py 和 run_image_attack.py 的結果,算出簡報用的表格(不用 GPU,幾秒鐘)。

「攻擊情境」= 攻擊者只擾動仇恨迷因(gt=Yes),正常迷因不動:
  - 被 run_image_attack.py 攻擊的仇恨迷因 → 用攻擊後圖片上的預測
  - 其他樣本(正常迷因、模型本來就答錯的仇恨迷因、縮放就翻掉的 3 筆)→ 用乾淨圖片上的預測
然後在整個 dev 500 筆上算 accuracy / recall / F1,
以及每個防禦相對「攻擊後不防禦」救回幾筆、弄錯幾筆。

用法(CMD):
  python make_slide_tables.py --cleaneval ./output/cleaneval_FB_Qwen2-VL-2B-Instruct_n-1_seed0_noise0.3_ns5_toknospace.json --imgattack ./output/imgattack_FB_dev_vqa_Qwen2-VL-2B-Instruct_n-1_seed0_eps8_a2_s10_noise0.3_ns5_jpeg75.json
"""
import argparse
import json

DEFENSES = [("text", "Text (majority)"), ("pixel", "Pixel"), ("combo", "Combined")]


def metrics(preds, gts):
    tp = sum(p == "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fn = sum(p != "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fp = sum(p == "Yes" and g == "No" for p, g in zip(preds, gts))
    tn = sum(p == "No" and g == "No" for p, g in zip(preds, gts))
    n = len(preds)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return (tp + tn) / n, tp / (tp + fn) if tp + fn else 0.0, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaneval", required=True, help="run_clean_eval.py output on the full dev set")
    parser.add_argument("--imgattack", required=True, help="run_image_attack.py output")
    args = parser.parse_args()

    with open(args.cleaneval, encoding="utf-8") as f:
        clean = json.load(f)["results"]
    with open(args.imgattack, encoding="utf-8") as f:
        attacked = {r["image"]: r for r in json.load(f)["results"]}

    gts = [r["ground_truth"] for r in clean]
    missing = [img for img in attacked if img not in {r["image"] for r in clean}]
    if missing:
        print(f"[WARN] {len(missing)} attacked images not found in cleaneval results")

    # Predictions under the attack scenario
    scen = {"none": []}
    scen.update({k: [] for k, _ in DEFENSES})
    for r in clean:
        a = attacked.get(r["image"])
        if a is not None:
            scen["none"].append(a["attacked_pred"])
            for k, _ in DEFENSES:
                scen[k].append(a["defense_preds"][k])
        else:
            scen["none"].append(r["preds"]["clean_gen"])
            for k, _ in DEFENSES:
                scen[k].append(r["preds"][k])

    print(f"Attacked hateful memes matched: {len(attacked)} | dev size: {len(clean)}\n")
    print("Table 1: overall performance on the full dev set")
    print(f"{'Setting':10s} {'Defense':18s} {'Accuracy':>9s} {'Recall':>8s} {'F1':>6s}")
    acc, rec, f1 = metrics([r["preds"]["clean_gen"] for r in clean], gts)
    print(f"{'Clean':10s} {'none':18s} {acc:9.1%} {rec:8.1%} {f1:6.3f}")
    for key, name in [("none", "none")] + DEFENSES:
        acc, rec, f1 = metrics(scen[key], gts)
        print(f"{'Attacked':10s} {name:18s} {acc:9.1%} {rec:8.1%} {f1:6.3f}")

    print("\nTable 2: vs. attacked + no defense (same 500 memes)")
    print(f"{'Defense':18s} {'Recovered':>10s} {'Harmed':>8s} {'Net':>6s}")
    base = scen["none"]
    for key, name in DEFENSES:
        rec = sum(b != g and d == g for b, d, g in zip(base, scen[key], gts))
        harm = sum(b == g and d != g for b, d, g in zip(base, scen[key], gts))
        print(f"{name:18s} {rec:10d} {harm:8d} {rec - harm:+6d}")


if __name__ == "__main__":
    main()
