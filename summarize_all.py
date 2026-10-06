# summarize_all.py
"""
把 ./output 裡新實驗的結果整理成 CSV 表格(可以直接用 Excel 開),不用再看終端機。
不用 GPU,幾秒鐘就跑完。舊的 summarize_results.py(舊攻擊實驗用)保持不變。

產生的檔案(都在 ./output/):
  _table_clean.csv      run_clean_eval.py:乾淨圖片上每種防禦的 accuracy / recall / F1 / 救回 / 弄錯
  _table_ablation.csv   ablate_templates.py:三個問法各自單獨、多數決、any-Yes
  _table_imgattack.csv  run_image_attack.py:真實圖片攻擊的成功率與各防禦救回比例
  _table_pipeline.csv   乾淨 → 被攻擊 → 加防禦,在整個 dev 上的 accuracy / recall / F1
                        (需要同一資料集的 cleaneval 全量檔 + imgattack 檔)

用法(CMD):
  python summarize_all.py
"""
import csv
import glob
import json
import os

OUTPUT_DIR = "./output"
PIPELINE_DEFENSES = [("text", "Text (majority)"), ("pixel", "Pixel (majority)"), ("combo", "Combined (majority)")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return "" if x is None else f"{x * 100:.1f}%"


def write_csv(name, rows):
    if not rows:
        print(f"  (no data for {name}, skipped)")
        return
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path} ({len(rows)} rows)")


def metrics(preds, gts):
    tp = sum(p == "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fn = sum(p != "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fp = sum(p == "Yes" and g == "No" for p, g in zip(preds, gts))
    tn = sum(p == "No" and g == "No" for p, g in zip(preds, gts))
    n = len(preds)
    return {"accuracy": (tp + tn) / n, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "correct": tp + tn, "n": n}


def clean_tables(files):
    rows = []
    for path in files:
        d = load(path)
        cfg = d["config"]
        for method, m in d["metrics"].items():
            e = d.get("effects", {}).get(method)
            fixed = harmed = ""
            if e:
                fixed = e["corrected_all"]["count"]
                harmed = e["harmed_all"]["count"]
            rows.append({
                "file": os.path.basename(path), "data": cfg.get("data_file") or "dev",
                "n": m["n"], "noise_std": cfg.get("noise_std"), "method": method,
                "accuracy": pct(m["accuracy"]), "recall": pct(m["recall"]), "f1": round(m["f1"], 3),
                "TP": m["TP"], "FN": m["FN"], "FP": m["FP"], "TN": m["TN"], "unclear": m["unclear"],
                "fixed(wrong->right)": fixed, "harmed(right->wrong)": harmed,
                "net": "" if fixed == "" else fixed - harmed,
            })
    return rows


def ablation_tables(files):
    rows = []
    for path in files:
        d = load(path)
        for rule, m in d["metrics"].items():
            rows.append({
                "file": os.path.basename(path), "rule": rule, "n": m["n"],
                "accuracy": pct(m["accuracy"]), "recall": pct(m["recall"]),
                "precision": pct(m["precision"]), "f1": round(m["f1"], 3),
                "TP": m["TP"], "FN": m["FN"], "FP": m["FP"], "TN": m["TN"],
            })
    return rows


def imgattack_tables(files):
    rows = []
    for path in files:
        d = load(path)
        cfg, s = d["config"], d["summary"]
        n = s.get("n_attacked_stable", s.get("n_clean_correct"))
        row = {
            "file": os.path.basename(path), "eps(/255)": cfg["eps"], "steps": cfg["num_steps"],
            "attacked_memes": n, "attack_success": s["n_attack_success"],
            "attack_success_rate": pct(s["attack_success_rate"]),
            "success_after_jpeg": pct(s["attack_success_after_jpeg_rate"]),
            "mean_psnr": round(s["mean_psnr"], 1) if s.get("mean_psnr") else "",
        }
        for k, v in s["recovery"].items():
            row[f"recovered_{k}"] = pct(v)
        rows.append(row)
    return rows


def pipeline_tables(clean_files, attack_files):
    """Attack scenario on the whole dev set: only attacked hateful memes use attacked-image predictions."""
    rows = []
    full_clean = [p for p in clean_files if load(p)["config"].get("num_samples") == -1
                  and not load(p)["config"].get("data_file")]
    for apath in attack_files:
        a = load(apath)
        acfg = a["config"]
        match = [p for p in full_clean if load(p)["config"]["dataset_root"] == acfg["dataset_root"]]
        if not match:
            print(f"  no full-dev cleaneval file for {os.path.basename(apath)}, pipeline table skipped")
            continue
        cpath = max(match, key=os.path.getmtime)
        clean = load(cpath)["results"]
        attacked = {r["image"]: r for r in a["results"]}
        gts = [r["ground_truth"] for r in clean]

        def add(setting, defense, preds):
            m = metrics(preds, gts)
            rows.append({"attack_file": os.path.basename(apath), "clean_file": os.path.basename(cpath),
                         "setting": setting, "defense": defense,
                         "accuracy": pct(m["accuracy"]), "recall": pct(m["recall"]),
                         "f1": round(m["f1"], 3), "correct": f"{m['correct']}/{m['n']}"})

        add("Clean", "none", [r["preds"]["clean_gen"] for r in clean])
        for key, name in PIPELINE_DEFENSES:
            add("Clean", name, [r["preds"][key] for r in clean])
        add("Attacked", "none",
            [attacked[r["image"]]["attacked_pred"] if r["image"] in attacked else r["preds"]["clean_gen"]
             for r in clean])
        for key, name in PIPELINE_DEFENSES:
            add("Attacked", name,
                [attacked[r["image"]]["defense_preds"][key] if r["image"] in attacked else r["preds"][key]
                 for r in clean])
    return rows


def main():
    clean_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "cleaneval_*.json")))
    ablation_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "ablate_templates_*.json")))
    attack_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "imgattack_*.json")))
    print(f"Found: {len(clean_files)} cleaneval, {len(ablation_files)} ablation, {len(attack_files)} imgattack")

    write_csv("_table_clean.csv", clean_tables(clean_files))
    write_csv("_table_ablation.csv", ablation_tables(ablation_files))
    write_csv("_table_imgattack.csv", imgattack_tables(attack_files))
    pipe = pipeline_tables(clean_files, attack_files)
    write_csv("_table_pipeline.csv", pipe)

    if pipe:
        print("\nClean -> Attacked -> Defended (whole dev set):")
        for r in pipe:
            print(f"  {r['setting']:9s} {r['defense']:20s} acc={r['accuracy']:>6s} "
                  f"recall={r['recall']:>6s} f1={r['f1']:.3f} ({r['correct']})")


if __name__ == "__main__":
    main()
