# aggregate_seeds.py
"""
把同一個實驗、不同 seed 的 run_robust_suite.py 結果合併,算「平均 ± 標準差」(不用 GPU,幾秒鐘)。

用法(CMD):檔名用 * 代表 seed 那一段,程式會自動找出所有符合的檔案
  python aggregate_seeds.py --pattern "./output/robust_attack_FB_dev_vqa_*_seed*_eps8_s10_atk-pgd-pgd_jpeg-pgd_eot_tq-resize50_eotdef_rs.json" --tag fb_dev
  python aggregate_seeds.py --pattern "./output/robust_clean_FB_dev_vqa_*_seed*_tq-resize50.json" --tag fb_dev_clean

輸出 ./output/_seeds_<tag>.csv,每格是 mean ± std(n = seed 數)。
提醒:seed 數太少(例如 3 個)時標準差不太準,論文裡請同時報 analyze_robust.py 的 95% 信賴區間。
"""
import argparse
import csv
import glob
import json
import os
import statistics

OUTPUT_DIR = "./output"


def fmt(values):
    if not values:
        return ""
    m = statistics.mean(values) * 100
    s = statistics.stdev(values) * 100 if len(values) > 1 else 0.0
    return f"{m:.1f} ± {s:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, help="glob pattern matching the per-seed result files")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files match {args.pattern}")
        return
    runs = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        runs.append((p, d))
        print(f"  seed {d['config'].get('seed')}: {os.path.basename(p)}")
    seeds = [d["config"].get("seed") for _, d in runs]
    if len(set(seeds)) != len(seeds):
        print("[WARN] some files share the same seed; check the pattern")

    rows = []
    clean_mode = runs[0][1]["config"].get("clean_only", False)
    if not clean_mode:
        attacks = list(runs[0][1]["summary"]["table"].keys())
        for atk in attacks:
            keys = [k for k in runs[0][1]["summary"]["table"][atk] if k != "tq_flag_for_review_rate"]
            row = {"attack": atk, "n_seeds": len(runs)}
            for k in keys:
                vals = [d["summary"]["table"][atk][k] for _, d in runs if atk in d["summary"]["table"]
                        and d["summary"]["table"][atk].get(k) is not None]
                row["attack_success" if k == "attack_success_rate" else f"still_detected_{k}"] = fmt(vals)
            rows.append(row)
    else:
        defenses = list(runs[0][1]["summary"]["metrics"].keys())
        for k in defenses:
            ms = [d["summary"]["metrics"][k] for _, d in runs if k in d["summary"]["metrics"]]
            n_neg = [m["FP"] + m["TN"] for m in ms]
            rows.append({"defense": k, "n_seeds": len(ms),
                         "accuracy": fmt([m["accuracy"] for m in ms]),
                         "recall": fmt([m["recall"] for m in ms]),
                         "f1(x100)": fmt([m["f1"] for m in ms]),
                         "false_alarm": fmt([m["FP"] / n for m, n in zip(ms, n_neg) if n])})

    print(f"\n===== mean ± std over {len(runs)} seeds (percent) =====")
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"_seeds_{args.tag}.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
