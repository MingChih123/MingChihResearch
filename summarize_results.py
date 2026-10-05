# summarize_results.py
"""
掃描 ./output/ 資料夾裡所有實驗 json,彙整成一張表格,
輸出成 CSV,方便直接貼進論文/簡報,也方便你自己核對數字來源。
"""
import json
import os
import csv

OUTPUT_DIR = "./output"
SUMMARY_CSV = "./output/_summary.csv"


def main():
    rows = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue

        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Only attack experiments (run_experiment.py) have these keys
        if "n_attacked" not in data:
            continue

        cfg = data.get("config", {})
        row = {
            "filename": fname,
            "dataset": data.get("dataset_tag", cfg.get("dataset_root", "")),
            "model": cfg.get("model_name", ""),
            "num_samples": cfg.get("num_samples", ""),
            "seed": cfg.get("seed", ""),
            "epsilon": cfg.get("epsilon", ""),
            "alpha": cfg.get("alpha", ""),
            "num_steps": cfg.get("num_steps", ""),
            "noise_std": cfg.get("noise_std", ""),
            "config_name": cfg.get("config_name", ""),
            # Files from before the token fix have no token_mode -> they used " Yes"/" No"
            "token_mode": cfg.get("token_mode", "space(old)"),
            "n_evaluated": data.get("n_evaluated", ""),
            "n_attacked": data.get("n_attacked", ""),
            "attack_success_rate": data.get("attack_success_rate", ""),
            "text_defense_recovery_rate": data.get("text_defense_recovery_rate", ""),
            "text_weighted_defense_recovery_rate": data.get("text_weighted_defense_recovery_rate", ""),
            "pixel_defense_recovery_rate": data.get("pixel_defense_recovery_rate", ""),
            "pixel_weighted_defense_recovery_rate": data.get("pixel_weighted_defense_recovery_rate", ""),
            "combo_defense_recovery_rate": data.get("combo_defense_recovery_rate", ""),
            "combo_weighted_defense_recovery_rate": data.get("combo_weighted_defense_recovery_rate", ""),
            "total_time_sec": data.get("total_time_sec", ""),
        }
        rows.append(row)

    if not rows:
        print("No result files found in ./output/")
        return

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summarized {len(rows)} experiment files.")
    print(f"Saved to: {SUMMARY_CSV}")
    print()
    print("Preview:")
    for r in rows:
        tag = r['config_name'] or 'default'
        print(f"  [{tag:12s}] tok={r['token_mode']:10s} eps={r['epsilon']:5} noise={r['noise_std']:5} "
              f"n_eval={r['n_evaluated']:3} n_atk={r['n_attacked']:3} "
              f"| text={r['text_defense_recovery_rate']} "
              f"text_w={r['text_weighted_defense_recovery_rate']} "
              f"pixel={r['pixel_defense_recovery_rate']} "
              f"pixel_w={r['pixel_weighted_defense_recovery_rate']} "
              f"combo={r['combo_defense_recovery_rate']} "
              f"combo_w={r['combo_weighted_defense_recovery_rate']}")


if __name__ == "__main__":
    main()