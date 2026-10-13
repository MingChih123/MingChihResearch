# analyze_robust.py
"""
用 run_robust_suite.py 已經存好的結果做進一步分析(不用 GPU,幾秒鐘)。

1. 顯著性:主要防禦兩兩比較的 McNemar 精確檢定(同一批圖片的成對比較),
   以及每個數字的 95% 信賴區間(Wilson)。
2. 成本消融:tq 每張圖要問 15 次(5 種轉換 × 3 個問法)。這裡直接用存下來的投票,
   算「只用部分轉換 / 部分問法」時的效果,不用重跑模型。
   注意:要挑「用哪幾個轉換」請看 train 的結果,dev 只用來報最後結果。

用法(CMD):
  python analyze_robust.py --attack_json ./output/robust_attack_....json --clean_json ./output/robust_clean_....json --tag dev
輸出:
  ./output/_analysis_significance_<tag>.csv
  ./output/_analysis_subsets_<tag>.csv
"""
import argparse
import csv
import itertools
import json
import math
import os

OUTPUT_DIR = "./output"
TRANSFORM_NAMES = ["jpeg50", "jpeg75", "resize50", "blur1", "median3"]  # order used in run_robust_suite.py
QUESTION_NAMES = ["q0", "q1", "q2"]
PAIRS = [("tq_anyq", "none"), ("tq_anyq", "jpeg"), ("tq_anyq", "text_anyyes"),
         ("tq_majority", "none"), ("tq_majority", "jpeg")]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def majority(votes):
    valid = [v for v in votes if v in ("Yes", "No")]
    if not valid or valid.count("Yes") == valid.count("No"):
        return "Unclear"
    return "Yes" if valid.count("Yes") > valid.count("No") else "No"


def tq_from_votes(grid, t_idx, q_idx):
    """Recompute tq_majority / tq_anyq from saved votes using a subset of transforms and questions."""
    rows = [[grid[str(q)][t] for t in t_idx] for q in q_idx]
    flat = [v for r in rows for v in r]
    per_q = [majority(r) for r in rows]
    return majority(flat), ("Yes" if "Yes" in per_q else "No")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_json", required=True)
    parser.add_argument("--clean_json", required=True)
    parser.add_argument("--tag", default="dev", help="name used in the output files, e.g. dev / train")
    args = parser.parse_args()

    with open(args.attack_json, encoding="utf-8") as f:
        attack = json.load(f)
    with open(args.clean_json, encoding="utf-8") as f:
        clean = json.load(f)
    a_res, c_res = attack["results"], clean["results"]
    attacks = list(a_res[0]["attacks"].keys()) if a_res else []
    gts = [r["ground_truth"] for r in c_res]

    # ---------- 1. significance ----------
    sig_rows = []
    print("===== Paired comparisons (McNemar exact) =====")
    for atk in attacks:
        preds = [r["attacks"][atk]["preds"] for r in a_res]
        n = len(preds)
        for a, b in PAIRS:
            if a not in preds[0] or b not in preds[0]:
                continue
            ya = [p[a] == "Yes" for p in preds]
            yb = [p[b] == "Yes" for p in preds]
            only_a = sum(x and not y for x, y in zip(ya, yb))
            only_b = sum(y and not x for x, y in zip(ya, yb))
            lo, hi = wilson(sum(ya), n)
            pval = mcnemar_exact(only_a, only_b)
            sig_rows.append({"setting": f"attack:{atk}", "metric": "still detected (gt=Yes)", "A": a, "B": b,
                             "A_rate": f"{sum(ya) / n:.1%}", "A_95CI": f"{lo:.1%}-{hi:.1%}",
                             "B_rate": f"{sum(yb) / n:.1%}", "only_A_right": only_a, "only_B_right": only_b,
                             "p_value": round(pval, 4)})
            print(f"  {atk:10s} {a:12s} {sum(ya) / n:6.1%} vs {b:12s} {sum(yb) / n:6.1%} "
                  f"(only A {only_a}, only B {only_b}) p={pval:.4f}")
    cp = [r["preds"] for r in c_res]
    for a, b in PAIRS:
        if a not in cp[0] or b not in cp[0]:
            continue
        ra = [p[a] == g for p, g in zip(cp, gts)]
        rb = [p[b] == g for p, g in zip(cp, gts)]
        only_a = sum(x and not y for x, y in zip(ra, rb))
        only_b = sum(y and not x for x, y in zip(ra, rb))
        lo, hi = wilson(sum(ra), len(ra))
        pval = mcnemar_exact(only_a, only_b)
        sig_rows.append({"setting": "clean", "metric": "accuracy (mixed)", "A": a, "B": b,
                         "A_rate": f"{sum(ra) / len(ra):.1%}", "A_95CI": f"{lo:.1%}-{hi:.1%}",
                         "B_rate": f"{sum(rb) / len(rb):.1%}", "only_A_right": only_a, "only_B_right": only_b,
                         "p_value": round(pval, 4)})
        print(f"  {'clean':10s} {a:12s} {sum(ra) / len(ra):6.1%} vs {b:12s} {sum(rb) / len(rb):6.1%} "
              f"(only A {only_a}, only B {only_b}) p={pval:.4f}")

    # ---------- 2. cost ablation from saved votes ----------
    has_votes = a_res and "tq" in a_res[0]["attacks"][attacks[0]]["votes"] and "tq" in c_res[0]["votes"]
    sub_rows = []
    if has_votes:
        configs = []
        for k in range(1, len(TRANSFORM_NAMES) + 1):
            for t_idx in itertools.combinations(range(len(TRANSFORM_NAMES)), k):
                configs.append((t_idx, (0, 1, 2)))
        for k in (1, 2):
            for q_idx in itertools.combinations(range(3), k):
                configs.append((tuple(range(len(TRANSFORM_NAMES))), q_idx))

        for t_idx, q_idx in configs:
            row = {"transforms": "+".join(TRANSFORM_NAMES[t] for t in t_idx),
                   "questions": "+".join(QUESTION_NAMES[q] for q in q_idx),
                   "queries_per_image": len(t_idx) * len(q_idx)}
            for rule_i, rule in enumerate(("majority", "anyq")):
                cpred = [tq_from_votes(r["votes"]["tq"], t_idx, q_idx)[rule_i] for r in c_res]
                tp = sum(p == "Yes" and g == "Yes" for p, g in zip(cpred, gts))
                fp = sum(p == "Yes" and g == "No" for p, g in zip(cpred, gts))
                n_neg = sum(g == "No" for g in gts)
                row[f"{rule}_clean_acc"] = f"{sum(p == g for p, g in zip(cpred, gts)) / len(gts):.1%}"
                row[f"{rule}_clean_recall"] = f"{tp / max(1, sum(g == 'Yes' for g in gts)):.1%}"
                row[f"{rule}_clean_false_alarm"] = f"{fp / max(1, n_neg):.1%}"
                for atk in attacks:
                    ap = [tq_from_votes(r["attacks"][atk]["votes"]["tq"], t_idx, q_idx)[rule_i] for r in a_res]
                    row[f"{rule}_{atk}"] = f"{sum(p == 'Yes' for p in ap) / len(ap):.1%}"
            sub_rows.append(row)

        print("\n===== Cost ablation (tq_anyq), selected rows =====")
        show = sorted((r for r in sub_rows if r["queries_per_image"] in (15, 10, 9, 5)),
                      key=lambda r: -r["queries_per_image"])
        for r in show[:20]:
            atk_cells = " ".join(f"{atk}={r[f'anyq_{atk}']}" for atk in attacks)
            print(f"  {r['queries_per_image']:2d} q | {r['transforms']:38s} {r['questions']:9s} | "
                  f"clean acc={r['anyq_clean_acc']} false_alarm={r['anyq_clean_false_alarm']} | {atk_cells}")
    else:
        print("\n(no saved tq votes found, cost ablation skipped)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_significance_{args.tag}.csv"), sig_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_subsets_{args.tag}.csv"), sub_rows)


if __name__ == "__main__":
    main()
