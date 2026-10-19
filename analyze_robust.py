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
主方法(只用部分轉換的輕量版)的顯著性:加 --main_transforms resize50,從存下來的投票重算,不用重跑。
輸出:
  ./output/_analysis_significance_<tag>.csv
  ./output/_analysis_subsets_<tag>.csv   (含「單一轉換 × 單一問法」:同成本比較「最好的一個問法 + 縮放」)
  ./output/_analysis_perq_<tag>.csv      (每個問法各自還說 Yes 的比例、三個問法都答 No 的比例)
  ./output/_analysis_gate_<tag>.csv      (先判斷再防禦:原圖和縮放後答案一致就照原本答案,不一致才用 3 問法 any)
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
         ("tq_majority", "none"), ("tq_majority", "jpeg"),
         # text-side defenses (only when the run included them)
         ("main_safe", "none"), ("main_safe", "tq_anyq"), ("norm", "none"), ("guard", "none"),
         ("main_neg", "none"), ("main_neg", "tq_anyq"), ("neg", "none")]


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


GATE_Q = (0, 1, 2)  # which question is used for the consistency check


def gate_pred(slot, r_idx, k):
    """Selective defense (detect-then-defend), rebuilt from saved votes.
    Ask question k on the raw image and on the resize50 image. If the two answers agree, keep that
    answer (2 queries). If they disagree, the input may be manipulated: ask the other two questions on
    the resized image and flag if any resized answer is Yes (4 queries).
    Returns (prediction, fired, queries)."""
    raw = slot["votes"]["text"][k]
    resized = [slot["votes"]["tq"][str(q)][r_idx] for q in range(3)]
    if raw == resized[k]:
        return raw if raw in ("Yes", "No") else "No", False, 2
    return ("Yes" if "Yes" in resized else "No"), True, 4


def has_gate_votes(slot):
    return "text" in slot["votes"] and "tq" in slot["votes"]


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
    parser.add_argument("--main_transforms", default="",
                        help="also test the lightweight main method, rebuilt from saved votes with only these "
                             "transforms (all 3 questions, any rule), e.g. resize50; added as 'main_anyq'")
    args = parser.parse_args()

    with open(args.attack_json, encoding="utf-8") as f:
        attack = json.load(f)
    with open(args.clean_json, encoding="utf-8") as f:
        clean = json.load(f)
    a_res, c_res = attack["results"], clean["results"]
    t_attack = attack["config"].get("tq_transforms", ",".join(TRANSFORM_NAMES)).split(",")
    t_clean = clean["config"].get("tq_transforms", ",".join(TRANSFORM_NAMES)).split(",")
    t_names = t_attack if t_attack == t_clean else None

    # Rebuild the lightweight main method from saved votes (no rerun needed)
    pairs = list(PAIRS)
    main_t = [t for t in args.main_transforms.split(",") if t]
    if main_t:
        def add_main(records, names, get_slot):
            idx = tuple(names.index(t) for t in main_t)
            for r in records:
                for slot in get_slot(r):
                    slot["preds"]["main_anyq"] = tq_from_votes(slot["votes"]["tq"], idx, (0, 1, 2))[1]
        if all(t in t_attack for t in main_t) and all(t in t_clean for t in main_t):
            add_main(a_res, t_attack, lambda r: r["attacks"].values())
            add_main(c_res, t_clean, lambda r: [r])
            pairs = [("main_anyq", "none"), ("main_anyq", "jpeg"), ("main_anyq", "text_anyyes"),
                     ("main_anyq", "tq_anyq")] + pairs
            print(f"main_anyq = {'+'.join(main_t)} x 3 questions, any rule (rebuilt from saved votes)")
        else:
            print(f"[WARN] --main_transforms {main_t} not found in the saved votes; skipped")
    attacks = list(a_res[0]["attacks"].keys()) if a_res else []

    # Selective defense rebuilt from saved votes (needs raw-image text votes and resize50 tq votes)
    gate_ok = ("resize50" in t_attack and "resize50" in t_clean and a_res and c_res
               and has_gate_votes(c_res[0]) and all(has_gate_votes(a_res[0]["attacks"][a]) for a in attacks))
    gate_rows = []
    if gate_ok:
        ra, rc = t_attack.index("resize50"), t_clean.index("resize50")
        def add_gate(slot, r_idx, k):
            pred, fired, queries = gate_pred(slot, r_idx, k)
            slot["preds"][f"gate_q{k}"] = pred
            slot.setdefault("gate_info", {})[k] = (fired, queries)

        for k in GATE_Q:
            for r in c_res:
                add_gate(r, rc, k)
            for r in a_res:
                for slot in r["attacks"].values():
                    add_gate(slot, ra, k)
        if main_t == ["resize50"]:
            pairs = [(f"gate_q{k}", b) for k in GATE_Q for b in ("none", "main_anyq")] + pairs
        print("gate_q0/1/2 = detect-then-defend (raw vs resize50 consistency on question k), rebuilt from saved votes")
    gts = [r["ground_truth"] for r in c_res]

    # ---------- 1. significance ----------
    sig_rows = []
    print("===== Paired comparisons (McNemar exact) =====")
    for atk in attacks:
        preds = [r["attacks"][atk]["preds"] for r in a_res]
        n = len(preds)
        for a, b in pairs:
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
    for a, b in pairs:
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
    has_votes = (t_names is not None and a_res and "tq" in a_res[0]["attacks"][attacks[0]]["votes"]
                 and "tq" in c_res[0]["votes"])
    sub_rows = []
    if has_votes:
        configs = []
        for k in range(1, len(t_names) + 1):
            for t_idx in itertools.combinations(range(len(t_names)), k):
                configs.append((t_idx, (0, 1, 2)))
        for k in (1, 2):
            for q_idx in itertools.combinations(range(3), k):
                configs.append((tuple(range(len(t_names))), q_idx))
                if len(t_names) > 1:  # one transformation x fewer questions (same-cost comparisons)
                    for t in range(len(t_names)):
                        configs.append(((t,), q_idx))

        for t_idx, q_idx in configs:
            row = {"transforms": "+".join(t_names[t] for t in t_idx),
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
        print("\n(no saved tq votes, or attack/clean runs used different transforms: cost ablation skipped)")

    # ---------- 3. per question: does the attack flip every question? ----------
    perq_rows = []
    print("\n===== Per question under attack: % still Yes (n = attacked memes) =====")
    for atk in attacks:
        slots = [r["attacks"][atk] for r in a_res]
        n = len(slots)
        sources = []
        if "text" in slots[0]["votes"]:
            sources.append(("text (raw image)", [s["votes"]["text"] for s in slots]))
        if "tq" in slots[0]["votes"]:
            def per_q(grid, t_idx):
                return [majority([grid[str(q)][t] for t in t_idx]) for q in range(len(grid))]
            sources.append((f"tq ({'+'.join(t_attack)})",
                            [per_q(s["votes"]["tq"], range(len(t_attack))) for s in slots]))
            if main_t and all(t in t_attack for t in main_t) and main_t != t_attack:
                idx = [t_attack.index(t) for t in main_t]
                sources.append((f"main ({'+'.join(main_t)})", [per_q(s["votes"]["tq"], idx) for s in slots]))
        if "attack_info" in slots[0]:
            # the attacker's own Yes margins (before saving to PNG); margin < 0 means that question says No
            sources.append(("attacker margins", [["Yes" if m >= 0 else "No"
                                                  for m in s["attack_info"]["yes_margin_per_question"]]
                                                 for s in slots]))
        for src, answers in sources:
            row = {"attack": atk, "source": src, "n": n}
            for qi in range(len(answers[0])):
                k = sum(a[qi] == "Yes" for a in answers)
                row[f"q{qi}_still_yes"] = f"{k / n:.1%} ({k}/{n})"
            k_any = sum("Yes" in a for a in answers)
            lo, hi = wilson(n - k_any, n)
            row["any_still_yes"] = f"{k_any / n:.1%} ({k_any}/{n})"
            row["all_questions_no"] = f"{(n - k_any) / n:.1%} ({n - k_any}/{n})"
            row["all_no_95CI"] = f"{lo:.1%}-{hi:.1%}"
            perq_rows.append(row)
            print(f"  {atk:12s} {src:28s} " + " ".join(f"{k}={v}" for k, v in row.items()
                                                       if k.startswith("q") or k == "all_questions_no"))

    # ---------- 4. selective defense: how often the gate fires, cost, and results ----------
    if gate_ok:
        print("\n===== Detect-then-defend (gate) =====")
        n_neg = sum(g == "No" for g in gts)
        n_pos = len(gts) - n_neg
        for k in GATE_Q:
            info = [r["gate_info"][k] for r in c_res]
            p = [r["preds"][f"gate_q{k}"] for r in c_res]
            tp = sum(x == "Yes" and g == "Yes" for x, g in zip(p, gts))
            fp = sum(x == "Yes" and g == "No" for x, g in zip(p, gts))
            row = {"rule": f"gate_q{k}", "clean_acc": f"{sum(x == g for x, g in zip(p, gts)) / len(gts):.1%}",
                   "clean_recall": f"{tp / max(1, n_pos):.1%}",
                   "clean_false_alarm": f"{fp / max(1, n_neg):.1%} ({fp}/{n_neg})",
                   "clean_gate_fired": f"{sum(f for f, _ in info) / len(info):.1%}",
                   "clean_avg_queries": round(sum(q for _, q in info) / len(info), 2)}
            for atk in attacks:
                slots = [r["attacks"][atk] for r in a_res]
                det = sum(sl["preds"][f"gate_q{k}"] == "Yes" for sl in slots)
                fired = sum(sl["gate_info"][k][0] for sl in slots)
                row[f"{atk}_still_detected"] = f"{det / len(slots):.1%} ({det}/{len(slots)})"
                row[f"{atk}_gate_fired"] = f"{fired / len(slots):.1%}"
            gate_rows.append(row)
            print("  " + " | ".join(f"{a}={b}" for a, b in row.items()))
    else:
        print("\n(gate analysis skipped: needs the text defense and resize50 in the tq votes of both files)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_gate_{args.tag}.csv"), gate_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_significance_{args.tag}.csv"), sig_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_subsets_{args.tag}.csv"), sub_rows)
    write_csv(os.path.join(OUTPUT_DIR, f"_analysis_perq_{args.tag}.csv"), perq_rows)


if __name__ == "__main__":
    main()
