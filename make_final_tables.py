# make_final_tables.py
"""
把 config_name final 的結果整理成論文用的表格(不用 GPU)。

用法(CMD):
  python make_final_tables.py
  → 讀 output/ 裡檔名結尾是 _final.json 的檔案,寫出 output/_FINAL_TABLES.md

表格:
  A. 乾淨圖:準確率(95% CI)、抓到的仇恨迷因、被誤判的正常迷因
  B. 攻擊下的仍認出率:模型「沒被攻擊時本來就認得出」的仇恨迷因裡,被攻擊後還認得出的比例(95% CI)
     + McNemar 精確檢定(ours vs 其他方法)
  C. 整體準確率(--attack_all:全部圖都加上攻擊)
dual_any(高偵測版)舊版結果檔沒有這個 key 時,用 strip_any OR nocap 重算。
"""
import glob
import json
import math
import os
import sys

OUTPUT_DIR = "./output"
METHODS = [("none", "No defense"), ("jpeg", "JPEG"), ("tq_anyq", "Resize-and-Rephrase (old)"),
           ("f3", "F3"), ("dual", "Dual-Channel (ours)"), ("dual_any", "Dual-Channel-High (ours)")]
ATTACK_NAMES = {
    "caption_persuade": "Persuade (caption)", "caption_persuade_blend": "Blend (caption)",
    "caption_inject": "Inject (caption)", "caption_evade": "Evade (caption, adaptive)",
    "image_text_pad": "Text-in-image", "image_text_persuade": "Text-in-image + Persuade",
    "pgd": "PGD", "pgd_jpeg": "PGD-JPEG", "pgd_persuade": "PGD + Persuade",
    "pgd_eot_max": "PGD-EOT (adaptive)", "saltpepper": "Salt & pepper", "spread": "Spread",
}


def fill_dual_any(p):
    if "dual_any" not in p and "strip_any" in p and "nocap" in p:
        p["dual_any"] = "Yes" if "Yes" in (p["strip_any"], p["nocap"]) else "No"
    return p


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "-"


def ci_cell(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.0f}–{100 * hi:.0f}]"


def dataset_of(d):
    model = d["config"].get("model_name", "qwen2b")
    return d["config"]["dataset_root"] + ("" if model == "qwen2b" else f" ({model})")


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_final.json")))
    if not files:
        print("No *_final.json files found in ./output")
        return
    data = [json.load(open(f, encoding="utf-8")) for f in files]
    lines = ["# Final tables (config_name final)", "",
             "Numbers in [ ] are 95% Wilson confidence intervals. "
             "p = McNemar exact test, Dual-Channel (ours) vs that method (paired on the same memes).", ""]
    for ds in sorted({dataset_of(d) for d in data}):
        lines += [f"## {ds}", ""]
        # A. clean
        for d in [d for d in data if dataset_of(d) == ds and d["config"]["clean_only"]]:
            res = d["results"]
            for r in res:
                fill_dual_any(r["preds"])
            gts = [r["ground_truth"] for r in res]
            n, n_pos, n_neg = len(gts), gts.count("Yes"), gts.count("No")
            lines += [f"### A. Clean memes (no attack): {n} memes ({n_pos} hateful / {n_neg} benign)", "",
                      "| Method | Accuracy | Hateful caught | Benign wrongly flagged | p (acc, vs ours) |",
                      "|---|---|---|---|---|"]
            ours = [r["preds"].get("dual") == r["ground_truth"] for r in res]
            for key, name in METHODS:
                if key not in res[0]["preds"]:
                    continue
                pr = [r["preds"][key] for r in res]
                ok = [p == g for p, g in zip(pr, gts)]
                tp = sum(p == "Yes" and g == "Yes" for p, g in zip(pr, gts))
                fp = sum(p == "Yes" and g == "No" for p, g in zip(pr, gts))
                b = sum(o and not x for o, x in zip(ours, ok))
                c = sum(x and not o for o, x in zip(ours, ok))
                p = "-" if key == "dual" else fmt_p(mcnemar(b, c))
                lines.append(f"| {name} | {ci_cell(sum(ok), n)} | {tp}/{n_pos} ({pct(tp, n_pos)}) | "
                             f"{fp}/{n_neg} ({pct(fp, n_neg)}) | {p} |")
            lines.append("")
        # B. detection rate under attack
        att = [d for d in data if dataset_of(d) == ds and not d["config"]["clean_only"]
               and not d["config"].get("attack_all")]
        if att:
            n_att = len(att[0]["results"])
            lines += [f"### B. Detection rate under attack: {n_att} hateful memes",
                      "",
                      f"{n_att} = hateful memes the model correctly detects before any attack. "
                      "Cell = share still detected after the attack (higher is better).", ""]
            keys = [(k, nm) for k, nm in METHODS if k in att[0]["results"][0]["attacks"][
                next(iter(att[0]["results"][0]["attacks"]))]["preds"] or k == "dual_any"]
            lines += ["| Attack | Attack success | " + " | ".join(nm for _, nm in keys) +
                      " | p vs none | p vs old | p vs F3 |",
                      "|---|---|" + "---|" * len(keys) + "---|---|---|"]
            for d in att:
                for atk in d["results"][0]["attacks"]:
                    rows = [fill_dual_any(r["attacks"][atk]["preds"]) for r in d["results"]]
                    n = len(rows)
                    succ = sum(r["attacks"][atk]["success"] for r in d["results"])
                    cells, yes = [], {}
                    for k, _ in keys:
                        yes[k] = [p.get(k) == "Yes" for p in rows]
                        cells.append(ci_cell(sum(yes[k]), n) if k == "dual" else pct(sum(yes[k]), n))
                    ps = []
                    for other in ("none", "tq_anyq", "f3"):
                        if other in yes:
                            b = sum(o and not x for o, x in zip(yes["dual"], yes[other]))
                            c = sum(x and not o for o, x in zip(yes["dual"], yes[other]))
                            ps.append(f"{fmt_p(mcnemar(b, c))} ({b} vs {c})")
                        else:
                            ps.append("-")
                    lines.append(f"| {ATTACK_NAMES.get(atk, atk)} | {pct(succ, n)} | " + " | ".join(cells)
                                 + " | " + " | ".join(ps) + " |")
            lines.append("")
        # C. attack_all
        for d in [d for d in data if dataset_of(d) == ds and d["config"].get("attack_all")]:
            res = d["results"]
            gts = [r["ground_truth"] for r in res]
            lines += [f"### C. Overall accuracy when every meme is attacked ({len(res)} memes)", "",
                      "| Attack | " + " | ".join(nm for k, nm in METHODS if k != "f3") + " |",
                      "|---|" + "---|" * (len(METHODS) - 1)]
            for atk in res[0]["attacks"]:
                rows = [fill_dual_any(r["attacks"][atk]["preds"]) for r in res]
                cells = []
                for k, _ in METHODS:
                    if k == "f3":
                        continue
                    if k not in rows[0]:
                        cells.append("-")
                        continue
                    cells.append(pct(sum(p[k] == g for p, g in zip(rows, gts)), len(res)))
                lines.append(f"| {ATTACK_NAMES.get(atk, atk)} | " + " | ".join(cells) + " |")
            lines.append("")
    out = os.path.join(OUTPUT_DIR, "_FINAL_TABLES.md")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
