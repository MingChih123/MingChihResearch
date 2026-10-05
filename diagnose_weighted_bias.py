# diagnose_weighted_bias.py
"""
P0 診斷:加權投票是否有偏向 Yes 的系統性傾向。

對同一個輸入同時取得:
  1. 生成文字解析的答案(多數決用的方式)
  2. 第一個 token 的 logit 比較答案(加權投票用的方式),分別試三種 token 設定:
       - space   : " Yes" / " No"   (目前 defense.py 用的)
       - nospace : "Yes"  / "No"
       - lse     : 兩種寫法的 logsumexp
然後算:與生成答案的一致率、對標準答案的混淆矩陣 / accuracy / F1、Yes 比例,
以及模型實際生成的第一個 token 是什麼。

只讀取既有模組,不修改任何舊檔案。

用法(CMD):
  python -u diagnose_weighted_bias.py --dataset_root FB --num_samples 100 --seed 0
  python -u diagnose_weighted_bias.py --dataset_root FB --num_samples 100 --seed 0 --attack
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import torch

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, build_inputs, generate_with_pixel_values, parse_answer, get_yes_no_token_ids
from attack import pgd_attack_first_token

OUTPUT_DIR = "./output"
VARIANTS = ["space", "nospace", "lse"]


def load_and_sample(data_path, num_samples, seed):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return shuffled if num_samples == -1 else shuffled[:num_samples]


def single_token_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        print(f"[WARN] '{text}' is tokenized into {len(ids)} tokens: {ids}; using the first one", flush=True)
    return ids[0]


@torch.no_grad()
def first_token_info(model, processor, inputs, pixel_values, tok):
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=inputs.image_grid_thw,
    )
    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    top1_id = int(torch.argmax(logits).item())
    lg = {k: logits[v].item() for k, v in tok.items()}
    pr = {k: probs[v].item() for k, v in tok.items()}

    yes_lse = torch.logsumexp(torch.tensor([lg["yes_sp"], lg["yes_ns"]]), dim=0).item()
    no_lse = torch.logsumexp(torch.tensor([lg["no_sp"], lg["no_ns"]]), dim=0).item()

    gaps = {
        "space": lg["yes_sp"] - lg["no_sp"],
        "nospace": lg["yes_ns"] - lg["no_ns"],
        "lse": yes_lse - no_lse,
    }
    # Strict ">" like defense.py; an exact tie would count as No here
    preds = {k: ("Yes" if g > 0 else "No") for k, g in gaps.items()}

    del outputs
    return {
        "top1_token_id": top1_id,
        "top1_token": processor.tokenizer.decode([top1_id]),
        "logits": lg,
        "probs": pr,
        "gaps_yes_minus_no": gaps,
        "logit_preds": preds,
    }


def binary_metrics(preds, gts):
    """Treat Yes as positive. Unclear predictions count as wrong and are tallied separately."""
    tp = sum(p == "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fn = sum(p != "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fp = sum(p == "Yes" and g == "No" for p, g in zip(preds, gts))
    tn = sum(p == "No" and g == "No" for p, g in zip(preds, gts))
    unclear = sum(p not in ("Yes", "No") for p in preds)
    n = len(preds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "unclear": unclear,
        "accuracy": (tp + tn) / n if n else None,
        "precision": prec, "recall": rec, "f1": f1,
        "yes_ratio": sum(p == "Yes" for p in preds) / n if n else None,
    }


def agreement(a, b):
    pairs = list(zip(a, b))
    agree = sum(x == y for x, y in pairs)
    # Disagreement direction: gen says X, logit says Y
    dirs = Counter(f"gen={x} / logit={y}" for x, y in pairs if x != y)
    return {"agree_rate": agree / len(pairs) if pairs else None, "disagreements": dict(dirs)}


def summarize(records, cond):
    gts = [r["ground_truth"] for r in records]
    gen = [r[cond]["gen_pred"] for r in records]
    out = {"generate": binary_metrics(gen, gts)}
    for v in VARIANTS:
        lp = [r[cond]["logit_preds"][v] for r in records]
        out[f"logit_{v}"] = binary_metrics(lp, gts)
        out[f"logit_{v}"]["vs_generate"] = agreement(gen, lp)
    out["first_token_counts"] = dict(Counter(r[cond]["top1_token"] for r in records).most_common(10))
    return out


def print_summary(title, s):
    print(f"\n===== {title} =====", flush=True)
    print(f"{'method':16s} {'acc':>6s} {'f1':>6s} {'yes%':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s} {'unc':>4s} {'agree_w_gen':>12s}")
    for k in ["generate"] + [f"logit_{v}" for v in VARIANTS]:
        m = s[k]
        ag = m.get("vs_generate", {}).get("agree_rate")
        ag_s = f"{ag:.2%}" if ag is not None else "-"
        print(f"{k:16s} {m['accuracy']:6.2%} {m['f1']:6.3f} {m['yes_ratio']:6.2%} "
              f"{m['TP']:4d} {m['FN']:4d} {m['FP']:4d} {m['TN']:4d} {m['unclear']:4d} {ag_s:>12s}")
    for v in VARIANTS:
        d = s[f"logit_{v}"]["vs_generate"]["disagreements"]
        if d:
            print(f"  disagreements ({v}): {d}")
    print(f"  first generated token (top-1) counts: {s['first_token_counts']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="FB or HarMeme")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=100, help="-1 = all (mixed Yes/No)")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--attack", action="store_true",
                        help="also run PGD (same settings as run_experiment.py) on every sample and re-measure")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--num_steps", type=int, default=3)
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    args.model_name = resolve_model_name(args.model_name)

    subset = load_and_sample(f"{dataset_root}/{data_file}", args.num_samples, args.seed)
    gt_counts = Counter(d["answer"] for d in subset)
    print(f"Dataset: {args.dataset_root} | {len(subset)} samples | gt counts: {dict(gt_counts)}", flush=True)

    model, processor = load_model(args.model_name, args.min_pixels, args.max_pixels)
    tk = processor.tokenizer
    tok = {
        "yes_sp": single_token_id(tk, " Yes"),
        "no_sp": single_token_id(tk, " No"),
        "yes_ns": single_token_id(tk, "Yes"),
        "no_ns": single_token_id(tk, "No"),
    }
    print(f"Token ids: {tok}", flush=True)
    # Same ids the attack and weighted vote use
    atk_yes, atk_no = get_yes_no_token_ids(processor)

    records = []
    t0 = time.time()
    for i, item in enumerate(subset):
        image_path = f"{dataset_root}/{item['image']}"
        inputs = build_inputs(processor, image_path, item["question"], model.device)

        conds = {"clean": inputs.pixel_values}
        if args.attack:
            # PGD pushes toward No for every sample (same objective as run_experiment.py)
            conds["attacked"] = pgd_attack_first_token(
                model, inputs, atk_yes, atk_no, args.epsilon, args.alpha, args.num_steps
            )

        rec = {"image": item["image"], "ground_truth": item["answer"]}
        for cond, pv in conds.items():
            gen_text = generate_with_pixel_values(model, processor, inputs, pv, args.max_new_tokens)
            info = first_token_info(model, processor, inputs, pv, tok)
            info["gen_text"] = gen_text
            info["gen_pred"] = parse_answer(gen_text)
            rec[cond] = info
        records.append(rec)

        c = rec["clean"]
        line = (f"  [{i+1}/{len(subset)}] gt={item['answer']:3s} gen={c['gen_pred']:7s} "
                f"sp={c['logit_preds']['space']:3s} ns={c['logit_preds']['nospace']:3s} "
                f"lse={c['logit_preds']['lse']:3s} top1={c['top1_token']!r}")
        if args.attack:
            a = rec["attacked"]
            line += f" | ATK gen={a['gen_pred']:7s} sp={a['logit_preds']['space']:3s} ns={a['logit_preds']['nospace']:3s}"
        print(line, flush=True)

        del inputs, conds
        torch.cuda.empty_cache()

    summary = {"clean": summarize(records, "clean")}
    print_summary("Clean", summary["clean"])
    if args.attack:
        summary["attacked"] = summarize(records, "attacked")
        print_summary("Attacked (PGD toward No)", summary["attacked"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_tag = args.model_name.split("/")[-1]
    out_path = os.path.join(
        OUTPUT_DIR,
        f"diagnose_{args.dataset_root}_{model_tag}_n{args.num_samples}_seed{args.seed}"
        + ("_attack" if args.attack else "") + ".json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "token_ids": tok, "summary": summary,
                   "total_time_sec": time.time() - t0, "results": records},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
