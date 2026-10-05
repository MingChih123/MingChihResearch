# run_clean_eval.py
"""
乾淨圖片(沒有攻擊)、Yes/No 混合樣本上的完整評估。
取代 run_error_correction_experiment.py + clean_control_check.py 的分析方式:
每一筆樣本都跑全部防禦(不只挑答錯的),然後報:

  1. 每種方法的整體 accuracy / precision / recall / F1 / 混淆矩陣 / Yes 比例
  2. 修正率,拆成兩個方向:
       - gt=Yes 但模型答 No(漏抓仇恨迷因)
       - gt=No  但模型答 Yes(誤判正常迷因)
  3. 傷害率:原本答對的樣本,套防禦後變錯的比例(也拆 gt=Yes / gt=No)

另外存下每筆的 logit 差距(Yes - No)與加權投票的總分,留給之後做 train-free 校正分析。

用法(CMD):
  python -u run_clean_eval.py --dataset_root FB --num_samples 200 --seed 0
"""
import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, get_yes_no_token_ids, build_inputs, generate_with_pixel_values, parse_answer
from defense import (
    get_answer_with_confidence, vote_score,
    paraphrase_defense, paraphrase_defense_weighted,
    randomized_smoothing_defense, randomized_smoothing_defense_weighted,
    combined_defense, combined_defense_weighted,
)

OUTPUT_DIR = "./output"
METHODS = ["clean_gen", "clean_logit", "text", "text_w", "pixel", "pixel_w", "combo", "combo_w"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_sample(data_path, num_samples, seed):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return shuffled if num_samples == -1 else shuffled[:num_samples]


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def binary_metrics(preds, gts):
    """Yes = positive class. Unclear counts as wrong."""
    tp = sum(p == "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fn = sum(p != "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fp = sum(p == "Yes" and g == "No" for p, g in zip(preds, gts))
    tn = sum(p == "No" and g == "No" for p, g in zip(preds, gts))
    n = len(preds)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "unclear": sum(p not in ("Yes", "No") for p in preds),
        "accuracy": (tp + tn) / n if n else None,
        "precision": prec, "recall": rec, "f1": f1,
        "yes_ratio": sum(p == "Yes" for p in preds) / n if n else None,
    }


def frac(num, den):
    return {"count": num, "total": den, "rate": num / den if den else None}


def correction_and_harm(records, method):
    """Compare a defense against the clean generated answer (the undefended system)."""
    out = {}
    for gt in ("Yes", "No"):
        wrong = [r for r in records if r["ground_truth"] == gt and r["preds"]["clean_gen"] != gt]
        right = [r for r in records if r["ground_truth"] == gt and r["preds"]["clean_gen"] == gt]
        out[f"corrected_gt{gt}"] = frac(sum(r["preds"][method] == gt for r in wrong), len(wrong))
        out[f"harmed_gt{gt}"] = frac(sum(r["preds"][method] != gt for r in right), len(right))
    all_wrong = [r for r in records if r["preds"]["clean_gen"] != r["ground_truth"]]
    all_right = [r for r in records if r["preds"]["clean_gen"] == r["ground_truth"]]
    out["corrected_all"] = frac(sum(r["preds"][method] == r["ground_truth"] for r in all_wrong), len(all_wrong))
    out["harmed_all"] = frac(sum(r["preds"][method] != r["ground_truth"] for r in all_right), len(all_right))
    return out


def fmt_rate(d):
    return f"{d['count']}/{d['total']}" + (f" ({d['rate']:.0%})" if d["rate"] is not None else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=200, help="-1 = all (mixed Yes/No)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--token_mode", type=str, default="nospace",
                        help="nospace (correct) / lse / space (old, buggy)")
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"cleaneval_{args.dataset_root}_"
        # Tag non-default splits (e.g. train_vqa) so they never overwrite dev results
        f"{os.path.splitext(os.path.basename(args.data_file))[0] + '_' if args.data_file else ''}"
        f"{model_name.split('/')[-1]}_n{args.num_samples}_"
        f"seed{args.seed}_noise{args.noise_std}_ns{args.num_noise_samples}_tok{args.token_mode}"
        f"{'_' + args.config_name if args.config_name else ''}.json"
    )

    subset = load_and_sample(f"{dataset_root}/{data_file}", args.num_samples, args.seed)
    n_yes = sum(d["answer"] == "Yes" for d in subset)
    print(f"Loaded {len(subset)} samples (Yes={n_yes}, No={len(subset) - n_yes})", flush=True)
    print(f"Output will be saved to: {output_path}", flush=True)

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)
    yes_id, no_id = get_yes_no_token_ids(processor, args.token_mode)
    print(f"Token mode: {args.token_mode} | yes_ids={yes_id} no_ids={no_id}", flush=True)

    records = []
    t0 = time.time()
    for i, item in enumerate(subset):
        s0 = time.time()
        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]
        caption = extract_caption(question)
        gt = item["answer"]
        seed_offset = args.seed * 1000 + i
        n_combo_noise = max(1, args.num_noise_samples // 2)

        inputs = build_inputs(processor, image_path, question, model.device)
        pv = inputs.pixel_values

        clean_gen = parse_answer(generate_with_pixel_values(model, processor, inputs, pv, args.max_new_tokens))
        clean_logit, clean_conf = get_answer_with_confidence(model, processor, inputs, pv, yes_id, no_id)

        text_pred, _ = paraphrase_defense(model, processor, image_path, caption, pv, args.max_new_tokens)
        text_w_pred, text_w_votes = paraphrase_defense_weighted(
            model, processor, image_path, caption, pv, yes_id, no_id, args.max_new_tokens)
        pixel_pred, _ = randomized_smoothing_defense(
            model, processor, inputs, pv, num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)
        pixel_w_pred, pixel_w_votes = randomized_smoothing_defense_weighted(
            model, processor, inputs, pv, yes_id, no_id, num_samples=args.num_noise_samples,
            noise_std=args.noise_std, max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)
        combo_pred, _ = combined_defense(
            model, processor, image_path, caption, inputs, pv, num_noise_samples=n_combo_noise,
            noise_std=args.noise_std, max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)
        combo_w_pred, combo_w_votes = combined_defense_weighted(
            model, processor, image_path, caption, inputs, pv, yes_id, no_id,
            num_noise_samples=n_combo_noise, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)

        preds = {
            "clean_gen": clean_gen, "clean_logit": clean_logit,
            "text": text_pred, "text_w": text_w_pred,
            "pixel": pixel_pred, "pixel_w": pixel_w_pred,
            "combo": combo_pred, "combo_w": combo_w_pred,
        }
        records.append({
            "image": item["image"],
            "ground_truth": gt,
            "preds": preds,
            # Signed scores (>0 leans Yes) for later calibration analysis
            "scores": {
                "clean_logit": clean_conf if clean_logit == "Yes" else -clean_conf,
                "text_w": vote_score(text_w_votes),
                "pixel_w": vote_score(pixel_w_votes),
                "combo_w": vote_score(combo_w_votes),
            },
            "time_sec": round(time.time() - s0, 2),
        })

        print(f"  [{i+1}/{len(subset)}] gt={gt:3s} gen={clean_gen:7s} logit={clean_logit:3s} | "
              f"text={text_pred:7s} text_w={text_w_pred:7s} pixel={pixel_pred:7s} pixel_w={pixel_w_pred:7s} "
              f"combo={combo_pred:7s} combo_w={combo_w_pred:7s} ({time.time() - s0:.1f}s)", flush=True)

        del inputs, pv
        gc.collect()
        torch.cuda.empty_cache()

    gts = [r["ground_truth"] for r in records]
    metrics = {m: binary_metrics([r["preds"][m] for r in records], gts) for m in METHODS}
    effects = {m: correction_and_harm(records, m) for m in METHODS if m != "clean_gen"}

    print("\n===== Overall (clean, mixed Yes/No) =====", flush=True)
    print(f"{'method':12s} {'acc':>7s} {'f1':>6s} {'recall':>7s} {'yes%':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s} {'unc':>4s}")
    for m in METHODS:
        x = metrics[m]
        print(f"{m:12s} {x['accuracy']:7.2%} {x['f1']:6.3f} {x['recall']:7.2%} {x['yes_ratio']:6.2%} "
              f"{x['TP']:4d} {x['FN']:4d} {x['FP']:4d} {x['TN']:4d} {x['unclear']:4d}")

    print("\n===== Correction / harm vs. undefended generate answer =====", flush=True)
    print(f"{'method':12s} {'fix gt=Yes':>14s} {'fix gt=No':>14s} {'harm gt=Yes':>14s} {'harm gt=No':>14s}")
    for m, e in effects.items():
        print(f"{m:12s} {fmt_rate(e['corrected_gtYes']):>14s} {fmt_rate(e['corrected_gtNo']):>14s} "
              f"{fmt_rate(e['harmed_gtYes']):>14s} {fmt_rate(e['harmed_gtNo']):>14s}")

    total = time.time() - t0
    print(f"\nTotal time: {total:.1f}s ({total / max(1, len(records)):.1f}s/sample)", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": metrics, "effects": effects,
                   "total_time_sec": total, "results": records}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
