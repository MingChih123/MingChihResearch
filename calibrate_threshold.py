# calibrate_threshold.py
"""
Train-free 門檻校正(不動模型權重)。

問題:模型偏向答 No(recall 低),因為「Yes 分數 > 0 才判 Yes」這個門檻對它來說太嚴。
做法:
  1. 在 train split(和 dev 不重疊)抽一些樣本,只做前向計算,算出分數:
       - clean_logit: 單次詢問的 logit(Yes) - logit(No)
       - text_w     : 3 種問法的 logit 差距加總(= 加權投票的總分)
  2. 在 train 上選門檻 tau,最大化 balanced accuracy
     (用 balanced accuracy 是因為 train 的 Yes 比例和 dev 不同,選門檻時不能依賴類別比例)。
  3. 把 tau 套到 run_clean_eval.py 已經存好的 dev 分數上,和原本的結果比較。
     dev 完全沒有參與選門檻,不會作弊。

用法(CMD):
  python -u calibrate_threshold.py --dataset_root FB --num_calib 300 --seed 0 --eval_json ./output/cleaneval_FB_Qwen2-VL-2B-Instruct_n200_seed0_noise0.3_ns5_toknospace.json
"""
import argparse
import gc
import json
import os
import random
import time

import torch

from config import resolve_dataset_root, resolve_model_name
from common import load_model, get_yes_no_token_ids, build_inputs
from defense import get_answer_with_confidence, paraphrase_defense_weighted, vote_score
from run_clean_eval import binary_metrics, correction_and_harm, fmt_rate, extract_caption

OUTPUT_DIR = "./output"
CALIB_FILE_MAP = {
    "FB": "train_vqa.json",
    "HarMeme": "annotations/train_vqa.json",
}
SCORE_METHODS = ["clean_logit", "text_w"]


def balanced_accuracy(scores, gts, tau):
    pos = [s for s, g in zip(scores, gts) if g == "Yes"]
    neg = [s for s, g in zip(scores, gts) if g == "No"]
    tpr = sum(s > tau for s in pos) / len(pos) if pos else 0.0
    tnr = sum(s <= tau for s in neg) / len(neg) if neg else 0.0
    return (tpr + tnr) / 2


def pick_threshold(scores, gts):
    """Search midpoints between sorted scores; ties broken by the threshold closest to 0."""
    uniq = sorted(set(scores))
    cands = [0.0] + [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
    best = max(cands, key=lambda t: (balanced_accuracy(scores, gts, t), -abs(t)))
    return best, balanced_accuracy(scores, gts, best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--calib_file", type=str, default=None, help="default: train split of the dataset")
    parser.add_argument("--num_calib", type=int, default=300)
    parser.add_argument("--eval_json", type=str, required=True, help="output json of run_clean_eval.py")
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--token_mode", type=str, default="nospace")
    args = parser.parse_args()

    with open(args.eval_json, encoding="utf-8") as f:
        eval_data = json.load(f)
    eval_cfg = eval_data["config"]
    if eval_cfg.get("token_mode", "nospace") != args.token_mode:
        print(f"[WARN] eval_json used token_mode={eval_cfg.get('token_mode')}, calibration uses {args.token_mode}")

    dataset_root = resolve_dataset_root(args.dataset_root)
    calib_file = args.calib_file or CALIB_FILE_MAP.get(args.dataset_root, "train_vqa.json")
    with open(f"{dataset_root}/{calib_file}", encoding="utf-8") as f:
        calib_all = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(calib_all)
    calib = calib_all if args.num_calib == -1 else calib_all[:args.num_calib]
    n_yes = sum(d["answer"] == "Yes" for d in calib)
    print(f"Calibration set: {calib_file} | {len(calib)} samples (Yes={n_yes}, No={len(calib) - n_yes})", flush=True)

    model, processor = load_model(resolve_model_name(args.model_name), args.min_pixels, args.max_pixels)
    yes_id, no_id = get_yes_no_token_ids(processor, args.token_mode)

    calib_scores = {m: [] for m in SCORE_METHODS}
    calib_gts = []
    t0 = time.time()
    for i, item in enumerate(calib):
        image_path = f"{dataset_root}/{item['image']}"
        if not os.path.exists(image_path):
            print(f"  [{i+1}/{len(calib)}] missing image {image_path}, skipped", flush=True)
            continue
        inputs = build_inputs(processor, image_path, item["question"], model.device)
        ans, conf = get_answer_with_confidence(model, processor, inputs, inputs.pixel_values, yes_id, no_id)
        _, text_votes = paraphrase_defense_weighted(
            model, processor, image_path, extract_caption(item["question"]), inputs.pixel_values, yes_id, no_id)
        calib_scores["clean_logit"].append(conf if ans == "Yes" else -conf)
        calib_scores["text_w"].append(vote_score(text_votes))
        calib_gts.append(item["answer"])
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(calib)}] {time.time() - t0:.0f}s", flush=True)
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    taus = {}
    print("\n===== Thresholds chosen on calibration (train) set =====")
    for m in SCORE_METHODS:
        tau, bacc = pick_threshold(calib_scores[m], calib_gts)
        bacc0 = balanced_accuracy(calib_scores[m], calib_gts, 0.0)
        taus[m] = tau
        print(f"{m:12s} tau={tau:+.3f}  balanced acc on train: {bacc0:.2%} (tau=0) -> {bacc:.2%}")

    # Apply to the dev scores saved by run_clean_eval.py
    records = eval_data["results"]
    for r in records:
        for m in SCORE_METHODS:
            r["preds"][f"{m}_cal"] = "Yes" if r["scores"][m] > taus[m] else "No"
    gts = [r["ground_truth"] for r in records]
    methods = ["clean_gen", "text"] + [x for m in SCORE_METHODS for x in (m, f"{m}_cal")]
    metrics = {m: binary_metrics([r["preds"][m] for r in records], gts) for m in methods}
    effects = {m: correction_and_harm(records, m) for m in methods if m != "clean_gen"}

    print(f"\n===== Dev results ({os.path.basename(args.eval_json)}) =====")
    print(f"{'method':16s} {'acc':>7s} {'bal_acc':>8s} {'f1':>6s} {'recall':>7s} {'yes%':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s}")
    for m in methods:
        x = metrics[m]
        tpr = x["TP"] / (x["TP"] + x["FN"]) if x["TP"] + x["FN"] else 0.0
        tnr = x["TN"] / (x["TN"] + x["FP"]) if x["TN"] + x["FP"] else 0.0
        x["balanced_accuracy"] = (tpr + tnr) / 2
        print(f"{m:16s} {x['accuracy']:7.2%} {x['balanced_accuracy']:8.2%} {x['f1']:6.3f} {x['recall']:7.2%} "
              f"{x['yes_ratio']:6.2%} {x['TP']:4d} {x['FN']:4d} {x['FP']:4d} {x['TN']:4d}")

    print("\n===== Correction / harm vs. undefended generate answer =====")
    print(f"{'method':16s} {'fix gt=Yes':>14s} {'fix gt=No':>14s} {'harm gt=Yes':>14s} {'harm gt=No':>14s}")
    for m, e in effects.items():
        print(f"{m:16s} {fmt_rate(e['corrected_gtYes']):>14s} {fmt_rate(e['corrected_gtNo']):>14s} "
              f"{fmt_rate(e['harmed_gtYes']):>14s} {fmt_rate(e['harmed_gtNo']):>14s}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR, "calib_" + os.path.basename(args.eval_json).replace(".json", "")
        + f"_calib{args.num_calib}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "calib_file": calib_file, "thresholds": taus,
                   "calib_scores": calib_scores, "calib_gts": calib_gts,
                   "dev_metrics": metrics, "dev_effects": effects}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
