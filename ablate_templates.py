# ablate_templates.py
"""
文字防禦(換問法)的消融實驗:效果是來自「投票」,還是只是某一個問法比較好?

每筆樣本用 defense.make_paraphrase_questions 的 3 個問法各問一次(generate,和多數決防禦完全一樣),
然後比較:
  - 每個問法單獨的結果(q0 = 原始問法,q1、q2 = 改寫)
  - 多數決(= text 防禦)
  - any-Yes:任一問法說 Yes 就標記(審核情境:寧可多送人工審核,也不要漏掉)
  - all-Yes:三個都說 Yes 才標記

用法(CMD):
  python -u ablate_templates.py --dataset_root FB --num_samples -1 --seed 0
"""
import argparse
import gc
import json
import os
import random
import time

import torch

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, build_inputs, generate_with_pixel_values, parse_answer
from defense import make_paraphrase_questions, majority_vote
from run_clean_eval import binary_metrics, extract_caption

OUTPUT_DIR = "./output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)

    with open(f"{dataset_root}/{data_file}", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    rng.shuffle(data)
    subset = data if args.num_samples == -1 else data[:args.num_samples]
    print(f"Loaded {len(subset)} samples from {data_file}", flush=True)

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)

    records = []
    t0 = time.time()
    for i, item in enumerate(subset):
        image_path = f"{dataset_root}/{item['image']}"
        caption = extract_caption(item["question"])
        votes = []
        pixel_values = None
        for q in make_paraphrase_questions(caption):
            inputs = build_inputs(processor, image_path, q, model.device)
            if pixel_values is None:
                pixel_values = inputs.pixel_values
            out = generate_with_pixel_values(model, processor, inputs, pixel_values, args.max_new_tokens)
            votes.append(parse_answer(out))
            del inputs
        records.append({"image": item["image"], "ground_truth": item["answer"], "votes": votes})
        print(f"  [{i+1}/{len(subset)}] gt={item['answer']:3s} votes={votes}", flush=True)
        del pixel_values
        gc.collect()
        torch.cuda.empty_cache()

    gts = [r["ground_truth"] for r in records]
    rules = {
        "q0 (original)": lambda v: v[0],
        "q1": lambda v: v[1],
        "q2": lambda v: v[2],
        "majority": majority_vote,
        "any-Yes": lambda v: "Yes" if "Yes" in v else "No",
        "all-Yes": lambda v: "Yes" if all(x == "Yes" for x in v) else "No",
    }
    metrics = {}
    print(f"\n===== Template ablation ({data_file}, n={len(records)}) =====")
    print(f"{'rule':14s} {'acc':>7s} {'f1':>6s} {'recall':>7s} {'prec':>6s} {'yes%':>6s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'TN':>4s}")
    for name, fn in rules.items():
        m = binary_metrics([fn(r["votes"]) for r in records], gts)
        metrics[name] = m
        print(f"{name:14s} {m['accuracy']:7.2%} {m['f1']:6.3f} {m['recall']:7.2%} {m['precision']:6.2%} "
              f"{m['yes_ratio']:6.2%} {m['TP']:4d} {m['FN']:4d} {m['FP']:4d} {m['TN']:4d}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split = os.path.splitext(os.path.basename(data_file))[0]
    out_path = os.path.join(OUTPUT_DIR, f"ablate_templates_{args.dataset_root}_{split}_"
                            f"{model_name.split('/')[-1]}_n{args.num_samples}_seed{args.seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "metrics": metrics, "total_time_sec": time.time() - t0,
                   "results": records}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
