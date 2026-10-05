# run_textfooler_experiment.py
"""
在多筆樣本上跑 TextFooler 攻擊,統計成功率。
先只驗證攻擊本身,不接防禦機制。
"""
import argparse
import json
import os
import time
import random

from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, get_yes_no_token_ids, build_inputs, generate_with_pixel_values, parse_answer
from textattack_wrapper import VLMTextAttackWrapper

OUTPUT_DIR = "./output"


def build_question_with_caption(caption):
    return (f'Given the meme image with the caption "{caption}", does this meme contain '
            f'hateful content targeting a specific group (e.g. race, religion, gender, '
            f'disability)? Answer only Yes or No.')


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def load_and_sample(data_path, num_samples, seed, only_yes=False):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if only_yes:
        data = [d for d in data if d["answer"] == "Yes"]
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return shuffled if num_samples == -1 else shuffled[:num_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="FB")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query_budget", type=int, default=200,
                         help="每句話最多讓 TextFooler 嘗試幾次查詢")
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"textfooler_{args.dataset_root}_{model_name.split('/')[-1]}_n{args.num_samples}_"
        f"seed{args.seed}_qb{args.query_budget}"
        f"{'_' + args.config_name if args.config_name else ''}.json"
    )

    subset = load_and_sample(f"{dataset_root}/{data_file}", args.num_samples, args.seed, only_yes=True)
    print(f"Loaded {len(subset)} samples (ground truth = Yes)", flush=True)
    print(f"Output will be saved to: {output_path}", flush=True)

    model, processor = load_model(model_name)
    yes_id, no_id = get_yes_no_token_ids(processor)

    results = []
    overall_start = time.time()

    for i, item in enumerate(subset):
        sample_start = time.time()
        image_path = f"{dataset_root}/{item['image']}"
        caption = extract_caption(item["question"])

        # 先確認模型在乾淨情況下答不答對
        clean_inputs = build_inputs(processor, image_path, item["question"], model.device)
        clean_output = generate_with_pixel_values(model, processor, clean_inputs, clean_inputs.pixel_values, 10)
        clean_pred = parse_answer(clean_output)

        if clean_pred != "Yes":
            print(f"  [{i+1}/{len(subset)}] skipped (clean_pred={clean_pred} != Yes)", flush=True)
            continue

        print(f"  [{i+1}/{len(subset)}] running TextFooler on: '{caption[:50]}...'", flush=True)

        wrapper = VLMTextAttackWrapper(model, processor, image_path, yes_id, no_id, build_question_with_caption)
        dataset = Dataset([(caption, 1)])
        attack = TextFoolerJin2019.build(wrapper)
        attack_args = AttackArgs(num_examples=1, query_budget=args.query_budget, disable_stdout=True)
        attacker = Attacker(attack, dataset, attack_args)

        attack_results = attacker.attack_dataset()
        attack_result = attack_results[0]

        result_type = type(attack_result).__name__  # SuccessfulAttackResult / FailedAttackResult / SkippedAttackResult
        attack_success = (result_type == "SuccessfulAttackResult")

        perturbed_caption = attack_result.perturbed_text() if attack_success else caption
        num_queries = attack_result.num_queries

        sample_elapsed = time.time() - sample_start

        record = {
            "image": item["image"],
            "original_caption": caption,
            "attack_success": attack_success,
            "perturbed_caption": perturbed_caption,
            "num_queries": num_queries,
            "time_sec": round(sample_elapsed, 2),
        }
        results.append(record)

        print(f"    -> {'SUCCESS' if attack_success else 'FAILED'} "
              f"(queries={num_queries}, {sample_elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - overall_start

    n_evaluated = len(results)
    n_success = sum(r["attack_success"] for r in results)

    print("\n===== TextFooler Summary =====", flush=True)
    print(f"Clean-correct samples evaluated: {n_evaluated}")
    if n_evaluated:
        print(f"Attack success count: {n_success} ({n_success/n_evaluated:.2%})")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/max(1,n_evaluated):.1f}s/sample avg)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "n_evaluated": n_evaluated,
            "n_success": n_success,
            "attack_success_rate": n_success / n_evaluated if n_evaluated else None,
            "total_time_sec": total_elapsed,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()