# run_experiment.py
import argparse
import json
import os
import time
import random
import gc
import numpy as np
import torch

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, get_yes_no_token_ids, build_inputs, generate_with_pixel_values, parse_answer
from attack import pgd_attack_first_token, apply_text_attack
from defense import (
    paraphrase_defense, paraphrase_defense_weighted,
    randomized_smoothing_defense, randomized_smoothing_defense_weighted,
    combined_defense, combined_defense_weighted
)

OUTPUT_DIR = "./output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_sample(data_path, num_samples, seed, only_yes=False):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if only_yes:
        data = [d for d in data if d["answer"] == "Yes"]
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return shuffled if num_samples == -1 else shuffled[:num_samples]


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def build_output_filename(args, dataset_tag):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_tag = args.model_name.split("/")[-1]
    parts = [
        dataset_tag, model_tag, args.attack_mode,
        f"n{args.num_samples}", f"seed{args.seed}",
        f"eps{args.epsilon}", f"a{args.alpha}", f"s{args.num_steps}",
        f"noise{args.noise_std}", f"ns{args.num_noise_samples}",
        f"tok{args.token_mode}",
    ]
    if args.attack_mode in ("text", "both"):
        parts.append(args.text_attack_type)
    if args.config_name:
        parts.append(args.config_name)
    return os.path.join(OUTPUT_DIR, "_".join(parts) + ".json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="例如 FB 或 HarMeme")
    parser.add_argument("--data_file", type=str, default=None,
                         help="不指定就依資料集自動選預設檔案")
    parser.add_argument("--model_name", type=str, default="qwen2b", help="例如 qwen2b 或 qwen7b")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--attack_mode", type=str, default="image",
                         help="image(只攻圖片,PGD) / text(只攻文字) / both(雙重攻擊)")
    parser.add_argument("--text_attack_type", type=str, default="leetspeak",
                         help="attack_mode 包含 text 時使用: leetspeak 或 char_swap")
    parser.add_argument("--text_corruption_rate", type=float, default=0.5,
                         help="leetspeak 攻擊的替換機率,0.0~1.0,越高攻擊越強")
    parser.add_argument("--text_num_swaps", type=int, default=3,
                         help="char_swap 攻擊要交換幾組相鄰字母,越多攻擊越強")
    parser.add_argument("--token_mode", type=str, default="nospace",
                         help="Yes/No token spelling for logit-based steps: nospace (correct) / lse / space (old, buggy)")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_tag = args.dataset_root
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    args.model_name = resolve_model_name(args.model_name)

    output_path = args.output_file if args.output_file else build_output_filename(args, dataset_tag)

    subset = load_and_sample(f"{dataset_root}/{data_file}", args.num_samples, args.seed, only_yes=True)
    print(f"Dataset: {dataset_tag} | Data file: {data_file}", flush=True)
    print(f"Loaded {len(subset)} samples (ground truth = Yes)", flush=True)
    print(f"Attack mode: {args.attack_mode}"
          + (f" (text attack: {args.text_attack_type})" if args.attack_mode in ("text", "both") else ""),
          flush=True)
    print(f"Output will be saved to: {output_path}", flush=True)

    model, processor = load_model(args.model_name, args.min_pixels, args.max_pixels)
    yes_id, no_id = get_yes_no_token_ids(processor, args.token_mode)
    print(f"Token mode: {args.token_mode} | yes_ids={yes_id} no_ids={no_id}", flush=True)

    results = []
    overall_start = time.time()

    for i, item in enumerate(subset):
        sample_start = time.time()
        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]
        caption = extract_caption(question)

        inputs = build_inputs(processor, image_path, question, model.device)
        clean_output = generate_with_pixel_values(model, processor, inputs, inputs.pixel_values, args.max_new_tokens)
        clean_pred = parse_answer(clean_output)

        if clean_pred != "Yes":
            print(f"  [{i+1}/{len(subset)}] skipped (clean_pred={clean_pred} != Yes)", flush=True)
            continue

        # ---- 依照 attack_mode 決定圖片跟文字要不要被攻擊 ----
        if args.attack_mode in ("image", "both"):
            pixel_values_final = pgd_attack_first_token(
                model, inputs, yes_id, no_id, args.epsilon, args.alpha, args.num_steps, verbose=False
            )
        else:
            pixel_values_final = inputs.pixel_values  # 圖片維持乾淨

        if args.attack_mode in ("text", "both"):
            attacked_caption = apply_text_attack(
                caption, args.text_attack_type, seed=args.seed * 1000 + i,
                corruption_rate=args.text_corruption_rate, num_swaps=args.text_num_swaps
            )
        else:
            attacked_caption = caption

        if args.attack_mode in ("text", "both"):
            attacked_question = question.replace(caption, attacked_caption)
            attack_inputs = build_inputs(processor, image_path, attacked_question, model.device)
            print(f"    [DEBUG] original_caption='{caption}'", flush=True)
            print(f"    [DEBUG] attacked_caption='{attacked_caption}'", flush=True)
            print(f"    [DEBUG] caption_changed={caption != attacked_caption}", flush=True)
        else:
            attack_inputs = inputs

        attacked_output = generate_with_pixel_values(
            model, processor, attack_inputs, pixel_values_final, args.max_new_tokens
        )
        attacked_pred = parse_answer(attacked_output)

        # ---- 防禦:全部吃「攻擊後的版本」(caption / inputs / pixel_values 都已依 attack_mode 決定) ----
        text_pred, text_votes = paraphrase_defense(
            model, processor, image_path, attacked_caption, pixel_values_final, args.max_new_tokens
        )
        text_weighted_pred, text_weighted_votes = paraphrase_defense_weighted(
            model, processor, image_path, attacked_caption, pixel_values_final,
            yes_id, no_id, args.max_new_tokens
        )
        pixel_pred, pixel_votes = randomized_smoothing_defense(
            model, processor, attack_inputs, pixel_values_final,
            num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        pixel_weighted_pred, pixel_weighted_votes = randomized_smoothing_defense_weighted(
            model, processor, attack_inputs, pixel_values_final, yes_id, no_id,
            num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        combo_pred, combo_votes = combined_defense(
            model, processor, image_path, attacked_caption, attack_inputs, pixel_values_final,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        combo_weighted_pred, combo_weighted_votes = combined_defense_weighted(
            model, processor, image_path, attacked_caption, attack_inputs, pixel_values_final,
            yes_id, no_id,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )

        sample_elapsed = time.time() - sample_start

        record = {
            "image": item["image"],
            "ground_truth": item["answer"],
            "attack_mode": args.attack_mode,
            "original_caption": caption,
            "attacked_caption": attacked_caption,
            "clean_pred": clean_pred,
            "attacked_pred": attacked_pred,
            "attack_success": attacked_pred != "Yes",
            "text_defense_pred": text_pred,
            "text_defense_recovered": (attacked_pred != "Yes") and (text_pred == "Yes"),
            "text_weighted_pred": text_weighted_pred,
            "text_weighted_recovered": (attacked_pred != "Yes") and (text_weighted_pred == "Yes"),
            "pixel_defense_pred": pixel_pred,
            "pixel_defense_recovered": (attacked_pred != "Yes") and (pixel_pred == "Yes"),
            "pixel_weighted_pred": pixel_weighted_pred,
            "pixel_weighted_recovered": (attacked_pred != "Yes") and (pixel_weighted_pred == "Yes"),
            "combo_defense_pred": combo_pred,
            "combo_defense_recovered": (attacked_pred != "Yes") and (combo_pred == "Yes"),
            "combo_weighted_pred": combo_weighted_pred,
            "combo_weighted_recovered": (attacked_pred != "Yes") and (combo_weighted_pred == "Yes"),
            "time_sec": round(sample_elapsed, 2),
        }
        results.append(record)

        print(f"  [{i+1}/{len(subset)}] clean=Yes -> attacked={attacked_pred:4s} | "
              f"text={text_pred:4s} text_w={text_weighted_pred:4s} "
              f"pixel={pixel_pred:4s} pixel_w={pixel_weighted_pred:4s} "
              f"combo={combo_pred:4s} combo_w={combo_weighted_pred:4s} "
              f"({sample_elapsed:.1f}s)", flush=True)

        del inputs, attack_inputs, pixel_values_final
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - overall_start

    def rate(key):
        attacked = [r for r in results if r["attack_success"]]
        if not attacked:
            return None
        return sum(r[key] for r in attacked) / len(attacked)

    n_evaluated = len(results)
    n_attacked = sum(r["attack_success"] for r in results)

    print("\n===== Summary =====", flush=True)
    print(f"Dataset: {dataset_tag} | Model: {args.model_name} | Attack mode: {args.attack_mode}")
    print(f"Clean-correct samples evaluated: {n_evaluated}")
    if n_evaluated:
        print(f"Attack success count: {n_attacked} ({n_attacked/n_evaluated:.2%} of evaluated)")
    print(f"Text defense recovery rate:  {rate('text_defense_recovered')}")
    print(f"Text-Weighted defense recovery rate: {rate('text_weighted_recovered')}")
    print(f"Pixel defense recovery rate: {rate('pixel_defense_recovered')}")
    print(f"Pixel-Weighted defense recovery rate: {rate('pixel_weighted_recovered')}")
    print(f"Combo defense recovery rate: {rate('combo_defense_recovered')}")
    print(f"Combo-Weighted defense recovery rate: {rate('combo_weighted_recovered')}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/max(1,n_evaluated):.1f}s/sample avg)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "dataset_tag": dataset_tag,
            "n_evaluated": n_evaluated,
            "n_attacked": n_attacked,
            "attack_success_rate": n_attacked / n_evaluated if n_evaluated else None,
            "text_defense_recovery_rate": rate("text_defense_recovered"),
            "text_weighted_defense_recovery_rate": rate("text_weighted_recovered"),
            "pixel_defense_recovery_rate": rate("pixel_defense_recovered"),
            "pixel_weighted_defense_recovery_rate": rate("pixel_weighted_recovered"),
            "combo_defense_recovery_rate": rate("combo_defense_recovered"),
            "combo_weighted_defense_recovery_rate": rate("combo_weighted_recovered"),
            "total_time_sec": total_elapsed,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()