# run_error_correction_experiment.py
"""
測試:模型在完全乾淨(沒有任何攻擊)的圖片+文字下,本來就判斷錯誤的樣本,
套用防禦機制(換句話問+投票、疊加雜訊+投票)後,能不能被修正成正確答案。
不涉及攻擊,純粹測防禦機制對「模型固有誤判」的修正能力。
"""
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


def load_and_sample(data_path, num_samples, seed):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return shuffled if num_samples == -1 else shuffled[:num_samples]


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=200,
                         help="這裡不篩選 Yes/No,兩種標籤都抽,因為要找的是「模型本來就答錯」的樣本")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"errorcorrection_{args.dataset_root}_{model_name.split('/')[-1]}_n{args.num_samples}_"
        f"seed{args.seed}_noise{args.noise_std}"
        f"{'_' + args.config_name if args.config_name else ''}.json"
    )

    subset = load_and_sample(f"{dataset_root}/{data_file}", args.num_samples, args.seed)
    print(f"Loaded {len(subset)} samples (mixed Yes/No labels)", flush=True)
    print(f"Output will be saved to: {output_path}", flush=True)

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)
    yes_id, no_id = get_yes_no_token_ids(processor)

    results = []
    overall_start = time.time()
    n_wrong_found = 0

    for i, item in enumerate(subset):
        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]
        caption = extract_caption(question)
        ground_truth = item["answer"]

        inputs = build_inputs(processor, image_path, question, model.device)
        clean_output = generate_with_pixel_values(model, processor, inputs, inputs.pixel_values, args.max_new_tokens)
        clean_pred = parse_answer(clean_output)

        # 只挑「模型本來就答錯」的樣本(不管是 Yes/No 哪個方向答錯)
        if clean_pred == ground_truth:
            continue

        n_wrong_found += 1
        sample_start = time.time()

        text_pred, _ = paraphrase_defense(model, processor, image_path, caption, inputs.pixel_values, args.max_new_tokens)
        text_weighted_pred, _ = paraphrase_defense_weighted(
            model, processor, image_path, caption, inputs.pixel_values, yes_id, no_id, args.max_new_tokens
        )
        pixel_pred, _ = randomized_smoothing_defense(
            model, processor, inputs, inputs.pixel_values,
            num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        pixel_weighted_pred, _ = randomized_smoothing_defense_weighted(
            model, processor, inputs, inputs.pixel_values, yes_id, no_id,
            num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        combo_pred, _ = combined_defense(
            model, processor, image_path, caption, inputs, inputs.pixel_values,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        combo_weighted_pred, _ = combined_defense_weighted(
            model, processor, image_path, caption, inputs, inputs.pixel_values,
            yes_id, no_id,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )

        sample_elapsed = time.time() - sample_start

        record = {
            "image": item["image"],
            "ground_truth": ground_truth,
            "clean_pred": clean_pred,  # 這裡一定跟 ground_truth 不同,因為是篩選出來的錯誤樣本
            "text_pred": text_pred,
            "text_corrected": text_pred == ground_truth,
            "text_weighted_pred": text_weighted_pred,
            "text_weighted_corrected": text_weighted_pred == ground_truth,
            "pixel_pred": pixel_pred,
            "pixel_corrected": pixel_pred == ground_truth,
            "pixel_weighted_pred": pixel_weighted_pred,
            "pixel_weighted_corrected": pixel_weighted_pred == ground_truth,
            "combo_pred": combo_pred,
            "combo_corrected": combo_pred == ground_truth,
            "combo_weighted_pred": combo_weighted_pred,
            "combo_weighted_corrected": combo_weighted_pred == ground_truth,
            "time_sec": round(sample_elapsed, 2),
        }
        results.append(record)

        print(f"  [{i+1}/{len(subset)}] wrong sample found (gt={ground_truth}, clean_pred={clean_pred}) | "
              f"text={text_pred:4s} text_w={text_weighted_pred:4s} "
              f"pixel={pixel_pred:4s} pixel_w={pixel_weighted_pred:4s} "
              f"combo={combo_pred:4s} combo_w={combo_weighted_pred:4s} "
              f"({sample_elapsed:.1f}s)", flush=True)

        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - overall_start

    def rate(key):
        if not results:
            return None
        return sum(r[key] for r in results) / len(results)

    print("\n===== Error Correction Summary =====", flush=True)
    print(f"Total samples scanned: {len(subset)}")
    print(f"Samples where model was originally wrong: {n_wrong_found}")
    print(f"Text correction rate:          {rate('text_corrected')}")
    print(f"Text-Weighted correction rate: {rate('text_weighted_corrected')}")
    print(f"Pixel correction rate:         {rate('pixel_corrected')}")
    print(f"Pixel-Weighted correction rate: {rate('pixel_weighted_corrected')}")
    print(f"Combo correction rate:         {rate('combo_corrected')}")
    print(f"Combo-Weighted correction rate: {rate('combo_weighted_corrected')}")
    print(f"Total time: {total_elapsed:.1f}s", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "total_scanned": len(subset),
            "n_wrong_found": n_wrong_found,
            "text_correction_rate": rate("text_corrected"),
            "text_weighted_correction_rate": rate("text_weighted_corrected"),
            "pixel_correction_rate": rate("pixel_corrected"),
            "pixel_weighted_correction_rate": rate("pixel_weighted_corrected"),
            "combo_correction_rate": rate("combo_corrected"),
            "combo_weighted_correction_rate": rate("combo_weighted_corrected"),
            "total_time_sec": total_elapsed,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()