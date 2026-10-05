# clean_control_check.py
"""
獨立檢查:防禦機制對「原本就答對的乾淨圖片」有沒有副作用。
不涉及攻擊,只測「加防禦前 vs 加防禦後」,乾淨圖片的答案會不會被雜訊搞壞。
"""
# clean_control_check.py
import argparse
import json
import time
import random
import gc
import numpy as np
import torch

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, build_inputs, generate_with_pixel_values, parse_answer
from defense import paraphrase_defense, randomized_smoothing_defense, combined_defense


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)

    with open(f"{dataset_root}/{data_file}", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(args.seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    subset = shuffled[:args.num_samples]

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)

    n_clean_correct = 0
    n_text_still_correct = 0
    n_pixel_still_correct = 0
    n_combo_still_correct = 0

    start = time.time()
    for i, item in enumerate(subset):
        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]
        caption = extract_caption(question)
        gt = item["answer"]

        inputs = build_inputs(processor, image_path, question, model.device)
        clean_output = generate_with_pixel_values(model, processor, inputs, inputs.pixel_values, args.max_new_tokens)
        clean_pred = parse_answer(clean_output)

        if clean_pred != gt:
            print(f"  [{i+1}/{len(subset)}] skipped (clean model already wrong)", flush=True)
            continue

        n_clean_correct += 1

        text_pred, _ = paraphrase_defense(model, processor, image_path, caption, inputs.pixel_values, args.max_new_tokens)
        pixel_pred, _ = randomized_smoothing_defense(
            model, processor, inputs, inputs.pixel_values,
            num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )
        combo_pred, _ = combined_defense(
            model, processor, image_path, caption, inputs, inputs.pixel_values,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=args.seed * 1000 + i
        )

        n_text_still_correct += (text_pred == gt)
        n_pixel_still_correct += (pixel_pred == gt)
        n_combo_still_correct += (combo_pred == gt)

        print(f"  [{i+1}/{len(subset)}] gt={gt} clean={clean_pred} | "
              f"text={text_pred} pixel={pixel_pred} combo={combo_pred}", flush=True)

        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    print("\n===== Clean-Control Summary =====")
    print(f"Samples where model was originally correct: {n_clean_correct}")
    if n_clean_correct:
        print(f"Still correct after text defense:  {n_text_still_correct}/{n_clean_correct} ({n_text_still_correct/n_clean_correct:.2%})")
        print(f"Still correct after pixel defense: {n_pixel_still_correct}/{n_clean_correct} ({n_pixel_still_correct/n_clean_correct:.2%})")
        print(f"Still correct after combo defense: {n_combo_still_correct}/{n_clean_correct} ({n_combo_still_correct/n_clean_correct:.2%})")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()