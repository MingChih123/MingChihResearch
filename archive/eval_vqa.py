# eval_vqa.py
import argparse
import json
import os
import time
import random
import numpy as np
from collections import Counter
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


DATASET_ROOT_MAP = {
    "FB": "./dataset/FB",
    "HarMeme": "./dataset/HarMeme",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_answer(output_text):
    text = output_text.strip().lower()
    if text.startswith("yes"):
        return "Yes"
    elif text.startswith("no"):
        return "No"
    elif "yes" in text and "no" not in text:
        return "Yes"
    elif "no" in text and "yes" not in text:
        return "No"
    else:
        return "Unclear"


def load_model(model_name):
    print(f"Loading model: {model_name} ...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Model loaded. Device: {model.device}", flush=True)
    return model, processor


def ask_vlm(model, processor, image_path, question, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text


def resolve_dataset_root(name_or_path):
    if name_or_path in DATASET_ROOT_MAP:
        return DATASET_ROOT_MAP[name_or_path]
    return name_or_path


def load_and_sample(data_path, num_samples, seed):
    with open(data_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    if num_samples == -1:
        return shuffled
    return shuffled[:num_samples]


def compute_metrics(results):
    """
    正類(positive class)定義為 "Yes"(仇恨內容)。
    模型回答 "Unclear" 視為「沒有判定為仇恨」,歸類進負類的判斷邊,
    因為模型沒有明確指出這是仇恨內容,等同於它沒抓到。
    """
    tp = fp = fn = tn = 0

    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        pred_positive = (pred == "Yes")
        gt_positive = (gt == "Yes")

        if pred_positive and gt_positive:
            tp += 1
        elif pred_positive and not gt_positive:
            fp += 1
        elif not pred_positive and gt_positive:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default="dev_vqa.json")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = resolve_dataset_root(args.dataset_root)

    if args.output_file is None:
        tag = os.path.basename(dataset_root)
        n_tag = "all" if args.num_samples == -1 else str(args.num_samples)
        args.output_file = f"results_{tag}_n{n_tag}_seed{args.seed}.json"

    subset = load_and_sample(f"{dataset_root}/{args.data_file}", args.num_samples, args.seed)
    print(f"Total samples to run: {len(subset)}", flush=True)

    label_dist = Counter(item["answer"] for item in subset)
    print(f"Sampled label distribution: {dict(label_dist)}", flush=True)

    model, processor = load_model(args.model_name)

    results = []
    overall_start = time.time()

    for i, item in enumerate(subset):
        sample_start = time.time()

        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]

        output_text = ask_vlm(model, processor, image_path, question, args.max_new_tokens)
        predicted = parse_answer(output_text)

        sample_elapsed = time.time() - sample_start

        results.append({
            "image": item["image"],
            "question": question,
            "raw_output": output_text,
            "ground_truth": item["answer"],
            "predicted": predicted,
            "correct": predicted == item["answer"],
            "time_sec": round(sample_elapsed, 2)
        })

        print(f"  [{i+1}/{len(subset)}] pred={predicted:8s} gt={item['answer']:8s} "
              f"{'OK' if predicted == item['answer'] else 'X '} "
              f"({sample_elapsed:.2f}s) raw='{output_text.strip()}'", flush=True)

    total_elapsed = time.time() - overall_start

    metrics = compute_metrics(results)
    unclear = sum(r["predicted"] == "Unclear" for r in results)

    print("\n===== Summary =====", flush=True)
    print(f"Samples: {len(results)}")
    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1']:.2%}")
    print(f"Confusion: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"Unclear responses: {unclear} ({unclear/len(results):.2%})")
    print(f"Total time: {total_elapsed:.1f}s  ({total_elapsed/len(results):.2f}s/sample avg)")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "metrics": metrics,
            "unclear_rate": unclear / len(results),
            "total_time_sec": total_elapsed,
            "avg_time_per_sample": total_elapsed / len(results),
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()