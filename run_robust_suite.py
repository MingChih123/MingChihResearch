# run_robust_suite.py
"""
攻擊組合 × 防禦組合,一次跑完,找出「哪個防禦在哪種攻擊下失效」。全部不訓練。

兩種模式:
  1. 攻擊模式(預設):只用「標準答案是仇恨、模型原本也答對」的迷因,
     每種攻擊做出一張對抗圖(存成 PNG 再讀回,和真實上傳一樣),再用每種防禦判斷。
  2. --clean_only:不攻擊,在 Yes/No 混合的樣本上跑所有防禦,量「對正常圖片的副作用」。

攻擊(--attacks,用逗號分隔):
  pgd         梯度攻擊,L-inf eps/255(和 run_image_attack.py 一樣)
  pgd_jpeg    會撐過 JPEG 的梯度攻擊:每一步先把圖做 JPEG 壓縮再算梯度(BPDA 直通估計)
  saltpepper  椒鹽雜訊:隨機把一部分像素變成黑或白(參考 HateProof 的 SaltPepper-I)
  spread      像素擾動:每個像素和附近隨機一個像素交換位置(參考 HateProof 的 Spread)
  (saltpepper / spread 的強度是我們自己設定的近似值,HateProof 用 GIMP 做,沒有給確切參數)

防禦(--defenses,用逗號分隔):
  none         不防禦
  jpeg         JPEG 重新壓縮一次(簡單 baseline)
  text         3 種問法 + 多數決
  text_anyyes  3 種問法,任一說 Yes 就判仇恨
  noise        加高斯雜訊 5 次 + 多數決
  transform    【新】5 種圖片轉換(JPEG50 / JPEG75 / 縮小再放大 / 模糊 / 中值濾波)+ 多數決
  tq           【新】5 種轉換 × 3 種問法 = 15 票:
                 tq_majority = 15 票多數決
                 tq_anyq     = 每個問法先對 5 種轉換多數決,再「任一問法說 Yes 就判仇恨」
               另外記錄投票是否一致(不一致 = 可以送人工審核)

用法(CMD):
  # 開發用:先在 train 上跑(之後在 dev 報最終結果)
  python -u run_robust_suite.py --dataset_root FB --data_file train_vqa.json --num_samples 150 --seed 0
  # 正常圖片的副作用(Yes/No 混合,不攻擊)
  python -u run_robust_suite.py --dataset_root FB --data_file train_vqa.json --num_samples 200 --seed 0 --clean_only
"""
import argparse
import gc
import io
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image, ImageFilter

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import (load_model, get_yes_no_token_ids, yes_no_logits,
                    build_inputs, generate_with_pixel_values, parse_answer)
from defense import make_paraphrase_questions, majority_vote, randomized_smoothing_defense
from run_image_attack import (PixelValuesBuilder, pil_to_tensor, tensor_to_pil, jpeg_reencode,
                              pgd_image_space, extract_caption)

OUTPUT_DIR = "./output"
ALL_ATTACKS = ["pgd", "pgd_jpeg", "saltpepper", "spread"]
ALL_DEFENSES = ["none", "jpeg", "text", "text_anyyes", "noise", "transform", "tq"]

# Image transformations used by the transform / tq defenses (odd count -> no ties)
TRANSFORMS = {
    "jpeg50": lambda im: jpeg_reencode(im, 50),
    "jpeg75": lambda im: jpeg_reencode(im, 75),
    "resize50": lambda im: im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.BICUBIC)
                             .resize(im.size, Image.BICUBIC),
    "blur1": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1)),
    "median3": lambda im: im.filter(ImageFilter.MedianFilter(size=3)),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- attacks ----------------

def jpeg_tensor(x, quality):
    """Real (non-differentiable) JPEG of a (3,H,W) [0,1] tensor."""
    return pil_to_tensor(jpeg_reencode(tensor_to_pil(x), quality), x.device)


def pgd_jpeg_aware(model, inputs, builder, x0, yes_ids, no_ids, eps, alpha, steps, quality):
    """PGD through a JPEG step using BPDA: forward uses the real JPEG output,
    backward treats JPEG as identity. The result is meant to survive re-compression."""
    delta = torch.zeros_like(x0)
    gap = None
    for _ in range(steps):
        delta.requires_grad_(True)
        x = (x0 + delta).clamp(0, 1)
        xj = jpeg_tensor(x.detach(), quality)
        x_in = x + (xj - x).detach()  # straight-through
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        pixel_values=builder(x_in), image_grid_thw=inputs.image_grid_thw)
        logit_yes, logit_no = yes_no_logits(outputs.logits[0, -1, :], yes_ids, no_ids)
        loss = logit_no - logit_yes
        loss.backward()
        grad = delta.grad
        gap = loss.item()
        if grad is None:
            break
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = delta.clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0
        delta = delta.detach()
        del outputs, logit_yes, logit_no, loss, grad
        torch.cuda.empty_cache()
    return (x0 + delta).clamp(0, 1).detach(), gap


def salt_pepper(img, amount, rng):
    arr = np.asarray(img).copy()
    mask = rng.random(arr.shape[:2])
    arr[mask < amount / 2] = 0
    arr[(mask >= amount / 2) & (mask < amount)] = 255
    return Image.fromarray(arr)


def spread(img, radius, rng):
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    ys = np.clip(ys + rng.integers(-radius, radius + 1, size=(h, w)), 0, h - 1)
    xs = np.clip(xs + rng.integers(-radius, radius + 1, size=(h, w)), 0, w - 1)
    return Image.fromarray(arr[ys, xs])


# ---------------- defenses ----------------

def predict_path(model, processor, path, question, max_new_tokens):
    inputs = build_inputs(processor, path, question, model.device)
    out = parse_answer(generate_with_pixel_values(model, processor, inputs, inputs.pixel_values, max_new_tokens))
    del inputs
    return out


def run_defenses(model, processor, img_path, question, caption, defenses, args, seed_offset, work_dir, tag):
    """Returns {defense_name: prediction} plus vote details for one image file."""
    preds, votes = {}, {}
    questions = make_paraphrase_questions(caption)

    if "none" in defenses:
        preds["none"] = predict_path(model, processor, img_path, question, args.max_new_tokens)

    if "jpeg" in defenses:
        jp = os.path.join(work_dir, f"{tag}_jpeg{args.jpeg_quality}.jpg")
        jpeg_reencode(Image.open(img_path).convert("RGB"), args.jpeg_quality).save(jp, quality=100)
        preds["jpeg"] = predict_path(model, processor, jp, question, args.max_new_tokens)

    if "text" in defenses or "text_anyyes" in defenses:
        tv = [predict_path(model, processor, img_path, q, args.max_new_tokens) for q in questions]
        votes["text"] = tv
        preds["text"] = majority_vote(tv)
        preds["text_anyyes"] = "Yes" if "Yes" in tv else "No"

    if "noise" in defenses:
        inputs = build_inputs(processor, img_path, question, model.device)
        preds["noise"], nv = randomized_smoothing_defense(
            model, processor, inputs, inputs.pixel_values, num_samples=args.num_noise_samples,
            noise_std=args.noise_std, max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)
        votes["noise"] = nv
        del inputs

    if "transform" in defenses or "tq" in defenses:
        base = Image.open(img_path).convert("RGB")
        t_paths = {}
        for name, fn in TRANSFORMS.items():
            p = os.path.join(work_dir, f"{tag}_t_{name}.png")
            fn(base).save(p)
            t_paths[name] = p

        if "transform" in defenses:
            tv = [predict_path(model, processor, p, question, args.max_new_tokens) for p in t_paths.values()]
            votes["transform"] = tv
            preds["transform"] = majority_vote(tv)

        if "tq" in defenses:
            grid = {}  # question index -> votes over transforms
            for qi, q in enumerate(questions):
                grid[qi] = [predict_path(model, processor, p, q, args.max_new_tokens) for p in t_paths.values()]
            flat = [v for vs in grid.values() for v in vs]
            per_q = [majority_vote(vs) for vs in grid.values()]
            votes["tq"] = grid
            preds["tq_majority"] = majority_vote(flat)
            preds["tq_anyq"] = "Yes" if "Yes" in per_q else "No"
            valid = [v for v in flat if v in ("Yes", "No")]
            preds["tq_unanimous"] = len(set(valid)) == 1 and len(valid) == len(flat)

    return preds, votes


# ---------------- summaries ----------------

def binary_metrics(preds, gts):
    tp = sum(p == "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fn = sum(p != "Yes" and g == "Yes" for p, g in zip(preds, gts))
    fp = sum(p == "Yes" and g == "No" for p, g in zip(preds, gts))
    tn = sum(p == "No" and g == "No" for p, g in zip(preds, gts))
    n = len(preds)
    return {"n": n, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "accuracy": (tp + tn) / n if n else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def defense_keys(defenses):
    """Prediction keys reported in the summary, in order."""
    keys = []
    for d in defenses:
        if d == "tq":
            keys += ["tq_majority", "tq_anyq"]
        elif d == "text":
            keys += ["text", "text_anyyes"]  # both come from the same 3 answers
        else:
            keys.append(d)
    return list(dict.fromkeys(keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None, help="use train_vqa.json while developing")
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=150,
                        help="attack mode: number of gt=Yes memes to scan; clean mode: mixed memes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean_only", action="store_true", help="no attack; mixed Yes/No; measure side effects")
    parser.add_argument("--attacks", type=str, default=",".join(ALL_ATTACKS))
    parser.add_argument("--defenses", type=str, default=",".join(ALL_DEFENSES))
    parser.add_argument("--eps", type=float, default=8.0, help="PGD L-inf budget in /255")
    parser.add_argument("--alpha", type=float, default=2.0, help="PGD step in /255")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--saltpepper_amount", type=float, default=0.05, help="fraction of pixels set to black/white")
    parser.add_argument("--spread_radius", type=int, default=3, help="max pixel displacement")
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--token_mode", type=str, default="nospace")
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()

    attacks = [a for a in args.attacks.split(",") if a]
    defenses = [d for d in args.defenses.split(",") if d]
    for a in attacks:
        assert a in ALL_ATTACKS, f"unknown attack {a}"
    for d in defenses:
        assert d in ALL_DEFENSES, f"unknown defense {d}"
    keys = defense_keys(defenses)

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)
    split = os.path.splitext(os.path.basename(data_file))[0]
    mode = "clean" if args.clean_only else "attack"
    ds_tag = os.path.basename(os.path.normpath(args.dataset_root))
    tag = (f"robust_{mode}_{ds_tag}_{split}_{model_name.split('/')[-1]}_n{args.num_samples}_seed{args.seed}"
           + ("" if args.clean_only else f"_eps{args.eps:g}_s{args.num_steps}")
           + (f"_{args.config_name}" if args.config_name else ""))
    output_path = os.path.join(OUTPUT_DIR, tag + ".json")
    work_dir = os.path.join(OUTPUT_DIR, "robust_images", tag)
    os.makedirs(work_dir, exist_ok=True)

    with open(f"{dataset_root}/{data_file}", encoding="utf-8") as f:
        data = json.load(f)
    if not args.clean_only:
        data = [d for d in data if d["answer"] == "Yes"]
    rng = random.Random(args.seed)
    rng.shuffle(data)
    subset = data if args.num_samples == -1 else data[:args.num_samples]
    print(f"Mode: {mode} | {len(subset)} samples from {data_file} | gt counts {dict(Counter(d['answer'] for d in subset))}")
    if not args.clean_only:
        print(f"Attacks: {attacks}")
    print(f"Defenses: {keys}\nOutput: {output_path}", flush=True)

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)
    yes_ids, no_ids = get_yes_no_token_ids(processor, args.token_mode)
    builder = PixelValuesBuilder(processor, model.device)
    eps, alpha = args.eps / 255.0, args.alpha / 255.0
    np_rng = np.random.default_rng(args.seed)

    results, n_skipped = [], Counter()
    t0 = time.time()
    for i, item in enumerate(subset):
        s0 = time.time()
        image_path = f"{dataset_root}/{item['image']}"
        if not os.path.exists(image_path):
            n_skipped["missing_image"] += 1
            continue
        question, caption, gt = item["question"], extract_caption(item["question"]), item["answer"]
        name = os.path.splitext(os.path.basename(item["image"]))[0]
        seed_offset = args.seed * 1000 + i

        # Resize to the exact model resolution and save, so every variant goes through the same pipeline
        inputs = build_inputs(processor, image_path, question, model.device)
        _, gh, gw = inputs.image_grid_thw[0].tolist()
        H, W = gh * builder.patch, gw * builder.patch
        del inputs
        x0_pil = Image.open(image_path).convert("RGB").resize((W, H), Image.BICUBIC)
        x0_path = os.path.join(work_dir, f"{name}_clean.png")
        x0_pil.save(x0_path)

        rec = {"image": item["image"], "ground_truth": gt}

        if args.clean_only:
            rec["preds"], rec["votes"] = run_defenses(model, processor, x0_path, question, caption, defenses,
                                                      args, seed_offset, work_dir, f"{name}_clean")
            results.append(rec)
            print(f"  [{i+1}/{len(subset)}] gt={gt:3s} " + " ".join(f"{k}={rec['preds'].get(k)}" for k in keys)
                  + f" ({time.time() - s0:.1f}s)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue

        clean_pred = predict_path(model, processor, x0_path, question, args.max_new_tokens)
        if clean_pred != "Yes":
            n_skipped["clean_not_yes"] += 1
            print(f"  [{i+1}/{len(subset)}] skipped (clean pred={clean_pred})", flush=True)
            continue

        x0 = pil_to_tensor(x0_pil, model.device)
        x0_inputs = build_inputs(processor, x0_path, question, model.device)
        rec["attacks"] = {}
        line = []
        for atk in attacks:
            if atk == "pgd":
                xa, _ = pgd_image_space(model, x0_inputs, builder, x0, yes_ids, no_ids, eps, alpha, args.num_steps)
                adv = tensor_to_pil(xa)
            elif atk == "pgd_jpeg":
                xa, _ = pgd_jpeg_aware(model, x0_inputs, builder, x0, yes_ids, no_ids, eps, alpha,
                                       args.num_steps, args.jpeg_quality)
                adv = tensor_to_pil(xa)
            elif atk == "saltpepper":
                adv = salt_pepper(x0_pil, args.saltpepper_amount, np_rng)
            else:
                adv = spread(x0_pil, args.spread_radius, np_rng)
            adv_path = os.path.join(work_dir, f"{name}_{atk}.png")
            adv.save(adv_path)

            preds, votes = run_defenses(model, processor, adv_path, question, caption, defenses,
                                        args, seed_offset, work_dir, f"{name}_{atk}")
            # Attack success is judged without any defense
            if "none" not in preds:
                preds["none"] = predict_path(model, processor, adv_path, question, args.max_new_tokens)
            rec["attacks"][atk] = {"preds": preds, "votes": votes, "success": preds["none"] != "Yes"}
            line.append(f"{atk}:{'OK' if preds['none'] != 'Yes' else '--'}")
            gc.collect()
            torch.cuda.empty_cache()

        results.append(rec)
        print(f"  [{i+1}/{len(subset)}] " + " ".join(line) + f" ({time.time() - s0:.1f}s)", flush=True)
        del x0, x0_inputs
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    summary = {"skipped": dict(n_skipped)}
    print("\n===== Summary =====")
    if n_skipped:
        print(f"Skipped: {dict(n_skipped)}")

    if args.clean_only:
        gts = [r["ground_truth"] for r in results]
        base = [r["preds"].get("none") for r in results] if "none" in keys else None
        summary["metrics"] = {}
        print(f"Clean images, n={len(results)}")
        print(f"{'defense':14s} {'acc':>7s} {'recall':>7s} {'f1':>6s} {'fixed':>6s} {'harmed':>7s}")
        for k in keys:
            p = [r["preds"].get(k) for r in results]
            m = binary_metrics(p, gts)
            if base is not None and k != "none":
                m["fixed"] = sum(b != g and x == g for b, x, g in zip(base, p, gts))
                m["harmed"] = sum(b == g and x != g for b, x, g in zip(base, p, gts))
            summary["metrics"][k] = m
            print(f"{k:14s} {m['accuracy']:7.1%} {m['recall']:7.1%} {m['f1']:6.3f} "
                  f"{m.get('fixed', ''):>6} {m.get('harmed', ''):>7}")
        if "tq" in defenses:
            unanimous = sum(r["preds"]["tq_unanimous"] for r in results)
            summary["tq_flag_for_review_rate"] = 1 - unanimous / len(results) if results else None
            print(f"tq votes not unanimous (would go to human review): {len(results) - unanimous}/{len(results)}")
    else:
        n = len(results)
        print(f"Hateful memes the model originally detected: {n}")
        print("Table: % still detected as hateful (higher = better). Row = attack, column = defense")
        header = f"{'attack':12s} {'success':>8s} " + " ".join(f"{k:>11s}" for k in keys)
        print(header)
        summary["table"] = {}
        for atk in attacks:
            rows = [r["attacks"][atk] for r in results]
            succ = sum(x["success"] for x in rows)
            row = {"attack_success_rate": succ / n if n else None}
            cells = []
            for k in keys:
                acc = sum(x["preds"].get(k) == "Yes" for x in rows) / n if n else None
                row[k] = acc
                cells.append(f"{acc:11.1%}" if acc is not None else f"{'-':>11s}")
            if "tq" in defenses and rows:
                row["tq_flag_for_review_rate"] = 1 - sum(x["preds"]["tq_unanimous"] for x in rows) / n
            summary["table"][atk] = row
            print(f"{atk:12s} {succ / n if n else 0:8.1%} " + " ".join(cells))

    total = time.time() - t0
    print(f"Total time: {total:.1f}s", flush=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary, "total_time_sec": total,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
