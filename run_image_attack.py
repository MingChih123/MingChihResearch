# run_image_attack.py
"""
真實情境的圖片攻擊:擾動加在「圖片像素」上,存成真的 PNG 檔,再像使用者上傳一樣重新讀進來判斷。

和 run_experiment.py(攻擊直接改模型內部的 pixel_values 張量)的差別:
  - 擾動在 [0,1] 像素空間,L-inf 預算用 /255 表示(例如 8/255),並限制在合法像素範圍內
  - 攻擊後量化成 8-bit 存成 PNG → 重新走一般的讀圖流程(和真實上傳一樣)
  - 多一個 JPEG 重新壓縮的情境:很多平台上傳後會轉存 JPEG,攻擊要能活下來才算數;
    同時 JPEG 也是常見的簡單防禦 baseline

流程(只攻擊 gt=Yes 且乾淨時答對的樣本,威脅情境:讓仇恨迷因被判成沒問題):
  1. 用一般流程算出模型實際使用的解析度(image_grid_thw),把原圖縮放到這個大小 → x0
  2. 在 x0 上做 PGD(可微分地重現 Qwen2-VL 的正規化與切 patch),目標:讓第一個 token 的 No 勝過 Yes
  3. 量化成 uint8 存成 PNG → 重新讀檔 → 判斷是否攻擊成功
  4. 防禦全部作用在讀進來的對抗圖片上:換問法(多數決 / any-Yes)、雜訊、組合、JPEG

用法(CMD):
  python -u run_image_attack.py --dataset_root FB --num_samples -1 --seed 0
"""
import argparse
import gc
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import (load_model, get_yes_no_token_ids, yes_no_logits,
                    build_inputs, generate_with_pixel_values, parse_answer)
from defense import paraphrase_defense, randomized_smoothing_defense, combined_defense

OUTPUT_DIR = "./output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


class PixelValuesBuilder:
    """Differentiable copy of Qwen2VLImageProcessor's rescale -> normalize -> patchify
    (transformers image_processing_qwen2_vl.py::_preprocess), for an image already at target size."""

    def __init__(self, processor, device):
        ip = processor.image_processor
        self.patch = getattr(ip, "patch_size", 14)
        self.merge = getattr(ip, "merge_size", 2)
        self.temporal = getattr(ip, "temporal_patch_size", 2)
        self.mean = torch.tensor(ip.image_mean, device=device, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, img):
        """img: (3, H, W) float32 in [0, 1], H and W multiples of patch*merge."""
        c, h, w = img.shape
        x = (img - self.mean) / self.std
        x = x.unsqueeze(0).repeat(self.temporal, 1, 1, 1)  # single image -> temporal tile
        gh, gw = h // self.patch, w // self.patch
        x = x.reshape(1, self.temporal, c, gh // self.merge, self.merge, self.patch,
                      gw // self.merge, self.merge, self.patch)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gh * gw, c * self.temporal * self.patch * self.patch)


def pil_to_tensor(img, device):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def tensor_to_pil(x):
    """Quantize to 8-bit like a real saved image."""
    arr = (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def jpeg_reencode(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pgd_image_space(model, inputs, builder, x0, yes_ids, no_ids, eps, alpha, steps):
    """L-inf PGD on raw pixels in [0,1]. Maximizes logit(No) - logit(Yes) of the first token."""
    delta = torch.zeros_like(x0)
    gap = None
    for _ in range(steps):
        delta.requires_grad_(True)
        pv = builder((x0 + delta).clamp(0, 1))
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=pv,
            image_grid_thw=inputs.image_grid_thw,
        )
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
            delta = (x0 + delta).clamp(0, 1) - x0  # stay a valid image
        delta = delta.detach()
        del outputs, logit_yes, logit_no, loss, grad, pv
        torch.cuda.empty_cache()
    return (x0 + delta).clamp(0, 1).detach(), gap


def predict_file(model, processor, image_path, question, max_new_tokens):
    inputs = build_inputs(processor, image_path, question, model.device)
    pred = parse_answer(generate_with_pixel_values(model, processor, inputs, inputs.pixel_values, max_new_tokens))
    return pred, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen2b")
    parser.add_argument("--num_samples", type=int, default=-1, help="-1 = all gt=Yes samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8.0, help="L-inf budget in /255 units")
    parser.add_argument("--alpha", type=float, default=2.0, help="step size in /255 units")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--num_noise_samples", type=int, default=5)
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--token_mode", type=str, default="nospace")
    parser.add_argument("--config_name", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset_root)
    data_file = resolve_data_file(args.dataset_root, args.data_file)
    model_name = resolve_model_name(args.model_name)
    split = os.path.splitext(os.path.basename(data_file))[0]

    tag = (f"imgattack_{args.dataset_root}_{split}_{model_name.split('/')[-1]}_n{args.num_samples}_seed{args.seed}_"
           f"eps{args.eps:g}_a{args.alpha:g}_s{args.num_steps}_noise{args.noise_std}_ns{args.num_noise_samples}_"
           f"jpeg{args.jpeg_quality}" + (f"_{args.config_name}" if args.config_name else ""))
    output_path = os.path.join(OUTPUT_DIR, tag + ".json")
    adv_dir = os.path.join(OUTPUT_DIR, "adv_images", tag)
    os.makedirs(adv_dir, exist_ok=True)

    with open(f"{dataset_root}/{data_file}", encoding="utf-8") as f:
        data = [d for d in json.load(f) if d["answer"] == "Yes"]
    rng = random.Random(args.seed)
    rng.shuffle(data)
    subset = data if args.num_samples == -1 else data[:args.num_samples]
    print(f"Loaded {len(subset)} gt=Yes samples from {data_file}", flush=True)
    print(f"Attack: L-inf eps={args.eps:g}/255, alpha={args.alpha:g}/255, steps={args.num_steps}", flush=True)
    print(f"Output: {output_path}\nAdversarial images: {adv_dir}", flush=True)

    model, processor = load_model(model_name, args.min_pixels, args.max_pixels)
    yes_ids, no_ids = get_yes_no_token_ids(processor, args.token_mode)
    builder = PixelValuesBuilder(processor, model.device)
    eps, alpha = args.eps / 255.0, args.alpha / 255.0

    results = []
    checked_builder = False
    t0 = time.time()
    for i, item in enumerate(subset):
        s0 = time.time()
        image_path = f"{dataset_root}/{item['image']}"
        question = item["question"]
        caption = extract_caption(question)
        name = os.path.splitext(os.path.basename(item["image"]))[0]

        # 1. Clean prediction on the original file (same as the other experiments)
        clean_pred, inputs = predict_file(model, processor, image_path, question, args.max_new_tokens)
        if clean_pred != "Yes":
            print(f"  [{i+1}/{len(subset)}] skipped (clean_pred={clean_pred})", flush=True)
            del inputs
            continue

        # 2. Resize the original to the exact resolution the model uses, so re-loading needs no resize
        _, gh, gw = inputs.image_grid_thw[0].tolist()
        H, W = gh * builder.patch, gw * builder.patch
        x0_pil = Image.open(image_path).convert("RGB").resize((W, H), Image.BICUBIC)
        x0_path = os.path.join(adv_dir, f"{name}_clean.png")
        x0_pil.save(x0_path)
        x0 = pil_to_tensor(x0_pil, model.device)
        x0_pred, x0_inputs = predict_file(model, processor, x0_path, question, args.max_new_tokens)

        if not checked_builder:
            # Sanity check: our differentiable preprocessing must match the real processor
            diff = (builder(x0) - x0_inputs.pixel_values.float()).abs().max().item()
            same_grid = torch.equal(x0_inputs.image_grid_thw.cpu(), inputs.image_grid_thw.cpu())
            print(f"  [check] preprocessing max abs diff = {diff:.2e}, same grid = {same_grid}", flush=True)
            if diff > 1e-2 or not same_grid:
                raise RuntimeError("Differentiable preprocessing does not match the processor; stop and report this.")
            checked_builder = True

        # 3. PGD in pixel space, then quantize and save as a real PNG
        x_adv, final_gap = pgd_image_space(model, x0_inputs, builder, x0, yes_ids, no_ids,
                                           eps, alpha, args.num_steps)
        adv_pil = tensor_to_pil(x_adv)
        adv_path = os.path.join(adv_dir, f"{name}_adv.png")
        adv_pil.save(adv_path)
        adv_q = pil_to_tensor(adv_pil, model.device)
        linf = (adv_q - x0).abs().max().item() * 255
        mse = ((adv_q - x0) ** 2).mean().item()
        psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")

        attacked_pred, adv_inputs = predict_file(model, processor, adv_path, question, args.max_new_tokens)

        # JPEG re-encoding of the adversarial image (platform re-encode / simple baseline defense)
        jpeg_path = os.path.join(adv_dir, f"{name}_adv_jpeg{args.jpeg_quality}.jpg")
        jpeg_reencode(adv_pil, args.jpeg_quality).save(jpeg_path, quality=100)
        jpeg_pred, _ = predict_file(model, processor, jpeg_path, question, args.max_new_tokens)

        # 4. Defenses on the re-loaded adversarial image (majority vote only)
        seed_offset = args.seed * 1000 + i
        pv = adv_inputs.pixel_values
        text_pred, text_votes = paraphrase_defense(model, processor, adv_path, caption, pv, args.max_new_tokens)
        any_yes_pred = "Yes" if "Yes" in text_votes else "No"
        pixel_pred, _ = randomized_smoothing_defense(
            model, processor, adv_inputs, pv, num_samples=args.num_noise_samples, noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)
        combo_pred, _ = combined_defense(
            model, processor, adv_path, caption, adv_inputs, pv,
            num_noise_samples=max(1, args.num_noise_samples // 2), noise_std=args.noise_std,
            max_new_tokens=args.max_new_tokens, seed_offset=seed_offset)

        attack_success = attacked_pred != "Yes"
        preds = {"jpeg": jpeg_pred, "text": text_pred, "text_anyyes": any_yes_pred,
                 "pixel": pixel_pred, "combo": combo_pred}
        results.append({
            "image": item["image"], "adv_image": adv_path,
            "clean_pred": clean_pred, "clean_resized_pred": x0_pred,
            "attacked_pred": attacked_pred, "attack_success": attack_success,
            "final_gap_no_minus_yes": final_gap, "linf_255": linf, "psnr": psnr,
            "defense_preds": preds, "text_votes": text_votes,
            "time_sec": round(time.time() - s0, 2),
        })
        print(f"  [{i+1}/{len(subset)}] clean=Yes resized={x0_pred:3s} -> adv={attacked_pred:7s} "
              f"(Linf={linf:.0f}/255 PSNR={psnr:.1f}) | jpeg={jpeg_pred:7s} text={text_pred:7s} "
              f"anyYes={any_yes_pred:3s} pixel={pixel_pred:7s} combo={combo_pred:7s} "
              f"({time.time() - s0:.1f}s)", flush=True)

        del inputs, x0_inputs, adv_inputs, x0, x_adv, adv_q, pv
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    n_eval = len(results)
    attacked = [r for r in results if r["attack_success"]]
    survived = [r for r in results if r["attack_success"] and r["defense_preds"]["jpeg"] != "Yes"]

    def recovery(key):
        if not attacked:
            return None
        return sum(r["defense_preds"][key] == "Yes" for r in attacked) / len(attacked)

    # Defenses can also break samples the attack did not flip
    def harm_on_unflipped(key):
        unflipped = [r for r in results if not r["attack_success"]]
        if not unflipped:
            return None
        return sum(r["defense_preds"][key] != "Yes" for r in unflipped) / len(unflipped)

    summary = {
        "n_clean_correct": n_eval,
        "n_resized_still_yes": sum(r["clean_resized_pred"] == "Yes" for r in results),
        "n_attack_success": len(attacked),
        "attack_success_rate": len(attacked) / n_eval if n_eval else None,
        "attack_success_after_jpeg_rate": len(survived) / n_eval if n_eval else None,
        "recovery": {k: recovery(k) for k in ["jpeg", "text", "text_anyyes", "pixel", "combo"]},
        "harm_on_unflipped": {k: harm_on_unflipped(k) for k in ["jpeg", "text", "text_anyyes", "pixel", "combo"]},
        "mean_psnr": float(np.mean([r["psnr"] for r in results])) if results else None,
    }

    print("\n===== Image-space attack summary =====")
    print(f"Clean-correct gt=Yes samples attacked: {n_eval} "
          f"(still Yes after resize/save: {summary['n_resized_still_yes']})")
    if n_eval:
        print(f"Attack success (saved PNG):          {len(attacked)}/{n_eval} ({summary['attack_success_rate']:.1%})")
        print(f"Attack success after JPEG q={args.jpeg_quality}:     {len(survived)}/{n_eval} "
              f"({summary['attack_success_after_jpeg_rate']:.1%})")
        print(f"Mean PSNR: {summary['mean_psnr']:.1f} dB")
    print(f"{'defense':12s} {'recovery':>10s} {'harm(unflipped)':>16s}")
    for k in summary["recovery"]:
        rec, harm = summary["recovery"][k], summary["harm_on_unflipped"][k]
        print(f"{k:12s} {('-' if rec is None else f'{rec:.1%}'):>10s} {('-' if harm is None else f'{harm:.1%}'):>16s}")
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "summary": summary, "total_time_sec": time.time() - t0,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
