# attack.py
"""
PGD 對抗攻擊:針對 VLM 即將生成的第一個 token(Yes/No)做梯度攻擊。
攻擊作用在模型實際吃進去的正規化張量 pixel_values 上,
這個做法在精神上跟 RAMPART 論文 Eq. 3.10-3.11(在正規化空間內攻擊)一致。
"""
# attack.py
"""
攻擊模組:包含圖片攻擊(PGD)跟文字攻擊(leetspeak / char_swap)。
"""
import torch
import random


# ---------- 圖片攻擊 ----------

def pgd_attack_first_token(model, inputs, yes_id, no_id, epsilon, alpha, num_steps, verbose=False):
    pixel_values_orig = inputs.pixel_values.clone().detach()
    pixel_values_adv = pixel_values_orig.clone().detach()

    for step in range(num_steps):
        pixel_values_adv.requires_grad_(True)
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=pixel_values_adv,
            image_grid_thw=inputs.image_grid_thw,
        )
        last_logits = outputs.logits[0, -1, :]
        logit_yes = last_logits[yes_id]
        logit_no = last_logits[no_id]
        loss = logit_no - logit_yes
        loss.backward()

        grad = pixel_values_adv.grad
        if verbose:
            gnorm = grad.abs().mean().item() if grad is not None else None
            print(f"      step {step}: gap(no-yes)={loss.item():.3f}  grad_mean_abs={gnorm}", flush=True)
        if grad is None:
            break

        with torch.no_grad():
            pixel_values_adv = pixel_values_adv + alpha * grad.sign()
            delta = torch.clamp(pixel_values_adv - pixel_values_orig, -epsilon, epsilon)
            pixel_values_adv = pixel_values_orig + delta
        pixel_values_adv = pixel_values_adv.detach()

        del outputs, last_logits, logit_yes, logit_no, loss, grad
        torch.cuda.empty_cache()

    return pixel_values_adv


# ---------- 文字攻擊 ----------

LEET_MAP = {
    "a": "@", "A": "@",
    "i": "!", "I": "!",
    "o": "0", "O": "0",
    "e": "3", "E": "3",
    "s": "$", "S": "$",
}


def leetspeak_attack(caption, corruption_rate=0.5, seed=0):
    rng = random.Random(seed)
    result = []
    for ch in caption:
        if ch in LEET_MAP and rng.random() < corruption_rate:
            result.append(LEET_MAP[ch])
        else:
            result.append(ch)
    return "".join(result)


def char_swap_attack(caption, num_swaps=3, seed=0):
    rng = random.Random(seed)
    chars = list(caption)
    alpha_positions = [i for i, c in enumerate(chars) if c.isalpha() and i + 1 < len(chars) and chars[i + 1].isalpha()]
    if not alpha_positions:
        return caption
    swap_positions = rng.sample(alpha_positions, min(num_swaps, len(alpha_positions)))
    for pos in swap_positions:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def apply_text_attack(caption, attack_type="none", seed=0, corruption_rate=0.5, num_swaps=3):
    """
    attack_type: "none" / "leetspeak" / "char_swap"
    corruption_rate: leetspeak 專用,替換機率(0.0~1.0)
    num_swaps: char_swap 專用,交換幾組相鄰字母
    """
    if attack_type == "none":
        return caption
    elif attack_type == "leetspeak":
        return leetspeak_attack(caption, corruption_rate=corruption_rate, seed=seed)
    elif attack_type == "char_swap":
        return char_swap_attack(caption, num_swaps=num_swaps, seed=seed)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")