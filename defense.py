# defense.py
"""
三種獨立、可組合的防禦,各自有多數決版跟加權投票版:
1. paraphrase_defense / paraphrase_defense_weighted: 文字層
2. randomized_smoothing_defense / randomized_smoothing_defense_weighted: 像素層
3. combined_defense / combined_defense_weighted: 交叉組合
"""
import torch
from common import build_inputs, generate_with_pixel_values, parse_answer, yes_no_logits


# ---------- 共用投票邏輯 ----------

def majority_vote(predictions):
    valid = [p for p in predictions if p in ("Yes", "No")]
    if not valid:
        return "Unclear"
    n_yes, n_no = valid.count("Yes"), valid.count("No")
    if n_yes == n_no:
        # Neutral tie rule (old version silently returned "No" on ties)
        return "Unclear"
    return "Yes" if n_yes > n_no else "No"


def weighted_vote(answers_with_confidence):
    yes_weight = sum(c for a, c in answers_with_confidence if a == "Yes")
    no_weight = sum(c for a, c in answers_with_confidence if a == "No")
    if yes_weight == no_weight:
        # Neutral tie rule (old version returned "Yes" on ties)
        return "Unclear"
    return "Yes" if yes_weight > no_weight else "No"


def vote_score(answers_with_confidence):
    """Signed total margin: >0 leans Yes, <0 leans No. Saved for later calibration analysis."""
    return sum(c if a == "Yes" else -c for a, c in answers_with_confidence)


@torch.no_grad()
def get_answer_with_confidence(model, processor, inputs, pixel_values, yes_id, no_id, max_new_tokens=10):
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=inputs.image_grid_thw,
    )
    last_logits = outputs.logits[0, -1, :]
    # yes_id / no_id are lists of token ids (see common.get_yes_no_token_ids)
    logit_yes, logit_no = yes_no_logits(last_logits, yes_id, no_id)
    logit_yes, logit_no = logit_yes.item(), logit_no.item()
    confidence = abs(logit_yes - logit_no)
    answer = "Yes" if logit_yes > logit_no else "No"
    return answer, confidence


# ---------- 文字層防禦 ----------

def make_paraphrase_questions(base_caption):
    templates = [
        'Given the meme image with the caption "{caption}", does this meme contain '
        'hateful content targeting a specific group (e.g. race, religion, gender, '
        'disability)? Answer only Yes or No.',

        'Looking at this meme, which has the text "{caption}", would you say it '
        'expresses hatred or hostility toward any particular group of people? '
        'Answer only Yes or No.',

        'This meme is captioned "{caption}". Is this meme discriminatory or '
        'offensive toward a specific demographic group? Answer only Yes or No.',
    ]
    return [t.format(caption=base_caption) for t in templates]


def paraphrase_defense(model, processor, image_path, original_caption, pixel_values, max_new_tokens=10):
    questions = make_paraphrase_questions(original_caption)
    predictions = []
    for q in questions:
        inputs = build_inputs(processor, image_path, q, model.device)
        output_text = generate_with_pixel_values(model, processor, inputs, pixel_values, max_new_tokens)
        predictions.append(parse_answer(output_text))
    return majority_vote(predictions), predictions


def paraphrase_defense_weighted(model, processor, image_path, original_caption, pixel_values,
                                  yes_id, no_id, max_new_tokens=10):
    questions = make_paraphrase_questions(original_caption)
    results = []
    for q in questions:
        q_inputs = build_inputs(processor, image_path, q, model.device)
        answer, confidence = get_answer_with_confidence(
            model, processor, q_inputs, pixel_values, yes_id, no_id, max_new_tokens
        )
        results.append((answer, confidence))
    return weighted_vote(results), results


# ---------- 像素層防禦 ----------

def randomized_smoothing_defense(model, processor, inputs, pixel_values, num_samples=5,
                                   noise_std=0.02, max_new_tokens=10, seed_offset=0, verbose=False):
    predictions = []
    for i in range(num_samples):
        torch.manual_seed(seed_offset + i)
        noise = torch.randn_like(pixel_values) * noise_std
        noisy_pixel_values = pixel_values + noise
        output_text = generate_with_pixel_values(model, processor, inputs, noisy_pixel_values, max_new_tokens)
        pred = parse_answer(output_text)
        predictions.append(pred)
        if verbose:
            print(f"        sample {i}: pred={pred}  raw='{output_text.strip()}'", flush=True)
    return majority_vote(predictions), predictions


def randomized_smoothing_defense_weighted(model, processor, inputs, pixel_values, yes_id, no_id,
                                            num_samples=5, noise_std=0.02, max_new_tokens=10, seed_offset=0):
    """
    randomized_smoothing_defense 的加權版本:新增,補齊 pixel 單獨的加權投票。
    """
    results = []
    for i in range(num_samples):
        torch.manual_seed(seed_offset + i)
        noise = torch.randn_like(pixel_values) * noise_std
        noisy_pixel_values = pixel_values + noise
        answer, confidence = get_answer_with_confidence(
            model, processor, inputs, noisy_pixel_values, yes_id, no_id, max_new_tokens
        )
        results.append((answer, confidence))
    return weighted_vote(results), results


# ---------- 組合防禦 ----------

def combined_defense(model, processor, image_path, original_caption, inputs, pixel_values,
                      num_noise_samples=3, noise_std=0.02, max_new_tokens=10, seed_offset=0):
    questions = make_paraphrase_questions(original_caption)
    predictions = []
    for q in questions:
        q_inputs = build_inputs(processor, image_path, q, model.device)
        for i in range(num_noise_samples):
            torch.manual_seed(seed_offset + i)
            noise = torch.randn_like(pixel_values) * noise_std
            noisy_pixel_values = pixel_values + noise
            output_text = generate_with_pixel_values(model, processor, q_inputs, noisy_pixel_values, max_new_tokens)
            predictions.append(parse_answer(output_text))
    return majority_vote(predictions), predictions


def combined_defense_weighted(model, processor, image_path, original_caption, inputs, pixel_values,
                                yes_id, no_id, num_noise_samples=3, noise_std=0.02,
                                max_new_tokens=10, seed_offset=0):
    questions = make_paraphrase_questions(original_caption)
    results = []
    for q in questions:
        q_inputs = build_inputs(processor, image_path, q, model.device)
        for i in range(num_noise_samples):
            torch.manual_seed(seed_offset + i)
            noise = torch.randn_like(pixel_values) * noise_std
            noisy_pixel_values = pixel_values + noise
            answer, confidence = get_answer_with_confidence(
                model, processor, q_inputs, noisy_pixel_values, yes_id, no_id, max_new_tokens
            )
            results.append((answer, confidence))
    return weighted_vote(results), results