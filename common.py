# common.py
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(model_name="Qwen/Qwen2-VL-2B-Instruct",
                min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28):
    print(f"Loading model: {model_name} ...", flush=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=min_pixels, max_pixels=max_pixels
    )
    print(f"Model loaded. Device: {model.device}", flush=True)
    return model, processor


# Which token spellings count as "Yes" / "No" for the first generated token.
# The model's first token is "Yes"/"No" WITHOUT a leading space (verified by
# diagnose_weighted_bias.py: 100/100 first tokens were "Yes" or "No").
# "space" is the old (buggy) setting, kept only to reproduce old results.
TOKEN_MODES = {
    "nospace": (["Yes"], ["No"]),
    "space": ([" Yes"], [" No"]),
    "lse": (["Yes", " Yes"], ["No", " No"]),
}


def get_yes_no_token_ids(processor, mode="nospace"):
    """Return (yes_ids, no_ids) as lists of token ids."""
    if mode not in TOKEN_MODES:
        raise ValueError(f"unknown token mode: {mode} (choose from {list(TOKEN_MODES)})")
    yes_words, no_words = TOKEN_MODES[mode]
    tok = processor.tokenizer
    yes_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in yes_words]
    no_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in no_words]
    return yes_ids, no_ids


def yes_no_logits(last_logits, yes_ids, no_ids):
    """Combine logits of all Yes / No spellings with logsumexp (a single id -> its own logit).
    Keeps the autograd graph, so the PGD attack can use it too."""
    logit_yes = torch.logsumexp(last_logits[yes_ids].float(), dim=0)
    logit_no = torch.logsumexp(last_logits[no_ids].float(), dim=0)
    return logit_yes, logit_no


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


def build_inputs(processor, image_path, question, device):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    return inputs


@torch.no_grad()
def generate_with_pixel_values(model, processor, inputs, pixel_values, max_new_tokens=10):
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text