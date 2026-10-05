# test_textfooler.py
import json
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

from config import resolve_dataset_root, resolve_data_file, resolve_model_name
from common import load_model, get_yes_no_token_ids
from textattack_wrapper import VLMTextAttackWrapper


def build_question_with_caption(caption):
    return (f'Given the meme image with the caption "{caption}", does this meme contain '
            f'hateful content targeting a specific group (e.g. race, religion, gender, '
            f'disability)? Answer only Yes or No.')


def extract_caption(question):
    return question.split('caption "')[1].split('"')[0]


def main():
    dataset_root = resolve_dataset_root("FB")
    data_file = resolve_data_file("FB", None)
    model_name = resolve_model_name("qwen2b")

    with open(f"{dataset_root}/{data_file}", encoding="utf-8") as f:
        data = json.load(f)

    sample = next(d for d in data if d["answer"] == "Yes")
    image_path = f"{dataset_root}/{sample['image']}"
    caption = extract_caption(sample["question"])

    print(f"測試圖片: {sample['image']}")
    print(f"原始 caption: {caption}")

    model, processor = load_model(model_name)
    yes_id, no_id = get_yes_no_token_ids(processor)

    wrapper = VLMTextAttackWrapper(model, processor, image_path, yes_id, no_id, build_question_with_caption)

    dataset = Dataset([(caption, 1)])

    attack = TextFoolerJin2019.build(wrapper)

    # 用 Attacker 執行攻擊,query_budget 限制每句話最多嘗試幾次,避免第一次測試跑太久
    attack_args = AttackArgs(num_examples=1, query_budget=200, disable_stdout=False)
    attacker = Attacker(attack, dataset, attack_args)

    results = attacker.attack_dataset()

    for result in results:
        print("\n===== 攻擊結果 =====")
        print(result)


if __name__ == "__main__":
    main()