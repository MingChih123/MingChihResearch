# textattack_wrapper.py
"""
把我們的 VLM(Qwen2-VL)包裝成 TextAttack 看得懂的介面。
TextAttack 要求:輸入一批候選文字,輸出每一批對應的 [No的機率, Yes的機率] 矩陣。
"""
import torch
import torch.nn.functional as F
from textattack.models.wrappers import ModelWrapper
from common import build_inputs, yes_no_logits


class VLMTextAttackWrapper(ModelWrapper):
    def __init__(self, model, processor, image_path, yes_id, no_id, question_template):
        """
        image_path: 這次攻擊固定針對的那張圖片(乾淨圖,不動)
        question_template: 一個函式,吃 caption 字串,回傳完整的 question 字串
        """
        self.model = model
        self.processor = processor
        self.image_path = image_path
        self.yes_id = yes_id
        self.no_id = no_id
        self.question_template = question_template

    def __call__(self, text_input_list):
        """
        text_input_list: TextAttack 傳進來的一批候選 caption(字串列表)
        回傳: shape [batch, 2] 的機率矩陣,欄位順序是 [No的機率, Yes的機率]
        (TextAttack 慣例:index 0 通常對應「非目標類別」,index 1 對應「目標類別」,
         我們把 Yes 當作 label=1,No 當作 label=0,等一下設定攻擊目標時要對應好)
        """
        probs_batch = []

        with torch.no_grad():
            for caption in text_input_list:
                question = self.question_template(caption)
                inputs = build_inputs(self.processor, self.image_path, question, self.model.device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,
                )
                last_logits = outputs.logits[0, -1, :]

                # 只取 Yes/No 這兩個 token 的 logit,做 softmax 得到相對機率
                # yes_id / no_id are lists of token ids (see common.get_yes_no_token_ids)
                logit_yes, logit_no = yes_no_logits(last_logits, self.yes_id, self.no_id)
                two_logits = torch.tensor([logit_no.item(), logit_yes.item()])
                probs = F.softmax(two_logits, dim=0)
                probs_batch.append(probs.tolist())

        return torch.tensor(probs_batch)