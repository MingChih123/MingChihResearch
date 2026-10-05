# config.py
"""
集中管理所有資料集路徑、模型別名。
之後要加新資料集或新模型,只需要改這個檔案,不用去其他程式裡找。
"""

DATASET_ROOT_MAP = {
    "FB": "./dataset/FB",
    "HarMeme": "./dataset/HarMeme",
}

# 每個資料集預設要讀哪個 vqa json 檔(相對於 dataset_root 的路徑)
DATASET_DATA_FILE_MAP = {
    "FB": "dev_vqa.json",
    "HarMeme": "annotations/val_vqa.json",
}

MODEL_ALIAS_MAP = {
    "qwen2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen7b": "Qwen/Qwen2-VL-7B-Instruct",
}


def resolve_dataset_root(name_or_path):
    return DATASET_ROOT_MAP.get(name_or_path, name_or_path)


def resolve_data_file(dataset_key, explicit_data_file=None):
    """使用者有指定 --data_file 就用指定的,否則依資料集自動查表。"""
    if explicit_data_file:
        return explicit_data_file
    return DATASET_DATA_FILE_MAP.get(dataset_key, "dev_vqa.json")


def resolve_model_name(name_or_path):
    return MODEL_ALIAS_MAP.get(name_or_path, name_or_path)