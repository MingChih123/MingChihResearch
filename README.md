# 不需訓練的仇恨迷因審核防禦:Dual-Channel Reading — 操作手冊

用凍結的視覺語言模型(Qwen2-VL)或專門的審核模型(LlavaGuard)判斷迷因是不是仇恨 / 有害內容,
在**完全不訓練判斷模型**的前提下,抵抗「改迷因文字、在圖上加字、改圖片像素」的攻擊。

**主方法 Dual-Channel Reading:**
- 第一路:原圖 + 迷因文字,先把「替迷因開脫」的句子拿掉,再判斷。
- 第二路:縮小一半再放大的圖,完全不給迷因文字,再判斷。
- 任一路判仇恨 → 仇恨。可選的 OCR 遮罩會先把寫在圖上的開脫句塗掉。

> 這份手冊照順序做就能重現論文的數字。研究現況和結果在 `CLAUDE.md`,論文架構在 `THESIS_OUTLINE.md`。
> 舊版手冊(含已放棄的方法)在 git 歷史裡。

---

## 0. 需要什麼

| 項目 | 需求 |
|---|---|
| 作業系統 | Windows 10/11(指令以 CMD 為準) |
| GPU | NVIDIA,至少 8GB 顯存(原作者用 RTX 3070 8GB) |
| 硬碟 | 約 60GB(Qwen 2B 約 5GB、7B 約 15GB、LlavaGuard 7B 約 15GB,加上資料集和實驗圖片) |
| 軟體 | Miniconda / Anaconda、Git |

---

## 1. 安裝環境

```bash
conda create -n pda python=3.10 -y
conda activate pda
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install bitsandbytes
pip install easyocr --no-deps && pip install opencv-python-headless scikit-image python-bidi PyYAML Shapely pyclipper ninja scipy numpy==1.24.3
```

- `bitsandbytes`:LlavaGuard 7B 用 4-bit 載入才放得進 8GB。
- `easyocr` 用 `--no-deps` 安裝,避免 pip 換掉 torch。只有 OCR 遮罩(`--ocr_mask`)需要它。
- 所有模型第一次執行時會自動從 Hugging Face 下載,不用申請。
- 如果 transformers 因為 tensorflow 出錯,先執行 `set "USE_TF=0"`。

確認環境:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. 準備資料集

資料集不放在 GitHub(太大,而且有授權限制,只能在實驗室內部分享)。放好後的結構:

```
dataset\
├─ FB\           img\、dev_vqa.json(500 筆:250 / 250)
├─ HarMeme\      images\、annotations\(val.jsonl、test.jsonl、val_vqa.json)
├─ PrideMM\      Images\、PrideMM.csv
├─ HarmP\        Harm-P\data\datasets\memes\defaults\(annotations\test.jsonl、images\)
└─ MAMI\         Images\test_images\、test.tsv
```

轉成問答格式(每個只要做一次):
```bash
python fix_harmeme_paths.py & python make_harmeme_test_vqa.py & python make_extra_vqa.py
```
- `make_harmeme_test_vqa.py` 產生 HarMeme 的 `valtest_vqa.json`(val + test 合併);看到 `[STOP]` 代表規則對不上,不會產生檔案。
- `make_extra_vqa.py` 產生 PrideMM / Harm-P / MAMI 的 `test_vqa.json`(只用官方 test 切分),會印出每個資料集的筆數和缺圖數。
- `HarmP\Harm-C` 就是 HarMeme,不用另外跑。

問答格式:`{"image": "...", "question": "Given the meme image with the caption \"...\", ...? Answer only Yes or No.", "answer": "Yes"}`。問法依資料集自動選(`defense.py`)。

---

## 3. 程式

| 檔案 | 用途 |
|---|---|
| **`run_robust_suite.py`** | 所有實驗:攻擊 × 防禦、模型選擇、`--clean_only`(不攻擊,量誤判) |
| **`make_final_tables.py`** | 把 `output\*_final.json` 整理成 `output\_FINAL_TABLES.md`(含 95% 信賴區間、McNemar) |
| `make_extra_vqa.py`、`make_harmeme_test_vqa.py`、`fix_harmeme_paths.py` | 資料準備 |
| `defense.py`、`config.py`、`common.py`、`run_image_attack.py` | 被 `run_robust_suite.py` 用到的模組,不要刪 |

其他 `.py` 是舊實驗,現在不用。

`run_robust_suite.py` 常用參數:

| 參數 | 意思 |
|---|---|
| `--dataset_root FB / HarMeme / PrideMM / HarmP / MAMI` | 資料集(HarMeme 要加 `--data_file annotations/valtest_vqa.json`) |
| `--model_name` | `qwen2b`(預設)、`qwen7b`、`llavaguard05`、`llavaguard7b` |
| `--clean_only` | 不攻擊,只量乾淨圖的準確率和誤判 |
| `--attacks` | 攻擊清單(見下) |
| `--defenses` | `none,jpeg,tq,strip,nocap,f3`;`strip` + `nocap` 都開才會算出 `dual`(主方法) |
| `--channel_resize ch2` | **主設定**:只有第二路縮放(第一路讀原圖)。不加的話是舊設計(兩路都縮放) |
| `--ocr_mask` | 兩路讀圖前先用 OCR 找出圖上的開脫句並塗掉 |
| `--num_samples -1` | 全部(攻擊模式是「全部仇恨迷因」,其中模型原本認得出的才會被攻擊) |
| `--config_name final` | 檔名加 `_final`,`make_final_tables.py` 只讀這些 |

攻擊:`caption_persuade`(說服句)、`caption_inject`(叫審核者回答 No)、`caption_persuade_blend`(說服句黏進原文)、`caption_evade`(挑檢查器抓不到的開脫句)、`image_text_pad`(圖上加字)、`image_text_persuade`(圖上加字 + 說服句)、`pgd` / `pgd_jpeg` / `pgd_persuade`(像素攻擊,只能用在 Qwen)。

輸出檔在 `output\`,檔名會帶資料集、模型、攻擊和設定,不會互相覆蓋。

---

## 4. 實驗指令

每組都是「乾淨圖 + 攻擊」兩個指令用 `&` 串起來。很久的實驗建議放過夜。

### 4.1 Qwen2-VL-2B(主模型,含像素攻擊)

FB:
```bash
python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --channel_resize ch2 --clean_only --tq_transforms resize50 --defenses none,jpeg,tq,strip,nocap,f3 --config_name final & python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --channel_resize ch2 --attacks caption_persuade,caption_persuade_blend,caption_inject,caption_evade,image_text_pad,image_text_persuade,pgd,pgd_jpeg,pgd_persuade --tq_transforms resize50 --defenses none,jpeg,tq,strip,nocap,f3 --config_name final
```
- 其他資料集:把 `--dataset_root FB` 換成 `PrideMM`、`HarmP`、`MAMI`;HarMeme 是 `--dataset_root HarMeme --data_file annotations/valtest_vqa.json`。
- MAMI 有 1000 張,乾淨圖可以先用 `--num_samples 500`。
- 很花時間的是 PGD 類攻擊,趕時間可以先拿掉 `pgd,pgd_jpeg,pgd_persuade`。

### 4.2 Qwen2-VL-7B(只跑文字和圖上加字)

```bash
python -u run_robust_suite.py --dataset_root FB --num_samples 200 --seed 0 --model_name qwen7b --min_pixels 65536 --max_pixels 200704 --channel_resize ch2 --clean_only --tq_transforms resize50 --defenses none,tq,strip,nocap --config_name final & python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --model_name qwen7b --min_pixels 65536 --max_pixels 200704 --channel_resize ch2 --attacks caption_persuade,caption_inject,caption_evade,image_text_persuade --tq_transforms resize50 --defenses none,tq,strip,nocap --config_name final
```
8GB 放不下 7B,會借用系統記憶體,很慢。

### 4.3 審核模型 LlavaGuard v1.2(0.5B / 7B)

用 LlavaGuard 官方的安全政策 prompt,回答 Safe / Unsafe + 類別(任何類別的 Unsafe 都算有害)。迷因文字放在政策前面當 OCR 文字;第二路不放。開脫句檢查也用 LlavaGuard 自己(`--guard_detector self`)。

0.5B:
```bash
python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --model_name llavaguard05 --guard_detector self --channel_resize ch2 --clean_only --tq_transforms resize50 --defenses none,tq,strip,nocap --config_name final & python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --model_name llavaguard05 --guard_detector self --channel_resize ch2 --attacks caption_persuade,caption_inject,caption_evade,caption_persuade_blend,image_text_pad,image_text_persuade --tq_transforms resize50 --defenses none,tq,strip,nocap --config_name final
```
- 7B:把 `--model_name llavaguard05` 換成 `--model_name llavaguard7b --guard_4bit`。
- 答案存在 `output\guard_cache_*.jsonl`,同樣的問題重跑會直接讀快取。
- 只能跑不用梯度的攻擊(沒有 PGD)。

### 4.4 OCR 遮罩(`--ocr_mask`)

在任何指令後面加 `--ocr_mask` 就會啟用:用 EasyOCR 找出圖上每一行字,用和迷因文字同一個開脫句檢查器判斷,是開脫句的行塗成背景色,兩路再讀遮好的圖。迷因本身的仇恨文字不是開脫句,會留下來。

先測(LlavaGuard 0.5B):
```bash
python -u run_robust_suite.py --dataset_root FB --num_samples 100 --seed 0 --model_name llavaguard05 --guard_detector self --channel_resize ch2 --ocr_mask --clean_only --tq_transforms resize50 --defenses none,strip,nocap --config_name ocrtest & python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --model_name llavaguard05 --guard_detector self --channel_resize ch2 --ocr_mask --attacks image_text_pad,image_text_persuade,caption_persuade --tq_transforms resize50 --defenses none,strip,nocap --config_name ocrtest
```
- Summary 最後印 `ocr_mask [...]: masked a/b text lines`:乾淨圖應該接近 0;圖上加字的攻擊應該接近每張 1 行。
- GPU 不夠時加 `--ocr_cpu`。遮過的圖存在 `output\robust_images\...\*_ocrmask.png`,可以打開檢查。

### 4.5 整理表格

```bash
python make_final_tables.py
```
讀 `output\*_final.json`,依「資料集 × 模型」分節,輸出 `output\_FINAL_TABLES.md`。

---

## 5. 常見問題

- **CMD 串指令**:用 `&`(前一個失敗,下一個照跑)或 `&&`(前一個失敗就停)。不要用 `^` 換行。
- **`CUDA out of memory`**:關掉其他用 GPU 的程式;7B 一定要加 `--guard_4bit`;OCR 可以加 `--ocr_cpu`。
- **`Setting pad_token_id ...` 一直跳**:舊版程式的警告,`git pull` 更新就沒了;不影響結果。
- **跑到一半關機**:Qwen 要重跑;LlavaGuard 的答案有快取,重跑很快。
- **舊結果**:FB / HarMeme / PrideMM / Qwen 7B 早期的結果是舊設計(兩路都縮放,沒有 `_rs-ch2`);論文最終表用 `--channel_resize ch2` 的結果。
