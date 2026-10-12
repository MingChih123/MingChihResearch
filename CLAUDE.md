# 專案交接文件(給 Claude Code)

> 使用方式:把這份檔案放在專案根目錄(`MChih_Exp\2\`),建議改名成 `CLAUDE.md`,Claude Code 開啟專案時會自動讀取。
> 使用者是碩士生,語言用繁體中文溝通,程式碼註解與變數用英文即可。

---

## 0. 先看這裡:加權投票偏誤已查明(2026-10)

**結論:舊的加權投票數字(恢復率 66%~100%、誤判修正 46%~54%)全部作廢,是量錯 token 造成的假象。**

`diagnose_weighted_bias.py`(FB dev,n=100 混合,seed 0)的結果:

- 模型生成的第一個 token 100/100 都是**不帶空格**的 `"Yes"`(9454)/ `"No"`(2753)。舊程式比的是 `" Yes"`(7414)/ `" No"`(2308),模型根本不會輸出這兩個。
- 乾淨圖上,用 `" Yes"/" No"` 比 logit:和生成答案只有 68% 一致,Yes 比例 46%(生成只有 14%),32 筆不一致**全部**是「生成 No、logit Yes」→ 系統性偏 Yes。
- 換成 `"Yes"/"No"`:一致率 95%,Yes 比例 15%,不一致兩個方向都有(剩下的是邊界樣本 bf16 數值差異)。logsumexp 結果幾乎一樣。
- 攻擊後:被打成 No 的 7 筆 Yes 樣本,**完全不防禦**直接讀 `" Yes"/" No"` 就有 4 筆顯示 Yes(假的 57% 恢復率);讀 `"Yes"/"No"` 只剩 1 筆。
- 模型本身的判斷能力沒有變:生成答案 accuracy 67%、recall 28%(12/43),模型偏向答 No。
- PGD 攻擊與 TextFooler wrapper 也都用了錯的 token,攻擊是靠「順便」推動 `"No"` 才有效,實際強度被低估。

**已修正(新程式):**
- `common.get_yes_no_token_ids(processor, mode)` 回傳 id 的 list,`mode` = `nospace`(預設、正確)/ `lse` / `space`(舊的,只用來重現舊數字)。`common.yes_no_logits` 用 logsumexp 合併。
- `attack.py`、`defense.py`、`textattack_wrapper.py` 都改用上面兩個函式。
- `majority_vote`、`weighted_vote` 平手一律回傳 `Unclear`(舊版多數決平手判 No、加權平手判 Yes)。
- `run_experiment.py`、`run_error_correction_experiment.py` 加 `--token_mode`(預設 nospace),檔名帶 `tok<mode>`。
- 新增 `run_clean_eval.py`:乾淨混合樣本,每筆都跑全部防禦,報 accuracy/F1/混淆矩陣、兩個方向分開的修正率、傷害率,並存 logit 分數供校正分析。

**修正後尚未重跑的:** 攻擊強度/雜訊強度掃描、乾淨評估、TextFooler。第 6 節所有「加權」欄位與攻擊成功率都要以新結果為準。多數決(走 `generate`)的防禦結果本身沒受 token 影響,但攻擊變了,也要重跑。

---

## 1. 研究目標

- 任務:判斷迷因(圖+圖上文字)是否為仇恨迷因,輸出 Yes / No。
- 模型:通用視覺語言模型(Qwen2-VL),**完全凍結、不訓練、不微調**。以生成式 VQA 方式回答,不是專用分類器。
- 研究問題:對圖片做人眼看不出的對抗攻擊(PGD)會讓模型答錯;能不能只在推論階段(test-time)加防禦,把答案救回來。
- 防禦的核心:換問法、對圖片加隨機雜訊、多次詢問後投票(多數決 / 以 logit 差距當信心的加權投票)。
- 靈感來源:PDA(改寫+投票)、R-TPT(可靠度加權集成)、Self-Consistency。AOM、TTC 是同類方法,但本實驗沒有用到它們的機制(AOM 在特徵空間操作;TTC 先偵測再主動反擊)。
- 文字端:把迷因上的文字**明講寫進 prompt**,不讓模型自己 OCR,目的是讓攻擊只影響視覺語意判斷。

---

## 2. 環境

- Windows(CMD),conda 環境名稱 `pda`,Python 3.10。
- GPU:NVIDIA RTX 3070,**8GB 顯存**(影響很大,見下)。
- 主要套件:torch 2.4.1+cu121、transformers、accelerate、qwen-vl-utils、textattack 0.3.10、tensorflow 2.13、tensorflow_hub、numpy 1.24.3、nltk。
- **環境警告**:安裝 tensorflow 後 pip 報 `torch 2.4.1+cu121 requires typing-extensions>=4.8.0, but you have 4.5.0`。目前能跑,但可能是之後出奇怪錯誤的原因,若遇到莫名錯誤先檢查這個。
- 模型:`Qwen/Qwen2-VL-2B-Instruct`(主力)。`Qwen/Qwen2-VL-7B-Instruct` 只能降低圖片解析度後小樣本跑(`--min_pixels 65536 --max_pixels 200704`),一筆約 260~280 秒。
- 8GB 的限制:PGD 反向傳播時 7B 預設參數會 OOM;已凍結全部權重、`model.eval()`、不開 gradient checkpointing(開了會讓梯度變 None)。
- 專案路徑:`C:\Users\USER\OneDrive\Desktop\MChih_Exp\2\`

---

## 3. 資料集

| 資料集 | 位置 | 使用的切分 | 筆數 |
|---|---|---|---|
| Facebook Hateful Memes (FB) | `dataset/FB/`,圖在 `img/` | `dev_vqa.json` | 500(250 Yes / 250 No) |
| HarMeme | `dataset/HarMeme/`,圖在 `images/` | `annotations/val_vqa.json` | 177(61 Yes / 116 No) |

- 只用 dev / val 做評測,train 沒用到(沒有任何訓練步驟)。
- VQA 格式:`{"image": "...", "question": "...", "answer": "Yes"/"No"}`,`label=1` → `Yes`。
- FB 的 question 模板:`Given the meme image with the caption "{caption}", does this meme contain hateful content targeting a specific group (e.g. race, religion, gender, disability)? Answer only Yes or No.`
- HarMeme 的 question 措辭不同:`...does this meme contain hateful or harmful content targeting a specific group or individual? Answer only Yes or No.` ,而 `defense.py::make_paraphrase_questions` 的三個模板是 FB 風格,對 HarMeme 不完全一致(待決定要不要統一)。
- HarMeme 的 json 原本 `image` 欄位缺 `images/` 前綴,已用 `fix_harmeme_paths.py` 修好(可重複執行,不會重複加前綴)。
- HarMeme 的 caption 含換行,`extract_caption` 用 `question.split('caption "')[1].split('"')[0]` 抽取,caption 內若有雙引號會截斷。

---

## 4. 程式結構

```
config.py                       資料集路徑、預設資料檔、模型別名(qwen2b / qwen7b)
common.py                       載入模型、build_inputs、generate_with_pixel_values、parse_answer、get_yes_no_token_ids
attack.py                       pgd_attack_first_token(圖片)、leetspeak / char_swap / apply_text_attack(文字)
defense.py                      三類防禦,各有多數決與加權兩版(見下)
run_experiment.py               主實驗:抽樣 → 乾淨預測 → 攻擊 → 六種防禦 → 統計 → 存 json
clean_control_check.py          乾淨圖片套防禦,看會不會把原本答對的弄錯(尚未更新加權版)
run_error_correction_experiment.py   沒有攻擊,抓模型本來答錯的樣本,測防禦能不能修正
run_textfooler_experiment.py    TextFooler 文字攻擊(多樣本)
textattack_wrapper.py           把 VLM 包成 TextAttack 的 ModelWrapper(圖固定,換文字)
test_textfooler.py              TextFooler 單樣本最小測試
summarize_results.py            掃描 ./output/*.json 彙整成 ./output/_summary.csv
verify_weighted.py              比對多數決與加權投票逐筆差異(要手動改檔名與欄位)
diagnose_weighted_bias.py       P0 診斷:生成答案 vs 各種 token 的 logit 判斷
run_clean_eval.py               乾淨混合樣本完整評估(整體指標 + 雙向修正率 + 傷害率)
calibrate_threshold.py          train-free 門檻校正:在 train split 選門檻,套到 run_clean_eval 存好的 dev 分數
ablate_templates.py             換問法消融:每個問法單獨 vs 多數決 vs any-Yes / all-Yes
run_image_attack.py             真實像素空間 PGD(L-inf /255、存成 PNG 再讀回)+ JPEG 情境 + 防禦(多數決)
run_robust_suite.py             攻擊組合(pgd / pgd_jpeg / saltpepper / spread,另可加 pgd_eot)× 防禦組合(含新的 transform、tq),--clean_only 量正常圖副作用
summarize_all.py                把 output 裡新實驗的結果整理成 _table_*.csv
fix_harmeme_paths.py / check_harmeme.py   HarMeme 路徑修復與檢查
output/                         所有實驗結果 json
```

**defense.py 的函式:**

| 防禦 | 多數決 | 加權 |
|---|---|---|
| 文字(3 種同義問法) | `paraphrase_defense` | `paraphrase_defense_weighted` |
| 像素(疊加高斯雜訊 `noise_std`,取樣 `num_noise_samples` 次) | `randomized_smoothing_defense` | `randomized_smoothing_defense_weighted` |
| 組合(3 種問法 × `num_noise_samples//2` 次雜訊) | `combined_defense` | `combined_defense_weighted` |

共用:`majority_vote`、`weighted_vote`、`get_answer_with_confidence`、`vote_score`。信心分數 = 第一個生成 token 的 `|logit(Yes) - logit(No)|`(token 由 `--token_mode` 決定,預設不帶空格);加權 = 把 Yes 陣營與 No 陣營的信心分數各自加總比大小。

**攻擊細節:**

- PGD 作用在 processor 產生的 `pixel_values`(已正規化的 patch 張量),不是 [0,1] 像素。
- 目標:最大化第一個 token 的 `logit(No) - logit(Yes)`(2026-10 前用的是帶空格的錯誤 token)。
- 預設 `epsilon=0.1, alpha=0.04, num_steps=3`。
- 只攻擊「標準答案 Yes 且模型乾淨時也答 Yes」的樣本(`only_yes=True`)。**因此目前攻擊實驗完全沒有測到 gt=No 的方向。**
- 指標:攻擊成功率 = 攻擊成功數 / 乾淨答對數;恢復率 = 防禦後答對數 / 攻擊成功數。

---

## 5. 指令速查(Windows CMD)

單行長指令用 `&&` 串接,**不要混用 `^` 換行**(曾因此只有最後一個指令執行)。要換行就把 `&&^` 黏在行尾。

```bash
# 主實驗(三種攻擊模式: image / text / both)
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --attack_mode image
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --attack_mode text --text_attack_type leetspeak --text_corruption_rate 1.0
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --attack_mode text --text_attack_type char_swap --text_num_swaps 8
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --attack_mode both --epsilon 0.05 --text_attack_type char_swap --text_num_swaps 2 --config_name joint_light

# 攻擊強度 / 防禦強度掃描
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --epsilon 0.05 --config_name weak
python -u run_experiment.py --dataset_root FB --num_samples 200 --seed 0 --noise_std 0.1 --config_name noise01

# HarMeme(Yes 只有 61 筆,直接 -1 跑全部)
python -u run_experiment.py --dataset_root HarMeme --num_samples -1 --seed 0

# 7B(要降解析度)
python -u run_experiment.py --dataset_root FB --num_samples 30 --seed 0 --model_name qwen7b --min_pixels 65536 --max_pixels 200704 --config_name model7b

# 其他實驗
python -u clean_control_check.py --dataset_root FB --num_samples 30 --seed 0 --noise_std 0.3
python -u run_error_correction_experiment.py --dataset_root FB --num_samples 250 --seed 0
python -u run_textfooler_experiment.py --dataset_root FB --num_samples 30 --seed 0 --query_budget 200

# 彙整
python summarize_results.py
```

`run_experiment.py` 參數:`--dataset_root --data_file --model_name --num_samples --max_new_tokens --seed --epsilon --alpha --num_steps --noise_std --num_noise_samples --min_pixels --max_pixels --config_name --output_file --attack_mode --text_attack_type --text_corruption_rate --text_num_swaps`。輸出檔名會帶資料集、模型、攻擊模式、關鍵參數,不會互相覆蓋。

---

## 6. 已完成的實驗與數據

### 6.0 修正 token 後的新結果(2026-10,token_mode=nospace,這一節才是可以引用的數字)

**乾淨評估**(`run_clean_eval.py`,FB dev n=200 混合(Yes 93 / No 107),seed 0,noise 0.3,ns 5):

| 方法 | acc | F1 | recall | 修正 gt=Yes | 修正 gt=No | 傷害 gt=Yes | 傷害 gt=No |
|---|---|---|---|---|---|---|---|
| 無防禦(generate) | 65.5% | 0.457 | 31.2% | - | - | - | - |
| 文字(多數決) | **69.5%** | **0.573** | **44.1%** | 12/64 | 0/5 | 0/29 | 4/102 |
| 文字(加權) | 64.5% | 0.489 | 35.5% | 9/64 | 2/5 | 5/29 | 8/102 |
| 像素(多數決) | 64.5% | 0.423 | 28.0% | 0/64 | 2/5 | 3/29 | 1/102 |
| 像素(加權) | 64.0% | 0.438 | 30.1% | 6/64 | 2/5 | 7/29 | 4/102 |
| 組合(多數決) | 66.0% | 0.492 | 34.4% | 5/64 | 2/5 | 2/29 | 4/102 |
| 組合(加權) | 64.5% | 0.466 | 33.3% | 9/64 | 2/5 | 7/29 | 6/102 |

- 只有「文字(多數決)」明顯改善,而且幾乎沒有傷害。McNemar 12 vs 4,p≈0.08,n=200 還不顯著,要跑全部 500 筆。
- 加權投票全部不比多數決好,傷害較大 → 第一個 token 的 logit 差距不是可靠的信心分數。
- 單次 forward 的 logit 判斷和 generate 有 17/200 不一致(邊界樣本),小差距的樣本本身就不穩。
- 組合(多數決)有 6 筆 Unclear:3 問法 × 2 雜訊 = 6 票,偶數票會平手。

**圖片攻擊**(`run_experiment.py`,FB n=200 → 乾淨答對 49,eps 0.1,a 0.04,s 3,noise 0.3):
攻擊成功 24/49(49%)。恢復率:文字 20.8%、文字加權 16.7%、像素 33.3%、像素加權 33.3%、**組合 41.7%**、組合加權 25.0%。


**完整 dev(500 筆,Yes 250 / No 250,seed 0,noise 0.3,ns 5)— 目前最主要的數字:**

| 方法 | acc | F1 | recall | 修正 gt=Yes | 修正 gt=No | 傷害 gt=Yes | 傷害 gt=No |
|---|---|---|---|---|---|---|---|
| 無防禦(generate) | 59.2% | 0.374 | 24.4% | - | - | - | - |
| 文字(多數決) | **61.4%** | **0.453** | **32.0%** | 20/189 | 0/15 | 1/61 | 8/235 |
| 文字(加權) | 58.8% | 0.401 | 27.2% | 18/189 | 3/15 | 11/61 | 12/235 |
| 像素(多數決) | 57.6% | 0.329 | 20.8% | 0/189 | 3/15 | 9/61 | 2/235 |
| 像素(加權) | 60.2% | 0.402 | 26.8% | 19/189 | 5/15 | 13/61 | 6/235 |
| 組合(多數決) | 58.2% | 0.371 | 24.4% | 7/189 | 3/15 | 7/61 | 8/235 (11 Unclear) |
| 組合(加權) | 58.6% | 0.382 | 25.6% | 20/189 | 3/15 | 17/61 | 9/235 |

- 文字(多數決)McNemar 20 vs 9,p≈0.06(邊緣)。其他方法和無防禦沒有顯著差異。
- 攻擊(dev 全部 250 筆 Yes → 乾淨答對 61,攻擊成功 28 = 45.9%):恢復率 文字 25.0%、文字加權 17.9%、像素 32.1%、像素加權 25.0%、**組合 46.4%(13/28,95% CI 30%~64%)**、組合加權 39.3%。
- 門檻校正(`calibrate_threshold.py`,train 300 筆選門檻):clean_logit tau=-0.19、text_w tau=+0.31,dev 上幾乎沒改善(clean_logit_cal acc 65.0% vs 無防禦 65.5%,n=200)→ 單純調門檻沒用,Yes/No 分數本身重疊太多。
- 單次 forward 的 logit 判斷和 generate 不一致約 8%,原因未明(已確認 generation_config 沒有 repetition_penalty)。加權投票已決定不用,優先度低。

**換問法消融**(`ablate_templates.py`,dev 500):

| 規則 | acc | F1 | recall | precision |
|---|---|---|---|---|
| q0 原始問法 | 58.6% | 0.363 | 23.6% | 78.7% |
| q1 | 59.4% | 0.464 | 35.2% | 68.2% |
| q2 | **62.8%** | 0.505 | 38.0% | 75.4% |
| 多數決 | 61.4% | 0.453 | 32.0% | 77.7% |
| any-Yes | 61.4% | **0.537** | **44.8%** | 67.1% |
| all-Yes | 58.0% | 0.323 | 20.0% | 83.3% |

- **q2 單獨就比多數決好 → 文字防禦的乾淨增益主要來自比較好的問法,不是投票。** 不能再說「投票提升準確率」。
- any-Yes(任一問法說 Yes 就送人工審核)recall/F1 最高,符合審核情境。
- 注意:這是在 dev 上比較出來的,要選規則/問法必須在 train 上選,dev 只報最終結果。

**目前決定(2026-10-06):** 放棄加權投票與門檻校正;下一步先把攻擊改成真實的像素空間攻擊(`run_image_attack.py`)。

**真實像素空間攻擊**(`run_image_attack.py`,dev 全部 250 筆 Yes,L-inf 8/255、alpha 2/255、10 步、存 PNG 再讀回,PSNR 34.3 dB):
- 乾淨答對且縮放存檔後仍答 Yes:58 筆(另 3 筆只因縮放存檔就翻成 No,已排除)。
- 攻擊成功 36/58(62.1%);對抗圖再轉 JPEG q75 後仍成功只剩 16/58(27.6%)。
- 恢復率(36 筆):**JPEG 55.6%**、any-Yes 44.4%、雜訊 36.1%、組合 33.3%、換問法多數決 8.3%。攻擊沒翻掉的 22 筆:只有組合傷害 1 筆(4.5%)。
- **結論:最簡單的 JPEG baseline 目前贏過所有我們的防禦。** 下一步:JPEG-aware 攻擊(EOT)、JPEG + 我們防禦的組合與乾淨圖副作用、在 train 上選問法/規則、評估更強模型。
- 進度報告簡報(3 頁,2026-10 週四):https://claude.ai/artifact/TsBgSDZ3Ea6F7QAGskRHYS

**下一階段計畫(2026-10-07):**
1. `run_robust_suite.py` 先在 train 上跑(開發用),找出各防禦在哪種攻擊下失效;最終結果才在 dev 上報。
2. 新防禦:transform(5 種圖片轉換 + 多數決)、tq(5 轉換 × 3 問法;tq_majority / tq_anyq),投票不一致 → 送人工審核。
3. pgd_jpeg(BPDA 直通 JPEG)是「會撐過 JPEG」的攻擊;之後再做「知道防禦方式」的 adaptive 攻擊(EOT over transforms)。
4. 相關論文:HateProof(WWW'23,SaltPepper/Spread/文字攻擊經 OCR)、RA-HMD(EMNLP'25,Qwen2-VL-2B zero-shot HatefulMemes acc 54.2%)、Meme Trojan(AAAI'25,後門)、Mitigating...MULTILATE(未審查 preprint)。這四篇的防禦都要訓練;本研究是 training-free。

**攻擊組合 × 防禦組合(`run_robust_suite.py`,FB train,開發用,2026-10-08):**

攻擊模式(150 筆 Yes → 模型原本答對 71 筆;數字 = 攻擊後仍認出是仇恨的比例,越高越好):

| 攻擊 | 攻擊成功 | none | jpeg | text | text_anyyes | noise | transform | tq_majority | tq_anyq |
|---|---|---|---|---|---|---|---|---|---|
| pgd | 46.5% | 53.5% | 76.1% | 63.4% | 74.6% | 73.2% | 83.1% | 85.9% | **90.1%** |
| pgd_jpeg | 32.4% | 67.6% | **50.7%** | 77.5% | 80.3% | 73.2% | 67.6% | 80.3% | **85.9%** |
| saltpepper | 2.8% | 97.2% | 98.6% | 93.0% | 98.6% | 97.2% | 98.6% | 95.8% | 98.6% |
| spread | 19.7% | 80.3% | 74.6% | 76.1% | 88.7% | 73.2% | 83.1% | 80.3% | **90.1%** |

乾淨模式(200 筆混合:Yes 73 / No 127,不攻擊):

| 防禦 | acc | recall | F1 | FP | fixed | harmed |
|---|---|---|---|---|---|---|
| none | 69.5% | 34.2% | 0.450 | 13 | - | - |
| jpeg | 68.5% | 39.7% | 0.479 | 19 | 6 | 8 |
| text | 67.0% | 35.6% | 0.441 | 19 | 2 | 7 |
| text_anyyes | 67.0% | 57.5% | 0.560 | 35 | 17 | 22 |
| noise | 69.0% | 32.9% | 0.436 | 13 | 1 | 2 |
| transform | 69.0% | 35.6% | 0.456 | 15 | 1 | 2 |
| tq_majority | 69.5% | 41.1% | 0.496 | 18 | 6 | 6 |
| tq_anyq | 66.0% | 53.4% | 0.534 | 34 | 14 | 21 |

- **tq_anyq 在四種攻擊下都最好**;tq_majority 次之,而且乾淨圖準確率不掉(69.5% = none)、recall +7。
- **JPEG 在 pgd_jpeg 下失效**:50.7%,比不防禦(67.6%)還差 → JPEG baseline 擋不住知道會被壓縮的攻擊者。transform 含 JPEG 轉換,在 pgd_jpeg 下也掉到 67.6%。
- 代價:tq_anyq 乾淨圖 FP 13 → 34(正常迷因誤判率 10% → 27%);tq 每張圖要問 15 次。
- tq 15 票不一致比例:乾淨 29%、pgd 38%、pgd_jpeg 52% → 攻擊會讓投票更不一致,可能可以當偵測訊號,但乾淨圖就有 29%,直接送人工太多。
- 在 dev 上「文字多數決」比不防禦好(61.4 vs 59.2),但這裡 train 200 筆反而比較差(67.0 vs 69.5)→ 小幅差異不穩定,不要過度解讀。
- 樣本數:71 筆攻擊,單一比例的 95% 信賴區間約 ±10%。
- 下一步:`--attacks pgd_eot`(知道 3 問法 + 5 轉換的 adaptive 攻擊)在 train 上測 tq 是否仍撐得住;之後在 dev 上報最終結果,再做 HarMeme。


以下數字全部來自上面的程式。**加權投票那幾欄在處理第 0 節的問題前,先當作「待驗證」。**

### 6.1 攻擊強度掃描(FB,n=200 抽樣,noise_std=0.3)

乾淨答對 49 筆進入攻擊。

| epsilon | 被攻擊成功 / 49 | 文字(多數決) | 文字(加權) | 像素(多數決) | 組合(多數決) | 組合(加權) |
|---|---|---|---|---|---|---|
| 0.05 | 17 | 5.9% | 82.4% | 41.2% | 41.2% | 94.1% |
| 0.10 | 23 | 13.0% | 78.3% | 21.7% | 39.1% | 95.7% |
| 0.20 | 21 | 14.3% | 71.4% | 28.6% | 33.3% | 90.5% |

(較早、尚未加入加權投票的版本:ε=0.05/0.1/0.2 的攻擊成功率 36.7% / 51.0% / 44.9%,多數決恢復率文字 16.7/20.0/13.6、像素 44.4/40.0/18.2、組合 50.0/44.0/31.8。)

### 6.2 防禦強度掃描(FB,n=200,epsilon=0.1)

| noise_std | 被攻擊成功 | 文字(多數決) | 文字(加權) | 像素(多數決) | 組合(多數決) | 組合(加權) |
|---|---|---|---|---|---|---|
| 0.1 | 24 | 16.7% | 83.3% | 4.2% | 16.7% | 83.3% |
| 0.2 | 21 | 14.3% | 66.7% | 14.3% | 19.0% | 85.7% |
| 0.3 | 23 | 30.4% | 82.6% | 30.4% | 39.1% | 95.7% |
| 0.5 | 22 | 18.2% | 72.7% | 45.5% | 45.5% | 100% |

觀察:噪音從 0.1 到 0.3 恢復率上升,0.3 到 0.5 差不多持平。
**注意:`pixel_weighted` 在這些 n=200 的實驗裡還沒有數字**——這幾組是在加入 `randomized_smoothing_defense_weighted` 之前跑的。`summarize_results.py` 也還沒加這一欄。

### 6.3 乾淨副作用(clean-control)

只在 `noise_std=0.3`、混合抽樣 30 筆、乾淨答對 21 筆的情況下做過,**只有多數決版本**:文字 100%、像素 90.5%、組合 95.2%。加權版本、其他 noise_std、HarMeme 都沒做。

### 6.4 沒有攻擊的誤判修正(FB,混合 Yes/No,n=250 抽樣)

乾淨準確率 166/250 = 66.4%;原本答錯 84 筆。

| 防禦 | 多數決 | 加權 |
|---|---|---|
| 文字 | 14.3% | 53.6% |
| 像素 | 2.4% | 46.4% |
| 組合 | 8.3% | 51.2% |

**問題:修正率貼近 50%(二元亂猜期望值)**,還沒拆方向(gt=Yes 答成 No / gt=No 答成 Yes)分開算,也沒有量原本答對的 166 筆被防禦弄錯的比例。

### 6.5 文字攻擊

- 規則型(leetspeak、char_swap):即使 leetspeak 100% 替換、char_swap 交換 8 組,攻擊成功率 0%(小樣本,乾淨答對 4 筆)。
- TextFooler(TextAttack,query_budget=200,每筆約 35~57 秒):抽樣 30 筆、乾淨答對 8 筆,1 筆「成功」(12.5%)。查看該成功案例:
  - 原文 `doesnt have food, water, electricity proud of nuclear weapons`
  - 攻擊後 `haha got food, water, electricity grandiose of nuclear disarmament`
  - 是整句語意被反轉(不再是諷刺攻擊),不是保留仇恨語意的偽裝。TextFooler 只約束文字語意相近,不管圖文聯合語意。
- 模型在 TextFooler 過程中**有**看圖(wrapper 每次都帶同一張圖),判斷無惡意是因為文字本身語意已變成正面。

### 6.6 其他

- HarMeme(61 筆 Yes 全跑):乾淨答對 9、攻擊成功 5,樣本太小,早期版本(無加權),僅供參考。
- 7B(n=30):乾淨答對 11、攻擊成功 4,樣本太小。
- `--attack_mode both` 只跑過 n=10 的連通測試,結果與純圖片攻擊相同(文字攻擊沒有貢獻)。

---

## 7. 其他已知問題

1. **再現性**:相同參數重跑,文字防禦(`text`)的結果曾不同(medium 13.0% vs noise03 30.4%,設定相同)。`pixel` / `combo` 因為有 `seed_offset` 是可重現的。文字防禦沒有雜訊,懷疑是 GPU 浮點誤差在邊界樣本翻轉。
2. `combined_defense*` 內對每個問法都用相同的 `seed_offset + i`,所以每個問法看到的是同一組雜訊。
3. 平手規則偏向 Yes(見第 0 節)。
4. 模型本身偏向回答 No(乾淨時 recall 偏低:早期 100 筆測試 accuracy 68%、recall 41.9%)。
5. TextFooler 每筆要查詢上百次且逐一呼叫 VLM,慢;還用到 Universal Sentence Encoder(TensorFlow)。
6. 目前進入評測的樣本數:FB 約 20~25(被攻擊成功的數量),HarMeme / 7B / TextFooler 都只有個位數。

---

## 8. 待辦(依優先順序)

**P0 — 加權投票偏誤**:已查明並修正(見第 0 節)。接下來用新程式重跑:
- `run_clean_eval.py`(乾淨、混合 Yes/No:整體指標、雙向修正率、傷害率)。
- `run_experiment.py` 攻擊實驗(攻擊現在打在正確 token 上)。

**P1 — 誤判修正 / train-free 校正**(`calibrate_threshold.py` 已寫好,待跑)
- `run_clean_eval.py` 已拆方向、算傷害率。
- 模型偏向答 No(recall 低):用存下來的 logit 分數做 train-free 校正(在 train split 上選門檻或 contextual calibration,不動模型權重),在 dev 上評估。

**P2 — 補齊既有實驗**
- `summarize_results.py` 加 `pixel_weighted_defense_recovery_rate`。
- 重跑 noise_std 掃描,補 `pixel_weighted`。
- `clean_control_check.py` 加入加權版本,涵蓋各 noise_std 與混合 Yes/No。

**P3 — 延伸**
- 圖文聯合攻擊(輕度)的概念驗證:`--attack_mode both` 搭配較小的 epsilon 與輕度文字攻擊,並要有足夠樣本。
- 擴大 TextFooler 樣本數(目前只抽 30 筆、8 筆進入測試)。
- 考慮加入 TTC 式的偵測步驟(先判斷是否被攻擊,乾淨圖不套防禦),降低對乾淨圖的傷害。
- 改寫問法目前是人工寫死的三個模板,只適用這個任務;若要換任務得重寫模板或接外部 LLM 產生。

---

## 9. 使用者偏好(給 Claude Code)

- 回答用繁體中文,口語、直接,不要太多官樣文字或誇飾用語。
- 要程式碼時給**整份可複製貼上的檔案**,不要只給要自己拼接的片段;指令也要能直接貼到 CMD 執行。
- 使用者不熟 GitHub,不要假設她會用 git 流程。
- 動手改既有檔案前先說明改什麼;已經跑出數據的實驗腳本盡量不要破壞,新功能用新參數或新檔案加。
- 有數字看起來太好或太怪時,先指出疑點再往下做。
- 報告用的簡報講稿裡**不要出現「學姊的論文」**這類說法。
