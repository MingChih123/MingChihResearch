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
- 專案路徑:`C:\Users\USER\Desktop\MChih_Exp\Exp\`(2026-10-15 起;舊的在 `C:\Users\USER\OneDrive\Desktop\MChih_Exp\2\`)

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
config.py / common.py           資料集路徑、模型別名;載入模型、build_inputs、generate、parse_answer、get_yes_no_token_ids
attack.py / defense.py          舊版攻擊(改內部張量)、文字攻擊;各種防禦、問法模板(FB 預設 + HARMEME_TEMPLATES)
run_robust_suite.py             ★ 主實驗:攻擊組合 × 防禦組合;--clean_only、--tq_transforms、--eot_defense_only、
                                --template_set(auto:HarMeme 自動換問法)、--random_start(多 seed 用)
run_image_attack.py             第一版真實像素攻擊 + JPEG baseline(run_robust_suite 也 import 它的函式,不能刪)
run_clean_eval.py               乾淨混合樣本評估(含加權投票),ablate_templates / calibrate_threshold 會 import
ablate_templates.py             3 種問法消融
run_experiment.py               舊版攻擊實驗(論文中比較加權 vs 多數決用)
diagnose_weighted_bias.py       加權投票 token 錯誤診斷(附錄)
calibrate_threshold.py          門檻校正(負面結果)
run_textfooler_experiment.py / test_textfooler.py / textattack_wrapper.py   TextFooler 文字攻擊探索
summarize_all.py                把 output 結果整理成 _table_*.csv
analyze_robust.py               McNemar、95% CI、成本消融;--main_transforms resize50 從投票重算主方法
aggregate_seeds.py              多個 seed → 平均 ± 標準差(_seeds_<tag>.csv)
make_harmeme_test_vqa.py        HarMeme test.jsonl → test_vqa.json + valtest_vqa.json(先用 val 比對規則)
fix_harmeme_paths.py / check_harmeme.py   HarMeme 路徑修復與檢查
summarize_results.py            舊版攻擊實驗的結果整理
README.md                       ★ 給學弟妹的操作手冊(安裝、資料、所有指令、多 seed、雲端要放什麼)
THESIS_OUTLINE.md               ★ 論文大綱(每章的表格、結論、注意事項)
archive/                        已淘汰的舊檔案(pgd_attack、eval_vqa、clean_control_check 等),不要用
output/                         所有實驗結果 json / csv
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
**Adaptive 攻擊 pgd_eot(知道 3 問法 + 5 轉換;同樣 71 筆,2026-10-12):** 攻擊成功 39.4%;仍認出比例:none 60.6%、jpeg 67.6%、text 70.4%、text_anyyes 76.1%、noise 71.8%、transform 67.6%、tq_majority 73.2%、**tq_anyq 81.7%**。
- tq 在專門針對它的攻擊下有下降(tq_majority 85.9 → 73.2、tq_anyq 90.1 → 81.7),但仍高於不防禦(60.6)。
- 公平比較 = 每個防禦對上「知道它的攻擊」:JPEG vs pgd_jpeg 50.7%;tq_majority vs pgd_eot 73.2%;tq_anyq vs pgd_eot 81.7%;不防禦 vs pgd 53.5%。
- 注意:pgd_eot 只針對 tq 設計,其他防禦(noise、text)在這一列沒有被針對,數字偏樂觀。
- 只看仇恨迷因的「仍認出比例」會偏好容易說 Yes 的規則(anyq),要搭配乾淨模式的誤判率一起看:正常迷因誤判率 none 10%、jpeg 15%、tq_majority 14%、tq_anyq 27%。
- 下一步:在 dev 上跑最終結果(攻擊模式含 pgd_eot + 乾淨模式),再做 HarMeme。

**★ dev 最終結果(`run_robust_suite.py`,FB dev 全部,2026-10-13)— 論文主要數字:**

攻擊模式(250 筆 Yes → 模型原本答對且縮放後仍答對 59 筆;仍認出比例,越高越好):

| 攻擊 | 攻擊成功 | none | jpeg | text | text_anyyes | noise | transform | tq_majority | tq_anyq |
|---|---|---|---|---|---|---|---|---|---|
| pgd | 61.0% | 39.0% | 64.4% | 42.4% | 66.1% | 59.3% | 71.2% | 74.6% | **96.6%** |
| pgd_jpeg | 55.9% | 44.1% | 40.7% | 49.2% | 76.3% | 55.9% | 55.9% | 57.6% | **89.8%** |
| pgd_eot | 57.6% | 42.4% | 61.0% | 42.4% | 64.4% | 55.9% | 50.8% | 49.2% | **78.0%** |
| saltpepper | 11.9% | 88.1% | 86.4% | 91.5% | 100% | 88.1% | 88.1% | 93.2% | 100% |
| spread | 28.8% | 71.2% | 78.0% | 74.6% | 89.8% | 83.1% | 78.0% | 81.4% | 89.8% |

乾淨模式(dev 500:Yes 250 / No 250):

| 防禦 | acc | recall | F1 | FP(/250) | fixed | harmed |
|---|---|---|---|---|---|---|
| none | 59.2% | 23.6% | 0.366 | 13 | - | - |
| jpeg | 58.8% | 24.0% | 0.368 | 16 | 10 | 12 |
| text | 61.0% | 30.4% | 0.438 | 21 | 18 | 9 |
| text_anyyes | 62.0% | 45.6% | 0.545 | 54 | 55 | 41 |
| noise | 57.8% | 20.8% | 0.330 | 13 | 5 | 12 |
| transform | 59.6% | 25.2% | 0.384 | 15 | 9 | 7 |
| tq_majority | 60.2% | 28.8% | 0.420 | 21 | 18 | 13 |
| tq_anyq | **62.8%** | **47.2%** | **0.559** | 54 | 59 | 41 |

- **tq_anyq 在每種攻擊下都最好,而且乾淨圖準確率 / F1 也最高。** 在 dev(仇恨:正常 = 1:1)上 recall 提升抵過誤判增加;train(仇恨約 36%)上 tq_anyq 準確率反而略降 → 結果受類別比例影響,論文要說明。
- **tq_anyq 和 text_anyyes 的乾淨圖誤判數完全一樣(54/250 = 21.6%)**,但攻擊下 tq_anyq 強很多(pgd 96.6 vs 66.1、pgd_jpeg 89.8 vs 76.3、pgd_eot 78.0 vs 64.4)→ 同樣的誤判成本,圖片轉換帶來額外的強健性。
- 消融(各元件都有貢獻):只有轉換(transform)pgd 71.2 / pgd_eot 50.8;只有問法 any-Yes 66.1 / 64.4;兩者合起來 tq_anyq 96.6 / 78.0。
- **tq_majority 在 dev 上撐不住 adaptive 攻擊**:pgd_eot 只剩 49.2%,和不防禦(42.4%)差不多(train 是 73.2%)→ 主推 tq_anyq,tq_majority 當作對照。
- **JPEG 在 pgd_jpeg 下再次失效**(40.7% < 不防禦 44.1%),和 train 一致。
- dev 上攻擊比 train 強(pgd 成功 61% vs 46.5%)。
- 代價:tq_anyq 正常迷因誤判率 5.2% → 21.6%、每張圖問 15 次。
- 顯著性與成本消融:`analyze_robust.py`(用存下來的投票重算,不用 GPU)。

**顯著性(McNemar 精確檢定,dev,59 筆攻擊 / 500 筆乾淨):**
- tq_anyq vs 不防禦:pgd 34 vs 0、pgd_jpeg 27 vs 0、pgd_eot 22 vs 1(皆 p<0.001);saltpepper p=0.016、spread p=0.001。
- tq_anyq vs JPEG:pgd p<0.001、pgd_jpeg p<0.001、pgd_eot p=0.021、saltpepper p=0.008、spread p=0.039 → 全部顯著。
- tq_anyq vs text_anyyes(同樣 3 個問法、同樣誤判數):pgd p<0.001、pgd_jpeg p=0.008、pgd_eot p=0.022 → 圖片轉換在梯度攻擊下的貢獻是顯著的;saltpepper / spread 不顯著。
- tq_majority:多數比較不顯著(pgd_eot vs 不防禦 p=0.29)。
- 乾淨圖準確率:tq_anyq 62.8% vs 不防禦 59.2%,p=0.089(不顯著,但至少沒有變差);train 上 66.0% vs 69.5%,p=0.31。
- train 上主要結論一致(tq_anyq vs 不防禦、vs JPEG 在 pgd / pgd_jpeg 下顯著)。

**成本消融(用存下來的投票重算,`_analysis_subsets_*.csv`):**
- 在 train 上挑:只用 **resize50 一種轉換 × 3 問法 + anyq(每張圖 3 次)** 就和 15 次一樣好(train:pgd 94.4、pgd_jpeg 94.4、spread 93.0、誤判 28.3%;15 次是 90.1 / 85.9 / 90.1 / 26.8%)。
- dev 上驗證 resize50 × 3 問法 anyq:乾淨 acc 64.4% / recall 50.0% / 誤判 21.2%;pgd 96.6、pgd_jpeg 91.5、pgd_eot 81.4、saltpepper 100、spread 93.2 → 每一項都 ≥ 15 次的版本,成本只有 1/5。
- 和 text_anyyes(也是每張 3 次、誤判 21.6%)比:pgd 96.6 vs 66.1、pgd_jpeg 91.5 vs 76.3 → 同樣成本、同樣誤判,多了大量抗攻擊能力。
- JPEG 類轉換是比較弱的一環(會被 pgd_jpeg 打穿),縮小再放大最強。
- **注意**:dev 上的 pgd_eot 是針對 5 種轉換設計的,不是針對 resize-only;要說 resize-only 版本強健,必須再測「只針對 resize50 的 adaptive 攻擊」:
  `python -u run_robust_suite.py --dataset_root FB --num_samples -1 --seed 0 --attacks pgd_eot --tq_transforms resize50 --eot_defense_only --defenses none,jpeg,text,tq`
- `run_robust_suite.py` 新參數:`--tq_transforms`(tq / transform / pgd_eot 用哪些轉換)、`--eot_defense_only`(pgd_eot 只打防禦看得到的轉換版本)。

**resize-only 版本的專門攻擊(dev 59 筆,`--attacks pgd_eot --tq_transforms resize50 --eot_defense_only`,2026-10-13):**
攻擊成功(對不防禦)49.2%;仍認出:none 50.8%、jpeg 66.1%、text 52.5%、text_anyyes 76.3%、tq_majority(resize×3 多數決)50.8%、**tq_anyq(resize×3 any)76.3%**。(noise / transform 沒跑所以 CSV 是空白。)
- resize-only tq_anyq 被專門攻擊後 76.3%,和 15 次版本被它的 adaptive 攻擊後 78.0% 差不多 → **3 次版本一樣強健,成本 1/5,定為主方法。**
- 這一列的 text_anyyes(76.3%)不能拿來比:攻擊只優化「縮小後」的圖,原圖沒有被針對,所以 text_anyyes 偏樂觀。
- tq_majority 在專門攻擊下 = 不防禦(50.8%)→ 多數決規則不夠,要用 any 規則。
- 小差異:15 次版本的 pgd_eot 有混入原圖(沒用 --eot_defense_only),攻擊稍弱;要完全公平可以用 --eot_defense_only 重跑 15 次版本(選做)。

**主方法(暫定名稱 Resize-and-Rephrase):把圖縮小一半再放大 → 用 3 個問法各問一次 → 任一說 Yes 就標記。每張圖 3 次查詢、不訓練。**

dev 主表(仍認出比例 / 乾淨圖):

| | 不防禦 | JPEG | 換問法 any-Yes | 主方法(3 次) |
|---|---|---|---|---|
| 乾淨 acc | 59.2% | 58.8% | 62.0% | **64.4%** |
| 乾淨 recall | 23.6% | 24.0% | 45.6% | **50.0%** |
| 正常迷因誤判率 | 5.2% | 6.4% | 21.6% | 21.2% |
| pgd | 39.0% | 64.4% | 66.1% | **96.6%** |
| pgd_jpeg | 44.1% | 40.7% | 76.3% | **91.5%** |
| 針對自己的攻擊 | 39.0%(pgd) | 40.7%(pgd_jpeg) | 未測 | **76.3%**(pgd_eot,只針對 resize) |
| saltpepper | 88.1% | 86.4% | 100% | 100% |
| spread | 71.2% | 78.0% | 89.8% | **93.2%** |

- 未做(選做):針對「換問法 any-Yes」本身的 adaptive 攻擊(EOT 只對 3 個問法、原圖)。

**下一步(2026-10-13 起):**
1. HarMeme:`fix_harmeme_paths.py` → `make_harmeme_test_vqa.py`(會先比對 val 規則,看到 [STOP] 要回報)→ README 4.5 的三個指令(用 `annotations/valtest_vqa.json`,不調參)。
2. 主方法顯著性:`analyze_robust.py ... --main_transforms resize50`(不用重跑)。
3. 多 seed:README 第 5 節(`--random_start`,3 個 seed,`aggregate_seeds.py`)。
4. 論文大綱見 `THESIS_OUTLINE.md`,操作手冊見 `README.md`。

**HarMeme 結果(val+test 531 筆,不調參,2026-10-15):**
- 乾淨(185 Yes / 346 No):none acc 67.4% / recall 12.4%(23/185)/ FP 11;tq_anyq(15 次版)67.6% / 16.2% / FP 17;text_anyyes recall 14.1%。模型在 HarMeme 上極度保守,any-Yes 幾乎拉不起 recall(和 FB 差很多)。
- 攻擊(只有 23 筆進入攻擊,95% CI 約 ±20%):pgd none 21.7 / jpeg 78.3 / tq_anyq 82.6;pgd_jpeg none 39.1 / jpeg 34.8 / tq_anyq 73.9;pgd_eot(5 轉換)none 34.8 / tq_anyq 52.2。
- **針對 resize 的 adaptive 攻擊(eotdef):tq_anyq 65.2% = 不防禦 65.2%** → 主方法在 HarMeme 上撐不住專門攻擊。要誠實寫進限制。

**老師回饋(10/15 報告後)與處理:** 全文見 `ADVISOR_FEEDBACK.md`(第 8 節是驗證紀錄)。
- 威脅模型要先講明:只改圖片 ≤8/255、模型乾淨可信、攻擊和回答同一模型、問法人工寫;投毒/後門、文字注入、黑盒遷移不在範圍內。
- 不要做:重新訓練、後門偵測或移除。
- 舊 `pgd_eot` 的損失是三個問法加總,不保證三個都被打成 No → 76.3% 可能偏高。新增 `pgd_eot_max`(soft-max over 問法,打最還在說 Yes 的那個),舊結果不動。`analyze_robust.py` 新增 `_analysis_perq_<tag>.csv`(每問法成功率、三個都 No 比例)與「resize50 + 單一問法」同成本對照。
- 待做:P1 跑 `pgd_eot_max`(FB dev 10 步 / 30 步、HarMeme);P2 遷移測試;P3 caption 文字注入;P5 TextFooler 重跑;P6 全黑/空白、圖上文字。
- 2026-10-16:`run_robust_suite.py` 加了不用梯度的圖文攻擊 `homoglyph` / `caption_inject` / `image_text`,和文字端防禦 `norm` / `guard` / `main_safe`(norm + guard + 主方法),檔名帶 `_txtdef`。所有攻擊與防禦都放在 `run_robust_suite.py`,使用者不想要更多檔案。
- 2026-10-16 結果(FB dev 59 筆):pgd_eot_max 10 步 主方法 66.1%(三個都 No 33.9%)、30 步 55.9% vs 不防禦 44.1%(McNemar 11 vs 4,p=0.12,不顯著);HarMeme 23 筆 主方法 43.5% < 不防禦 65.2%。homoglyph 攻擊成功 0%(模型讀得懂同形字);caption_inject 成功 98.3%,所有防禦幾乎無效(main_safe 15.3%);image_text 成功 69.5%,主方法救回到 62.7%。guard/norm 乾淨圖副作用很小但也擋不住。
- image_text 的白條會蓋住原圖下方的迷因文字(使用者發現),新增 `image_text_pad`(多接一條白色區域,不蓋原圖)。
- image_text_pad 結果(59 筆):攻擊成功 59.3%;主方法 69.5% vs 不防禦 40.7%(18 vs 1,p=0.0001)、vs JPEG 47.5%(p=0.004)、vs 只換問法 57.6%(p=0.039)→ 縮放對「圖上加字」也有貢獻。
- 2026-10-18:加反問防禦 `neg` / `main_neg`(3 正向 + 1 反問,答 No 才標記)和攻擊 `caption_persuade`(只說服、不下指令),用 `--config_name polar`。風險:模型偏向答 No,反問可能誤判大增。
- 2026-10-18 反問結果:失敗。乾淨 neg 誤判 66%、main_neg 82%;caption_inject 下 neg 0%(模型是「相信文字裡說無害」,不是照 Answer No 做);caption_persuade(只說服)成功 91.5%。放棄反問。
- 攻擊強度(非 adaptive,59 筆):PGD 主方法 ε4/8/16 = 100/96.6/91.5%,不防禦 44.1/39.0/30.5%;PGD_JPEG 主方法 94.9/91.5/89.8%,JPEG 57.6/40.7/44.1%。
- 2026-10-19:`analyze_robust.py` 加「先判斷再防禦」gate_q0/1/2(離線,用存好的 text + resize50 投票),輸出 `_analysis_gate_<tag>.csv`。
- gate 結果:沒有額外好處。gate_q2 ≈「resize50 + q2 單問法」(誤判 11.2% vs 10.8%、pgd 86.4% 兩者相同),pgd_eot_max 下 42.4% ≈ 不防禦。結論:主方法分「低誤判版 resize+q2(1 次,誤判 10.8%,pgd 86.4%)」和「高偵測版 resize×3 any(3 次,誤判 21.2%,pgd 96.6%)」。
- 2026-10-19:加 `--offset K`(檔名 `_off<K>`),用 train 裡開發時沒用過的樣本擴大攻擊樣本數(攻擊模式 train 用過前 150 筆 Yes,乾淨模式用過前 200 筆)。
- 2026-10-19:讀了 RAMPART(訓練式,CLIP+LoRA 對抗微調+RGCL,FB 測試 1000 張全部攻擊,ε=1/255,報 ACC/AUC;乾淨 FB acc 75.1%)。加 `--attack_all`(全部圖都攻擊:仇恨→No、正常→Yes,報整體/分類別準確率,檔名 `_all`),讓主表和其他論文同格式。簡報不要寫「學姊的論文」,寫 RAMPART / 先前研究。
- 2026-10-27 attack_all + pgd_eot_max(只針對 resize,500 張,仇恨→No、正常→Yes,goal=any 對正常迷因):整體準確率 none 51.0% / jpeg 53.6% / text_anyyes 50.0% / tq_majority 45.0% / **主方法 42.2%(比不防禦還低)**;主方法 仇恨 26.0%、正常 58.4%(乾淨時 78.8%)→ any 規則讓攻擊者只要騙到一個問法就能冤枉正常迷因,是主方法在 adaptive 下的主要弱點。
- 讀了 F3(ACM MM 2025,training-free,用注意力+梯度加雜訊淨化,測 Qwen2.5-VL-3B 等,有 adaptive AutoAttack;R&P 隨機縮放補邊在 adaptive 下只有 43.08)與 VALD(2026,圖片描述任務,轉換偵測 + Gemma-3-27B 整合)。建議論文定位為「仇恨迷因審核情境下 training-free 推論階段防禦的系統性評估 + 一個簡單方法」,F3 列為主要比較對象(github.com/btzyd/F3)。
- 2026-10-28:使用者決定主軸是「提出新方法」。新增(都在 `run_robust_suite.py`,`_v2def`):
  - `f3`:F3 移植到 Qwen2-VL(`f3_purify`,參考官方 `models/LLaVA.py::purify_f3`;用最後一個 prompt 位置對圖片 token 的注意力,α=16/255、β=32/255;讀注意力時暫時切 eager)。尚未實作「知道 F3 的攻擊」。
  - `cons`(新方法,暫名「多視角共識 + 人工審核」):K=4 個隨機縮放視角(比例 0.4~0.6、隨機放大方法)× 3 問法;迷因文字拿掉一句(沒標點時拿前半/後半)再問;Yes 票比例 ≥ high → Yes、≤ low → No、中間 → Review;圖片判 No 但某個拿掉一句的版本多數 Yes → Review。門檻預設 0.2 / 0.5,要在 train 上用 `_analysis_cons_<tag>.csv` 選。指標:無聲失誤率(有把握但錯)、人工審核率、自動判斷準確率。
  - `pgd_eot_cons`:知道 cons 的攻擊,每步從同一隨機分布抽 `--eot_views` 個視角,壓低平均 Yes 分數(推動多數)。
- 使用者已用 git 連上 GitHub(`git pull` 更新),不用再下載 ZIP。
- 使用者的資料夾已搬到 `C:\Users\USER\Desktop\MChih_Exp\Exp`(舊的是 `C:\Users\USER\OneDrive\Desktop\MChih_Exp\2`)。


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
- 2026-10-28 train 選門檻結果(offset 200;乾淨 200 張、attack_all 60 張 = 仇恨 20 / 正常 40,樣本小):
  - 乾淨:none 76.5%(這段 train 比 dev 簡單)、tq_majority 78.5%、tq_anyq 74.0%(FP 9→24)、cons 74.0%(Review 7% 算錯;自動判斷準確率 79.6%、無聲失誤 19%)、f3 74.0%(recall 46→42%)。
  - pgd:none 65.0 / tq_majority 78.3 / tq_anyq 76.7 / cons 71.7 / f3 75.0。攻擊成功率只有 17%(train 這段比較難打)。
  - **pgd_eot_cons(針對 cons):cons 58.3% < none 70.0%**,無聲失誤 35%;壞在正常迷因(benign 82.5→62.5%)。門檻怎麼調,最好也只有無聲失誤 25%(high=1.0,幾乎等於要全票才判 Yes)→ 調門檻救不了,cons 要改設計。
  - caption_persuade:全部防禦都救不回仇恨(5~20%);cons 的拿掉一句只能把 18% 送審核。
  - f3 還沒有針對它的攻擊,數字偏樂觀。
  - analyze_robust.py 修正:pgd_eot_cons 的 attack_info 沒有 yes_margin_per_question,原本會 KeyError。
- 2026-10-29:使用者不想用「送人工審核」當主方法(覺得不負責任)→ 主結果一律強制 Yes/No。離線重算 cons 強制版(Yes 票比例 ≥ t 判 Yes):pgd_eot_cons 下最好 68.3%(t=1.0),t=0.5 是 61.7%。
  注意公平比較:pgd_eot_cons 只打隨機視角,不防禦那欄 70% 沒有被針對;不防禦對上針對它的 pgd 是 65.0% → cons 強制版和不防禦差不多,沒有明顯好處。
- 2026-10-29:新增文字端 `strip`(METHOD_DEFENSES,檔名 _v2def):逐段問「這段是不是在替迷因下結論(無害 / 玩笑 / 不是仇恨)」,拿掉後用 resize50 × 3 問法判斷(strip = 多數決、strip_any = 任一)。新增攻擊 `caption_persuade_blend`(說服句改小寫、去標點,測切段弱點)。指令見 README 4.5d。
- 2026-10-31 strip 第一版結果(train,44 筆仇恨迷因;乾淨 200 張):caption_persuade 下 none 6.8% / tq_anyq 27.3% / **strip 90.9%**;blend 84.1%;caption_inject 90.9%;乾淨 strip 78.5%(= tq_majority)。
  **但數字是假象**:檢查器幾乎每段都說是(乾淨圖 344/352 段被拿掉、196/200 張文字全空)→ strip v1 ≈ 不給文字,模型自己讀圖上的字。
  真正的發現:攻擊只改「送進來的文字」時,不給文字就救得回來;但 FB 的文字本來就是圖上的字,現實中攻擊者要改文字就得寫在圖上(= image_text_pad),那時不給文字沒用。
  處理:`--strip_version 2`(預設)改成只給文字 + 例子(例子措辭和攻擊句不同);新增對照組 `nocap`;summary 會印 strip 拿掉的比例(乾淨圖應該接近 0)。指令見 README 4.5d,config_name strip2,攻擊加 image_text_pad。

- 2026-11-01 strip v2 結果(train;乾淨 200 張、攻擊 44 筆仇恨迷因,config strip2):
  - 乾淨:strip 拿掉 0% 的段落,準確率 78.5% = tq_majority(沒有副作用);nocap 也是 78.5%(FB 的圖說就是圖上的字,模型自己讀得到)。
  - 仍認出比例:caption_persuade none 6.8 / tq_majority 4.5 / tq_anyq 27.3 / **strip 95.5** / strip_any 100 / nocap 93.2;
    caption_persuade_blend 4.5 / 9.1 / 54.5 / **77.3** / 88.6 / 93.2;caption_inject 2.3 / 0 / 4.5 / **97.7** / 100 / 93.2;
    image_text_pad(字寫在圖上)63.6 / 75.0 / 93.2 / strip 75.0 / strip_any 93.2 / **nocap 54.5(會輸)**。
  - 攻擊時被拿掉的段落 20~35%(幾乎都是加進去的句子);blend 有 9/44 沒拿乾淨 → 混進原文是弱點。
  - 結論:strip 是目前唯一擋得住圖說說服 / 注入、又不傷乾淨圖、字寫在圖上時也不比主方法差的方法。nocap 在圖說攻擊下一樣強,但字在圖上時掉到 54.5%。
  - 還沒做:知道 strip 的攻擊(換說法躲過檢查器、改寫成不像免責聲明的句子)、dev 上的正式結果、HarMeme。
- 2026-11-01:新增 `caption_evade`(知道 strip 的攻擊):10 句開脫句(拐彎抹角 → 直白),逐句用 strip 的檢查器測,挑沒被抓且讓不防禦答 No 的;attack_info 存每句是否被抓 / 是否騙過。指令見 README 4.5d。
