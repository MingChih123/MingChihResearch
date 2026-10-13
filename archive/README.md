# archive/ — 已不再使用的舊檔案

這些檔案已被新程式取代,或只在早期實驗用過。保留在這裡只是為了紀錄,**新實驗請不要用**。
(放在 archive/ 裡的 .py 無法直接執行,因為它們要 import 上一層的 common.py 等模組。)

| 檔案 | 為什麼不用了 | 改用 |
|---|---|---|
| `pgd_attack.py` | 最早的獨立攻擊腳本,有自己一份讀錯 token(帶空格)的程式碼 | `run_robust_suite.py` |
| `eval_vqa.py` | 最早的乾淨評估腳本,和 `common.py` 重複 | `run_clean_eval.py` / `run_robust_suite.py --clean_only` |
| `testenv.py` | 安裝環境時測試 7B 模型能不能載入 | README 的「確認環境」步驟 |
| `verify_weighted.py` | 寫死舊檔名,比對加權投票(加權投票已不用) | 無 |
| `clean_control_check.py` | 只有多數決、樣本少的乾淨副作用檢查 | `run_robust_suite.py --clean_only` |
| `run_error_correction_experiment.py` | 只看「原本答錯」的樣本,不算副作用 | `run_clean_eval.py` |
| `make_slide_tables.py` | 單次簡報用的表格合併 | `summarize_all.py` |
| `PROGRESS_SUMMARY.md` | 2026-10 初的進度整理,數字已過時 | `CLAUDE.md`、`THESIS_OUTLINE.md` |
| `requirement.txt` | 舊的套件清單 | `requirements.txt` |
