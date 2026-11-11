# Final tables (config_name final)

How to read:
- Numbers in [ ] are 95% Wilson confidence intervals.
- p = McNemar exact test, Dual-Channel (ours) vs that method, paired on the same memes; (b vs c) = memes only ours gets right vs memes only the other method gets right.
- Attack success = share of the attacked hateful memes that the attack flips to "No" when there is no defense.
- Dual-Channel-High can equal Resize-and-Rephrase on clean memes: on clean memes the excuse detector removes nothing, so its Channel 1 sees the same input as Resize-and-Rephrase.
- A method missing from a section was not run for that dataset / model.

## FB

### A. Clean memes (no attack): 500 memes (250 hateful / 250 benign)

| Method | Accuracy | Hateful caught | Benign wrongly flagged | p (acc, vs ours) |
|---|---|---|---|---|
| No defense | 59.2% [55–63] | 59/250 (23.6%) | 13/250 (5.2%) | 0.014 |
| JPEG | 58.8% [54–63] | 60/250 (24.0%) | 16/250 (6.4%) | 0.008 |
| Resize-and-Rephrase (old) | 64.4% [60–68] | 125/250 (50.0%) | 53/250 (21.2%) | 0.314 |
| F3 | 59.0% [55–63] | 59/250 (23.6%) | 14/250 (5.6%) | 0.018 |
| Dual-Channel (ours) | 62.6% [58–67] | 89/250 (35.6%) | 26/250 (10.4%) | - |
| Dual-Channel-High (ours) | 64.4% [60–68] | 125/250 (50.0%) | 53/250 (21.2%) | 0.314 |

### B. Detection rate under attack: 59 hateful memes

59 = hateful memes the model correctly detects before any attack. Cell = share still detected after the attack (higher is better).

| Attack | Attack success | No defense | JPEG | Resize-and-Rephrase (old) | F3 | Dual-Channel (ours) | Dual-Channel-High (ours) | p vs none | p vs old | p vs F3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Persuade (caption) | 91.5% | 8.5% | 6.8% | 15.3% | 6.8% | 100.0% [94–100] | 100.0% | <0.001 (54 vs 0) | <0.001 (50 vs 0) | <0.001 (55 vs 0) |
| Blend (caption) | 91.5% | 8.5% | 5.1% | 44.1% | 8.5% | 98.3% [91–100] | 98.3% | <0.001 (53 vs 0) | <0.001 (32 vs 0) | <0.001 (53 vs 0) |
| Inject (caption) | 98.3% | 1.7% | 1.7% | 1.7% | 1.7% | 96.6% [88–99] | 100.0% | <0.001 (56 vs 0) | <0.001 (56 vs 0) | <0.001 (56 vs 0) |
| Evade (caption, adaptive) | 88.1% | 11.9% | 30.5% | 89.8% | 35.6% | 96.6% [88–99] | 98.3% | <0.001 (50 vs 0) | 0.219 (5 vs 1) | <0.001 (36 vs 0) |
| Text-in-image | 59.3% | 40.7% | 47.5% | 69.5% | 45.8% | 59.3% [47–71] | 71.2% | 0.013 (14 vs 3) | 0.070 (1 vs 7) | 0.115 (14 vs 6) |
| Text-in-image + Persuade | 96.6% | 3.4% | 1.7% | 8.5% | 3.4% | 59.3% [47–71] | 71.2% | <0.001 (34 vs 1) | <0.001 (30 vs 0) | <0.001 (34 vs 1) |
| PGD | 66.1% | 33.9% | 69.5% | 96.6% | 72.9% | 84.7% [73–92] | 96.6% | <0.001 (30 vs 0) | 0.016 (0 vs 7) | 0.065 (9 vs 2) |
| PGD-JPEG | 55.9% | 44.1% | 45.8% | 94.9% | 78.0% | 81.4% [70–89] | 94.9% | <0.001 (23 vs 1) | 0.008 (0 vs 8) | 0.754 (6 vs 4) |
| PGD + Persuade | 98.3% | 1.7% | 5.1% | 5.1% | 6.8% | 89.8% [80–95] | 96.6% | <0.001 (52 vs 0) | <0.001 (50 vs 0) | <0.001 (50 vs 1) |
| Salt & pepper | 11.9% | 88.1% | 86.4% | 100.0% | 71.2% | 93.2% [84–97] | 98.3% | 0.508 (6 vs 3) | 0.125 (0 vs 4) | 0.004 (16 vs 3) |
| Spread | 28.8% | 71.2% | 78.0% | 93.2% | 64.4% | 88.1% [77–94] | 94.9% | 0.006 (11 vs 1) | 0.375 (1 vs 4) | <0.001 (15 vs 1) |
| PGD-EOT (adaptive) | 52.5% | 47.5% | 69.5% | 74.6% | 74.6% | 55.9% [43–68] | 72.9% | 0.227 (8 vs 3) | <0.001 (0 vs 11) | 0.013 (3 vs 14) |

### C. Overall accuracy when every meme is attacked (500 memes)

| Attack | No defense | JPEG | Resize-and-Rephrase (old) | Dual-Channel (ours) | Dual-Channel-High (ours) |
|---|---|---|---|---|---|
| Persuade (caption) | 51.0% | 50.8% | 51.4% | 62.6% | 64.4% |
| Evade (caption, adaptive) | 51.0% | 54.2% | 62.4% | 62.2% | 63.4% |
| Text-in-image | 53.8% | 55.0% | 56.6% | 55.6% | 56.8% |

## FB (qwen7b)

### A. Clean memes (no attack): 200 memes (93 hateful / 107 benign)

| Method | Accuracy | Hateful caught | Benign wrongly flagged | p (acc, vs ours) |
|---|---|---|---|---|
| No defense | 76.5% [70–82] | 59/93 (63.4%) | 13/107 (12.1%) | 0.824 |
| Resize-and-Rephrase (old) | 72.0% [65–78] | 69/93 (74.2%) | 32/107 (29.9%) | 0.039 |
| Dual-Channel (ours) | 75.5% [69–81] | 68/93 (73.1%) | 24/107 (22.4%) |
| Dual-Channel-High (ours) | 72.0% [65–78] | 69/93 (74.2%) | 32/107 (29.9%) | 0.039 |

### B. Detection rate under attack: 50 hateful memes

50 = hateful memes the model correctly detects before any attack. Cell = share still detected after the attack (higher is better).

| Attack | Attack success | No defense | Resize-and-Rephrase (old) | Dual-Channel (ours) | Dual-Channel-High (ours) | p vs none | p vs old |
|---|---|---|---|---|---|---|---|
| Persuade (caption) | 62.0% | 38.0% | 76.0% | 100.0% [93–100] | 100.0% | <0.001 (31 vs 0) | <0.001 (12 vs 0) |
| Evade (caption, adaptive) | 26.0% | 74.0% | 100.0% | 100.0% [93–100] | 100.0% | <0.001 (13 vs 0) | 1.000 (0 vs 0) |
| Text-in-image + Persuade | 88.0% | 12.0% | 72.0% | 80.0% [67–89] | 84.0% | <0.001 (34 vs 0) | 0.219 (5 vs 1) |

## HarMeme

### A. Clean memes (no attack): 531 memes (185 hateful / 346 benign)

| Method | Accuracy | Hateful caught | Benign wrongly flagged | p (acc, vs ours) |
|---|---|---|---|---|
| No defense | 67.4% [63–71] | 23/185 (12.4%) | 11/346 (3.2%) | 0.093 |
| JPEG | 68.2% [64–72] | 27/185 (14.6%) | 11/346 (3.2%) | 0.405 |
| Resize-and-Rephrase (old) | 67.8% [64–72] | 32/185 (17.3%) | 18/346 (5.2%) | 0.118 |
| F3 | 68.4% [64–72] | 34/185 (18.4%) | 17/346 (4.9%) | 0.572 |
| Dual-Channel (ours) | 69.1% [65–73] | 38/185 (20.5%) | 17/346 (4.9%) | - |
| Dual-Channel-High (ours) | 68.0% [64–72] | 40/185 (21.6%) | 25/346 (7.2%) | 0.109 |

### B. Detection rate under attack: 23 hateful memes

23 = hateful memes the model correctly detects before any attack. Cell = share still detected after the attack (higher is better).

| Attack | Attack success | No defense | JPEG | Resize-and-Rephrase (old) | F3 | Dual-Channel (ours) | Dual-Channel-High (ours) | p vs none | p vs old | p vs F3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Persuade (caption) | 100.0% | 0.0% | 0.0% | 8.7% | 13.0% | 95.7% [79–99] | 100.0% | <0.001 (22 vs 0) | <0.001 (20 vs 0) | <0.001 (19 vs 0) |
| Blend (caption) | 100.0% | 0.0% | 0.0% | 8.7% | 13.0% | 91.3% [73–98] | 95.7% | <0.001 (21 vs 0) | <0.001 (19 vs 0) | <0.001 (18 vs 0) |
| Inject (caption) | 100.0% | 0.0% | 0.0% | 4.3% | 8.7% | 91.3% [73–98] | 95.7% | <0.001 (21 vs 0) | <0.001 (20 vs 0) | <0.001 (19 vs 0) |
| Evade (caption, adaptive) | 95.7% | 4.3% | 43.5% | 52.2% | 43.5% | 87.0% [68–95] | 95.7% | <0.001 (19 vs 0) | 0.021 (9 vs 1) | 0.002 (10 vs 0) |
| Text-in-image | 82.6% | 17.4% | 17.4% | 30.4% | 30.4% | 39.1% [22–59] | 56.5% | 0.062 (5 vs 0) | 0.500 (2 vs 0) | 0.625 (3 vs 1) |
| PGD | 73.9% | 26.1% | 73.9% | 91.3% | 95.7% | 82.6% [63–93] | 95.7% | <0.001 (13 vs 0) | 0.625 (1 vs 3) | 0.375 (1 vs 4) |
| PGD-JPEG | 43.5% | 56.5% | 30.4% | 91.3% | 87.0% | 78.3% [58–90] | 91.3% | 0.180 (7 vs 2) | 0.250 (0 vs 3) | 0.688 (2 vs 4) |
| PGD + Persuade | 100.0% | 0.0% | 0.0% | 8.7% | 4.3% | 78.3% [58–90] | 91.3% | <0.001 (18 vs 0) | <0.001 (16 vs 0) | <0.001 (17 vs 0) |
| Text-in-image + Persuade | 100.0% | 0.0% | 0.0% | 4.3% | 0.0% | 34.8% [19–55] | 60.9% | 0.008 (8 vs 0) | 0.016 (7 vs 0) | 0.008 (8 vs 0) |
| PGD-EOT (adaptive) | 39.1% | 60.9% | 82.6% | 60.9% | 100.0% | 52.2% [33–71] | 69.6% | 0.688 (2 vs 4) | 0.500 (0 vs 2) | <0.001 (0 vs 11) |

