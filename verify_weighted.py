import json

with open("./output/FB_Qwen2-VL-2B-Instruct_n10_seed0_eps0.1_a0.04_s3_noise0.3_ns5.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'image':20s} {'attacked':10s} {'combo(多數決)':14s} {'combo_weighted(加權)':20s} {'一樣嗎':6s}")
print("-" * 85)

same_count = 0
diff_count = 0

for r in data["results"]:
    if not r["attack_success"]:
        continue

    same = (r["combo_defense_pred"] == r["combo_weighted_pred"])
    same_count += same
    diff_count += (not same)

    print(f"{r['image']:20s} {r['attacked_pred']:10s} "
          f"{r['combo_defense_pred']:14s} {r['combo_weighted_pred']:20s} "
          f"{'是' if same else '不同'}")

print("-" * 85)
print(f"總共 {same_count + diff_count} 筆被攻擊成功的樣本")
print(f"兩種方法答案相同: {same_count} 筆")
print(f"兩種方法答案不同: {diff_count} 筆")