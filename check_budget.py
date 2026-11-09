configs = {
    "Config 1 (Balanced)": {
        "q": [32, 32, 33, 33],
        "v_sym": [34, 34, 33, 34],
        "v_anti": [32, 32, 31, 32],
    },
    "Config 2 (Wide)": {
        "q": [48, 48, 49, 49],
        "v_sym": [26, 26, 26, 25],
        "v_anti": [22, 21, 21, 21],
    },
    "Config 3 (Very Wide)": {
        "q": [64, 64, 65, 65],
        "v_sym": [18, 18, 18, 17],
        "v_anti": [10, 10, 10, 9],
    },
    "Config 4 (Aggressive)": {
        "q": [68, 72, 72, 76],
        "v_sym": [14, 14, 14, 14],
        "v_anti": [4, 4, 4, 4],
    },
    "Config 5 (Extreme)": {
        "q": [71, 76, 77, 78],
        "v_sym": [13, 12, 12, 12],
        "v_anti": [2, 1, 1, 1],
    },
}

d_model = 128
non_attn = 553856
target_layer = 51200
target_total = 758656

print("Target layer attn params:", target_layer, "Target total model:", target_total)

for name, cfg in configs.items():
    q = cfg["q"]
    b_params = sum(d * (d - 1) // 2 for d in q)
    wq_params = d_model * sum(q)
    
    # Symmetric
    wv_wo_sym = 2 * d_model * sum(cfg["v_sym"])
    attn_sym = wq_params + wv_wo_sym
    tot_sym = non_attn + 4 * attn_sym
    
    # Antisymmetric
    wv_wo_anti = 2 * d_model * sum(cfg["v_anti"])
    attn_anti = wq_params + wv_wo_anti + b_params
    tot_anti = non_attn + 4 * attn_anti
    
    print(f"{name}:")
    print(f"  sum(q)={sum(q)}, B_params/layer={b_params}")
    print(f"  Symmetric: sum(v)={sum(cfg['v_sym'])}, layer_attn={attn_sym}, total_model={tot_sym}")
    print(f"  Antisymmetric: sum(v)={sum(cfg['v_anti'])}, layer_attn={attn_anti}, total_model={tot_anti}")
    assert attn_sym == target_layer, f"Sym layer attn mismatch in {name}: {attn_sym}"
    assert attn_anti == target_layer, f"Anti layer attn mismatch in {name}: {attn_anti}"
    assert tot_sym == target_total, f"Sym total mismatch in {name}: {tot_sym}"
    assert tot_anti == target_total, f"Anti total mismatch in {name}: {tot_anti}"

print("\nALL 10 CONFIGURATIONS EXACTLY MATCH 51,200 PARAMS/LAYER AND 758,656 TOTAL MODEL PARAMS!")
