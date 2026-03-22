import os
import glob
import json

# --- ЭТАЛОННЫЕ КЛЮЧИ ИЗ FEATURE_REGISTRY ---
VALID_KEYS = [
    # SESSION / TIME
    "Judas_Short", "Judas_Long", "Fix_Fade_Short", "Fix_Fade_Long",
    "NY_Sweep_Short", "NY_Sweep_Long", "NY_Cont_Short", "NY_Cont_Long",
    "Asian_Box_Long", "Asian_Box_Short", "Tokyo_Trap_Long", "Tokyo_Trap_Short",
    "Friday_Reversal_Long", "Friday_Reversal_Short", "Monday_Reversion_Long", "Monday_Reversion_Short",
    "Tuesday_Resumption_Long", "Tuesday_Resumption_Short", "Wed_Fakeout_Long", "Wed_Fakeout_Short",
    "Thursday_Trend_Long", "Thursday_Trend_Short", "LO_True_Trend_Long", "LO_True_Trend_Short",
    "Algo_Vol_Crush_Long", "Algo_Vol_Crush_Short", "NY_Opened_In_Asia_Range",
    "NY_Sweep_Asia_High", "NY_Sweep_Asia_Low", "First_LDN_PDL_Long", "First_LDN_PDH_Short",
    "LDN_Protected_AL_Long", "LDN_Protected_AH_Short",

    # MACRO / NEWS
    "Geo_Shock_Long", "Geo_Shock_Short", "Election_Vol_Crush_Long", "Election_Vol_Crush_Short",
    "UK_Shock_Cont_Long", "UK_Shock_Cont_Short", "CPI_Momentum_Long", "CPI_Momentum_Short",
    "Macro_CPI_Div_Long", "Macro_CPI_Div_Short", "NFP_Fade_Long", "NFP_Fade_Short",
    "NFP_Resumption_Long", "NFP_Resumption_Short", "CB_Divergence_Long", "CB_Divergence_Short",
    "BoE_Tone_Shift_Long", "BoE_Tone_Shift_Short", "FOMC_Sell_News_Long", "FOMC_Sell_News_Short",
    "Retail_Div_Long", "Retail_Div_Short", "Unemp_Fakeout_Long", "Unemp_Fakeout_Short",
    "Sovereign_Risk_Long", "Sovereign_Risk_Short",

    # STRUCTURE / PA
    "Multi_TF_FVG", "HTF_Trend_Prob", "First_1W_Rej_Long", "First_1W_Rej_Short",
    "Swept_AL_Into_FVG", "Swept_AH_Into_FVG", "1h_Bullish_Flip", "1h_Bearish_Flip",
    "Prev_Boundaries", "Weekly_Swing", "1W_Swing", "1D_Swing", "FVG_Order_Flow", 
    "FVG_SR_Confluence", "Weekly_Floor", "Asia_FVG_Protection",

    # META
    "UA_Hour", "Optimal_Exit_Price"
]

# Создаем карту: "нижний_регистр" -> "Правильный_Регистр"
KEY_MAP = {k.lower(): k for k in VALID_KEYS}

def fix_json_structure(data):
    """Рекурсивно исправляет названия фичей во всех секциях JSON."""
    changed = False

    # 1. Секция required_features
    if "data_dependencies" in data and "required_features" in data["data_dependencies"]:
        new_features = []
        for feat in data["data_dependencies"]["required_features"]:
            if feat.lower() in KEY_MAP:
                correct_feat = KEY_MAP[feat.lower()]
                if feat != correct_feat:
                    changed = True
                new_features.append(correct_feat)
            else:
                new_features.append(feat)
        data["data_dependencies"]["required_features"] = new_features

    # 2. Секция logic (filters, triggers)
    def fix_feature_in_conditions(condition_list):
        nonlocal changed
        if not isinstance(condition_list, list): return
        for cond in condition_list:
            if isinstance(cond, dict) and "feature" in cond:
                feat = cond["feature"]
                if feat.lower() in KEY_MAP:
                    correct_feat = KEY_MAP[feat.lower()]
                    if feat != correct_feat:
                        cond["feature"] = correct_feat
                        changed = True

    if "logic" in data:
        logic = data["logic"]
        if "filters" in logic: 
            fix_feature_in_conditions(logic["filters"])
        if "entry_rules" in logic:
            fix_feature_in_conditions(logic["entry_rules"].get("long_trigger", []))
            fix_feature_in_conditions(logic["entry_rules"].get("short_trigger", []))

    return data, changed

def main():
    # Ищем файлы во всех папках конфигов
    json_files = glob.glob('configs/**/*.json', recursive=True)
    repair_count = 0

    print(f"🛠  Start checking {len(json_files)} files for casing issues...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            updated_content, is_changed = fix_json_structure(content)

            if is_changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_content, f, indent=2, ensure_ascii=False)
                print(f"✅ Fixed casing in: {file_path}")
                repair_count += 1

        except Exception as e:
            print(f"❌ Error in {file_path}: {e}")

    print(f"\n✨ Done! Corrected {repair_count} files.")

if __name__ == "__main__":
    main()