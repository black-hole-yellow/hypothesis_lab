import os
import glob
import json
import re

# 1. Список "эталонных" ключей из вашего feature_registry.py
# (Я составил его на основе нашей архитектуры)
VALID_KEYS = [
    "Judas_Short", "Judas_Long", "NY_Sweep_Short", "NY_Sweep_Long",
    "Fix_Fade_Short", "Fix_Fade_Long", "NY_Cont_Short", "NY_Cont_Long",
    "Asian_Sweep_Long", "Asian_Sweep_Short", "NY_Expansion_Long", "NY_Expansion_Short",
    "NY_SR_Touch", "Asian_SR_Alignment", "LO_Counter_Fractal", "LO_PDH_PDL_Sweep",
    "Weekend_Gap", "Friday_Reversal", "Monday_Gap_Reversion", "Tuesday_Resumption",
    "Wed_Fakeout", "Thursday_Expansion", "LO_True_Trend", "Asian_Box_Breakout",
    "Tokyo_Trap", "Algo_Vol_Crush", "Geo_Shock", "Election_Vol", "UK_Pol_Shock",
    "BoE_Hawkish", "UK_CPI_Momentum", "Sovereign_Risk", "BoE_Tone_Shift",
    "Macro_Inside_Bar", "NFP_Divergence", "NFP_Revision_Trap", "CPI_Match_Reversion",
    "CB_Divergence", "FOMC_Sell_News", "UK_US_CPI_Div", "Unemp_Fakeout",
    "Retail_Sales_Div", "Multi_TF_FVG", "Prev_Boundaries", "Weekly_Swing",
    "1W_Swing", "1D_Swing", "HTF_Trend_Prob", "FVG_Order_Flow", "FVG_SR_Confluence",
    "Weekly_Floor", "1W_Level_Rejection", "Asia_FVG_Protection"
]

# Создаем маппинг: нижний_регистр -> Правильный_Регистр
KEY_MAP = {k.lower(): k for k in VALID_KEYS}

def fix_json_content(data):
    """Рекурсивно ищет и исправляет названия фичей в JSON структуре."""
    changed = False

    # 1. Исправляем required_features (список строк)
    if "data_dependencies" in data and "required_features" in data["data_dependencies"]:
        features = data["data_dependencies"]["required_features"]
        new_features = []
        for f in features:
            if f.lower() in KEY_MAP and f != KEY_MAP[f.lower()]:
                new_features.append(KEY_MAP[f.lower()])
                changed = True
            else:
                new_features.append(f)
        data["data_dependencies"]["required_features"] = new_features

    # 2. Исправляем условия (filters, triggers)
    def fix_conditions(condition_list):
        nonlocal changed
        if not isinstance(condition_list, list): return
        for cond in condition_list:
            if isinstance(cond, dict) and "feature" in cond:
                feat = cond["feature"]
                if feat.lower() in KEY_MAP and feat != KEY_MAP[feat.lower()]:
                    cond["feature"] = KEY_MAP[feat.lower()]
                    changed = True

    if "logic" in data:
        logic = data["logic"]
        if "filters" in logic: fix_conditions(logic["filters"])
        if "entry_rules" in logic:
            fix_conditions(logic["entry_rules"].get("long_trigger", []))
            fix_conditions(logic["entry_rules"].get("short_trigger", []))
            
    return data, changed

def run_repair():
    json_files = glob.glob('configs/**/*.json', recursive=True)
    repair_count = 0

    print(f"🔍 Начинаю проверку {len(json_files)} файлов...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

            updated_content, is_changed = fix_json_content(content)

            if is_changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_content, f, indent=2, ensure_ascii=False)
                print(f"🛠 Исправлен регистр в: {file_path}")
                repair_count += 1

        except Exception as e:
            print(f"❌ Ошибка в {file_path}: {e}")

    print(f"\n✅ Проверка завершена. Исправлено файлов: {repair_count}")

if __name__ == "__main__":
    run_repair()