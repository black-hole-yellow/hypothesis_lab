import glob
import json

def standardize_configs():
    json_files = glob.glob('configs/**/*.json', recursive=True)
    count = 0

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            changed = False
            logic = data.get("logic", {})

            # 1. Удаляем рудименты
            if "evaluation_metric" in logic:
                del logic["evaluation_metric"]
                changed = True
            if "exit_rules" in logic:
                del logic["exit_rules"]  # Удаляем старое текстовое описание
                changed = True

            # 2. Формируем эталонный execution_rules
            if "execution_rules" not in logic or "mode" not in logic["execution_rules"]:
                old_rules = logic.get("execution_rules", {})
                
                logic["execution_rules"] = {
                    "mode": "risk_reward", # По умолчанию все старые стратегии - это RR
                    "risk_reward_ratio": old_rules.get("risk_reward_ratio", 2.0),
                    "sl_atr_multiplier": old_rules.get("default_sl_atr_multiplier", 1.0),
                    "max_hold_bars": None,
                    "max_trades_per_day": old_rules.get("max_trades_per_day", 1),
                    "allow_resweep": old_rules.get("allow_resweep", False)
                }
                changed = True

            data["logic"] = logic

            if changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                count += 1
                print(f"✅ Standardized: {file_path}")

        except Exception as e:
            print(f"❌ Error in {file_path}: {e}")

    print(f"\n🚀 Готово! Приведено к стандарту файлов: {count}")

if __name__ == "__main__":
    standardize_configs()