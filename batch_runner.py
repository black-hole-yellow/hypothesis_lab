import os
import json
import shutil
import glob
import pandas as pd
from sklearn import metrics 

from src.core.engine import LabEngine
from src.hypotheses.generic_json_hypothesis import GenericJSONHypothesis
from src.core.evaluator import SignalEvaluator

# --- DIRECTORY SETUP ---
PENDING_DIR = "configs/pending_hypotheses"
PRODUCTION_DIR = "configs/production"
REVIEW_DIR = "configs/review"

for directory in [PENDING_DIR, PRODUCTION_DIR, REVIEW_DIR]:
    os.makedirs(directory, exist_ok=True)

def process_pending_hypotheses():
    print("=========================================")
    print("      AUTOMATED QUANT PIPELINE v1.2      ")
    print("=========================================")
    
    pending_files = glob.glob(os.path.join(PENDING_DIR, "*.json"))
    
    if not pending_files:
        print("⏸️ No new hypotheses found in pending folder. Waiting...")
        return

    print(f"📂 Found {len(pending_files)} new hypotheses to test.\n")

    for file_path in pending_files:
        filename = os.path.basename(file_path)
        print(f"🧪 Testing: {filename}...")

        with open(file_path, "r") as f:
            config = json.load(f)

        timeframe = config.get("universe", {}).get("resolution", "1h")
        instruments = config.get("universe", {}).get("instruments", ["GBPUSD"])
        symbol = instruments[0] if instruments else "GBPUSD"
        processed_data_path = f"data/processed/{symbol}_{timeframe}.parquet"

        if not os.path.exists(processed_data_path):
            print(f"❌ Missing processed data: {processed_data_path}")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        engine = LabEngine(
            data_file=processed_data_path,
            start_date="2025-01-01",
            end_date="2026-02-27",
            timeframe=timeframe
        )

        # Load the data into memory
        if not engine.prepare_data():
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        hypothesis = GenericJSONHypothesis(config=config)

        required_features = config.get("data_dependencies", {}).get("required_features", [])
        engine.apply_custom_features(required_features)

        engine.run_hypothesis(hypothesis)

        if len(hypothesis.triggers) == 0:
            print("⚠️ No trades triggered. Moving to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        target_metric = config.get("logic", {}).get("evaluation_metric", "Close")
        evaluator = SignalEvaluator(engine.df, hypothesis.triggers, hypothesis.name, target_col=target_metric)
        metrics = evaluator.calculate_metrics()
        
        if not metrics or metrics.get('Status') == 'FAILED (Not enough data)':
            print(f"⚠️  Hypothesis '{hypothesis.name}' generated 0 trades or insufficient data.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        # ==========================================================
        #  ОБНОВЛЕННЫЙ АУДИТ: Берем данные напрямую из Trade Management
        # ==========================================================
        for i, trigger in enumerate(hypothesis.triggers):
            outcome = trigger.get('Outcome', 'Pending')
            
            # Синхронизируем лог с результатами симуляции
            hypothesis.daily_logs[i]['Outcome'] = outcome
            hypothesis.daily_logs[i]['Entry'] = round(trigger.get('Entry_Price', 0), 5)
            hypothesis.daily_logs[i]['SL'] = round(trigger.get('SL_Price', 0), 5)
            hypothesis.daily_logs[i]['TP'] = round(trigger.get('TP_Price', 0), 5)

        # Сохраняем аудит
        os.makedirs("output", exist_ok=True)
        safe_name = hypothesis.name.replace('/', '_').replace('\\', '_')
        pd.DataFrame(hypothesis.daily_logs).to_csv(f"output/{safe_name}_audit_log.csv", index=False)
        # ==========================================================

        print("=========================================")
        print(f"  TEAR SHEET: {metrics['Hypothesis']}")
        print("=========================================")
        print(f"  Frequency          : {metrics['Frequency']}")
        print(f"  Win Rate (1:2 RR)  : {metrics.get('Win_Rate_%', 0)}% ({metrics.get('Wins', 0)}W / {metrics.get('Losses', 0)}L)")
        print(f"  Expectancy (R)     : {metrics.get('Expectancy_R', 0)}")
        print(f"  T-Stat             : {metrics.get('T_Stat', 0)}")
        print("=========================================")
        
        if metrics['Status'] == 'PASSED':
            print("✅ PASSED: Promoted to PRODUCTION.")
            shutil.move(file_path, os.path.join(PRODUCTION_DIR, filename))
        else:
            print("❌ FAILED: Moved to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))

if __name__ == "__main__":
    process_pending_hypotheses()