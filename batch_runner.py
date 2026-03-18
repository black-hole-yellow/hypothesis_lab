import os
import json
import shutil
import glob
from src.core.engine import LabEngine
from src.hypotheses.generic_json_hypothesis import GenericJSONHypothesis
from src.core.evaluator import SignalEvaluator

# --- DIRECTORY SETUP ---
PENDING_DIR = "configs/pending_hypotheses"
PRODUCTION_DIR = "configs/production"
REVIEW_DIR = "configs/review"  # <--- Changed from Archive to Review

for directory in [PENDING_DIR, PRODUCTION_DIR, REVIEW_DIR]:
    os.makedirs(directory, exist_ok=True)

def process_pending_hypotheses():
    print("=========================================")
    print("      AUTOMATED QUANT PIPELINE v1.1      ")
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
            print(f"❌ Missing processed data: {processed_data_path}. Run DataPolisher first!")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        engine = LabEngine(
            data_file=processed_data_path,
            start_date="2015-01-01",
            end_date="2026-02-27",
            timeframe=timeframe
        )


        hypothesis = GenericJSONHypothesis(config=config)
        engine.run_hypothesis(hypothesis)

        if len(hypothesis.triggers) == 0:
            print(f"⚠️ No trades triggered. Moving {filename} to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        evaluator = SignalEvaluator(engine.df, hypothesis.triggers, hypothesis.name)
        metrics = evaluator.calculate_metrics()
        
        t_stat = metrics.get('T_Stat', 0.0)
        win_rate = metrics.get('Win_Rate', 0.0)

        # THE PRODUCTION HANDOFF DECISION
        if t_stat >= 2.0:
            print(f"✅ PASSED! T-Stat: {t_stat:.2f} | Win Rate: {win_rate:.2f}%")
            print(f"🚀 Promoting {filename} to PRODUCTION.")
            shutil.move(file_path, os.path.join(PRODUCTION_DIR, filename))
        else:
            print(f"❌ FAILED. T-Stat: {t_stat:.2f} (Required: 2.0).")
            print(f"🔍 Moving {filename} to REVIEW for manual analysis.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            
        print("-" * 40)

if __name__ == "__main__":
    process_pending_hypotheses()