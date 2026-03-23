import os
import json
import shutil
import glob
import sys
from pathlib import Path
# Force Python to recognize the project root
sys.path.append(str(Path(__file__).resolve().parent))

from src.core.engine import LabEngine
from src.hypotheses.generic_json_hypothesis import GenericJSONHypothesis
from src.core.evaluator import SignalEvaluator

# --- DIRECTORY SETUP ---
PENDING_DIR = "configs/pending_hypotheses"
PRODUCTION_DIR = "configs/production"
REVIEW_DIR = "configs/review"

for directory in [PENDING_DIR, PRODUCTION_DIR, REVIEW_DIR]:
    os.makedirs(directory, exist_ok=True)

def extract_required_features(config: dict) -> list:
    """Scans the strict schema JSON to find exactly which features to calculate."""
    features = set()
    logic = config.get("logic", {})
    
    # Scan filters
    for f in logic.get("filters", []):
        if "feature" in f: features.add(f["feature"])
        
    # Scan triggers
    entry_rules = logic.get("entry_rules", {})
    for rule in entry_rules.get("long_trigger", []) + entry_rules.get("short_trigger", []):
        if "feature" in rule: features.add(rule["feature"])
        
    return list(features)

def process_pending_hypotheses():
    print("=========================================")
    print("      AUTOMATED QUANT PIPELINE v2.0      ")
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

        # 1. Setup Data Paths
        universe = config.get("universe", {})
        timeframe = universe.get("resolution", "1h")
        symbol = universe.get("instruments", ["GBPUSD"])[0]
        processed_data_path = f"data/processed/{symbol}_{timeframe}.parquet"

        if not os.path.exists(processed_data_path):
            print(f"❌ Missing data: {processed_data_path}. Moving to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            # Fail fast
            break

        # 2. Initialize Engine
        engine = LabEngine(
            data_file=processed_data_path,
            start_date="2018-01-01",
            end_date="2026-02-27",
            timeframe=timeframe
        )

        # 3. Dynamic Feature Loading
        required_features = extract_required_features(config)
        if not engine.prepare_data(required_features):
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            break

        # 4. Run Strategy
        hypothesis = GenericJSONHypothesis(config=config)
        engine.run_hypothesis(hypothesis)

        # 5. Evaluate and Log
        evaluator = SignalEvaluator(hypothesis)
        metrics = evaluator.calculate_metrics()

        # 6. Tear Sheet
        print("=========================================")
        print(f"  TEAR SHEET: {metrics.get('Hypothesis', filename)}")
        print("=========================================")
        print(f"  Frequency   : {metrics.get('Frequency', 0)}")
        print(f"  Win Rate    : {metrics.get('Win_Rate_%', 0)}% ({metrics.get('Wins', 0)}W / {metrics.get('Losses', 0)}L)")
        print(f"  T-Stat      : {metrics.get('T_Stat', 0.0)}")
        print(f"  Status      : {metrics.get('Status', 'ERROR')}")
        print("=========================================")

        # 7. Routing
        if metrics.get('Status') == 'PASSED':
            print("✅ PASSED: Promoted to PRODUCTION.\n")
            shutil.move(file_path, os.path.join(PRODUCTION_DIR, filename))
        else:
            print("❌ FAILED: Moved to REVIEW.\n")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))

if __name__ == "__main__":
    process_pending_hypotheses()