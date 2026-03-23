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
            start_date="2020-01-01",
            end_date="2026-02-27",
            timeframe=timeframe
        )

        # Load the data into memory
        if not engine.prepare_data():
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        hypothesis = GenericJSONHypothesis(config=config)
        engine.run_hypothesis(hypothesis)

        if len(hypothesis.triggers) == 0:
            print("⚠️ No trades triggered. Moving to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        target_metric = config.get("logic", {}).get("evaluation_metric", "Close")
        evaluator = SignalEvaluator(engine.df, hypothesis.triggers, hypothesis.name, target_col=target_metric)
        metrics = evaluator.calculate_metrics()
        
        if not metrics or 'Optimal_Hold_Hours' not in metrics:
            print(f"⚠️  Hypothesis '{hypothesis.name}' generated 0 trades.")
            print("   (Check your logic for 'Geometry Traps' or impossible conditions)")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))
            continue

        h = metrics['Optimal_Hold_Hours']
        win_rate = metrics.get(f'Hit_Ratio_{h}H', 0)
        wins = metrics.get('Best_Win_Count', 0)
        losses = metrics.get('Best_Loss_Count', 0)
        t_stat = metrics.get(f'T_Stat_{h}H', 0.0)

        # ==========================================================
        #  NEW: INJECT WIN/LOSS OUTCOMES INTO THE CSV LOG
        # ==========================================================
        fwd_col = f'Fwd_Ret_{h}'
        for i, trigger in enumerate(hypothesis.triggers):
            dt = trigger['Datetime']
            if dt in evaluator.df.index:
                ret = evaluator.df.at[dt, fwd_col]
                signal = evaluator.df.at[dt, 'Signal']
                
                if pd.isna(ret):
                    # If the trade triggered on the very last day of data, 
                    # there is no future data to check yet!
                    outcome = "Pending"
                else:
                    # If signal matches the direction of the return, it's a win!
                    outcome = "Win" if (signal * ret) > 0 else "Loss"
            else:
                outcome = "Unknown"
                
            # Remove old Status columns and add Outcome
            hypothesis.daily_logs[i].pop('Status', None)
            hypothesis.daily_logs[i].pop('Signal_Triggered', None)
            hypothesis.daily_logs[i]['Outcome'] = outcome

        # Save dynamic audit log named after the hypothesis
        os.makedirs("output", exist_ok=True)
        safe_name = hypothesis.name.replace('/', '_').replace('\\', '_')
        audit_filename = f"output/{safe_name}_audit_log.csv"
        pd.DataFrame(hypothesis.daily_logs).to_csv(audit_filename, index=False)
        # ==========================================================

        print("=========================================")
        print(f"  TEAR SHEET: {metrics['Hypothesis']}")
        print("=========================================")
        print(f"  Frequency          : {metrics['Frequency']}")
        print(f"  Win Rate (1:2 RR)  : {metrics.get('Win_Rate_%', 0)}% ({metrics.get('Wins', 0)}W / {metrics.get('Losses', 0)}L)")
        print(f"  Expectancy (R)     : {metrics.get('Expectancy_R', 0)}")
        print(f"  T-Stat             : {metrics.get('T_Stat', 0)}")
        print("=========================================")
        print(f"Status: {metrics['Status']}")

        # Умная логика промоушена:
        freq = metrics.get('Frequency', 0)
        
        # 1. Стандартное правило больших чисел (T-Stat >= 2.0)
        statistically_significant = (t_stat >= 2.0)
        
        # 2. Исключение для Черных Лебедей (Макро-шоки, мало сделок, но высокий винрейт)
        macro_exemption = (freq > 0) and (freq < 15) and (win_rate >= 70.0)

        if statistically_significant or macro_exemption:
            if macro_exemption:
                print("✅ PASSED (MACRO EXEMPTION): Promoted to PRODUCTION.")
            else:
                print("✅ PASSED (STATISTICAL EDGE): Promoted to PRODUCTION.")
            shutil.move(file_path, os.path.join(PRODUCTION_DIR, filename))
        else:
            print("❌ FAILED: Moved to REVIEW.")
            shutil.move(file_path, os.path.join(REVIEW_DIR, filename))

if __name__ == "__main__":
    process_pending_hypotheses()