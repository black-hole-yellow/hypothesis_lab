import os
import pandas as pd

from src.utils.data_loader import load_and_prep_data
from src.core.evaluator import SignalEvaluator, save_hypothesis_results
from src.library.features import (
    add_log_returns, add_atr, add_normalized_slope, add_price_zscore, 
    add_shannon_entropy, add_hurst_exponent, add_hmm_volatility_regime, 
    add_volatility_ratio, add_williams_fractals
)
from src.library.htf_features import add_htf_trend_probability

class LabEngine:
    def __init__(self, data_file: str, start_date: str, end_date: str, timeframe: str = "1h"):
        self.data_file = data_file
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None

    def prepare_data(self):
        print(f"[1/3] Loading data from {self.data_file}...")
        try:
            # 1. Load into a temporary local variable
            if self.data_file.endswith('.parquet'):
                temp_df = pd.read_parquet(self.data_file)
            else:
                temp_df = pd.read_csv(self.data_file)
                temp_df['Datetime'] = pd.to_datetime(temp_df['Datetime'])
                temp_df.set_index('Datetime', inplace=True)

            # 2. Handle Timezones securely before slicing
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            
            # If the data index has a timezone (like our Parquet), make start/end aware too!
            if temp_df.index.tz is not None:
                start = start.tz_localize(temp_df.index.tz)
                end = end.tz_localize(temp_df.index.tz)

            # 3. Filter dates securely
            temp_df = temp_df.loc[start:end].copy()

            # 4. Assign the clean, filtered data to the engine
            self.df = temp_df

            # --- FEATURE ENGINEERING ---
            # Fixed: Using self.df instead of the undefined 'df' variable
            self.df = add_log_returns(self.df)
            self.df = add_atr(self.df, lookback=14)
            self.df = add_volatility_ratio(self.df, short_lookback=14, long_lookback=100)
            self.df = add_normalized_slope(self.df, lookback=20, atr_lookback=14)
            self.df = add_price_zscore(self.df, lookback=50)
            self.df = add_shannon_entropy(self.df, lookback=50)
            self.df = add_williams_fractals(self.df, timeframe=self.timeframe, n=2)
            self.df = add_hurst_exponent(self.df, lookback=100)
            self.df = add_hmm_volatility_regime(self.df)
            self.df = add_htf_trend_probability(self.df, htf='4h', lookback=60) 
            
            self.df.dropna(inplace=True)
            print(f"      -> DNA complete. Valid rows remaining: {len(self.df)}")
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

        return True

    def run_hypothesis(self, hypothesis):
        print(f"\n[2/3] Executing Hypothesis Engine: {hypothesis.name}...")
        current_day = None

        for index, row in self.df.iterrows():
            day_date = index.date()
            if day_date != current_day:
                print(f"\r      -> Processing date: {day_date} ...", end="", flush=True)
                current_day = day_date

            hypothesis.evaluate_row(row, index)

        # Save dynamic audit log named after the hypothesis!
        os.makedirs("output", exist_ok=True)
        audit_filename = f"output/{hypothesis.name}_audit_log.csv"
        audit_df = pd.DataFrame(hypothesis.daily_logs)
        audit_df.to_csv(audit_filename, index=False)
        
        print(f"\n      -> Hypothesis finished. Raw triggers generated: {len(hypothesis.triggers)}")
        print(f"      -> Daily Audit Trail saved to '{audit_filename}'")

    def evaluate(self, hypothesis):
        import json # Make sure json is imported at the top of engine.py
        
        print("\n[3/3] Running Quantitative Signal Evaluation...")
        if len(hypothesis.triggers) == 0:
            print("❌ No triggers generated.")
            return

        evaluator = SignalEvaluator(self.df, hypothesis.triggers, hypothesis.name)
        metrics = evaluator.calculate_metrics()
        
        print("\n=========================================")
        print("        HYPOTHESIS TEAR SHEET            ")
        print("=========================================")
        for key, value in metrics.items():
            print(f"  {key:<20} : {value}")
        print("=========================================\n")
        
        

        # --- THE PRODUCTION HANDOFF AUTOMATION ---
        # Assuming your SignalEvaluator returns metrics like 'T_Stat' and 'Win_Rate'
        t_stat = metrics.get('T_Stat', 0)
        
        # Define your strict production thresholds here
        if t_stat >= 2.0:
            print("✅ STRATEGY VALIDATED! Generating Production Artifacts...")
            
            config_filename = f"configs/production/{hypothesis.name}.json"
            
            # Save the exact configuration that passed the test
            with open(config_filename, "w") as f:
                json.dump(hypothesis.config, f, indent=4)
                
            print(f"      -> Production JSON Config saved to: {config_filename}")
        else:
            print("⚠️ Strategy failed statistical validation (T-Stat < 2.0). No production config generated.")