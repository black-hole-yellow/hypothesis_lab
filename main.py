import os
import pandas as pd

# 1. Pipeline Imports
from src.utils.data_loader import load_and_prep_data
from src.core.evaluator import SignalEvaluator, save_hypothesis_results

# 2. Feature (DNA) Imports
# (Make sure all these exist in your src/library/features.py or htf_features.py)
from src.library.features import (
    add_log_returns, 
    add_atr, 
    add_normalized_slope, 
    add_price_zscore, 
    add_shannon_entropy, 
    add_hurst_exponent, 
    add_hmm_volatility_regime, 
    add_volatility_ratio
    # add_session_tags,      # Uncomment if you have this
    # add_williams_fractals  # Uncomment if you have this
)

# 3. Hypothesis Import
from src.hypotheses.london_fake_move import LondonFakeMove


def run_lab():
    print("=========================================")
    print("      QUANT HYPOTHESIS LAB v2.0          ")
    print("=========================================")

    # ---------------------------------------------------------
    # STEP 1: LOAD DATA
    # ---------------------------------------------------------
    data_file = "data/gbpusd_data.csv"
    start_date = "2015-01-01"
    end_date = "2026-02-27"
    timeframe = "1h"

    df = load_and_prep_data(data_file, start_date, end_date, timeframe)
    
    if df is None or df.empty:
        print("❌ Lab Terminated: No data available.")
        return

    # ---------------------------------------------------------
    # STEP 2: APPLY QUANTITATIVE DNA (Feature Engineering)
    # ---------------------------------------------------------
    print("\n[1/3] Calculating Market DNA & Quant Features...")
    
    # Base technicals
    # df = add_session_tags(df) 
    # df = add_williams_fractals(df, timeframe=timeframe, n=2)
    
    # Core Mathematical Quant Features
    df = add_log_returns(df)
    df = add_atr(df, lookback=14)
    df = add_volatility_ratio(df, short_lookback=14, long_lookback=100)
    df = add_normalized_slope(df, lookback=20, atr_lookback=14)
    df = add_price_zscore(df, lookback=50)
    df = add_shannon_entropy(df, lookback=50)
    
    print("      -> Calculating Hurst Exponent (Heavy computation)...")
    df = add_hurst_exponent(df, lookback=100)
    
    print("      -> Training HMM for Volatility Regimes...")
    df = add_hmm_volatility_regime(df)

    # Clean up NaNs created by rolling windows before running the hypothesis
    df.dropna(inplace=True)
    print(f"      -> DNA complete. Valid rows remaining: {len(df)}")


    # ---------------------------------------------------------
    # STEP 3: RUN HYPOTHESIS
    # ---------------------------------------------------------
    print("\n[2/3] Executing Hypothesis Engine...")
    
    hypothesis_name = "London_Fake_Move_V1"
    hypothesis = LondonFakeMove(name=hypothesis_name)

    # Standard execution loop (simulating ticking data)
    for index, row in df.iterrows():
        # Pass the row and current index to your hypothesis logic
        # It should append valid signals to 'hypothesis.triggers'
        hypothesis.evaluate_row(row, index)
        
    print(f"      -> Hypothesis finished. Raw triggers generated: {len(hypothesis.triggers)}")


    # ---------------------------------------------------------
    # STEP 4: SIGNAL EVALUATION & METRICS
    # ---------------------------------------------------------
    print("\n[3/3] Running Quantitative Signal Evaluation...")
    
    if len(hypothesis.triggers) == 0:
        print("❌ No triggers generated. Check your hypothesis logic.")
        return

    # Ensure your triggers inside LondonFakeMove are saved as dictionaries 
    # Example: self.triggers.append({'Datetime': index, 'Direction': 'Long'})
    evaluator = SignalEvaluator(df, hypothesis.triggers, hypothesis_name)
    metrics = evaluator.calculate_metrics()
    
    print("\n=========================================")
    print("        HYPOTHESIS TEAR SHEET            ")
    print("=========================================")
    for key, value in metrics.items():
        # Formatting to make it look clean in the terminal
        print(f"  {key:<20} : {value}")
    print("=========================================\n")
    
    # ---------------------------------------------------------
    # STEP 5: SAVE TO REGISTRY (Only if it passes edge tests)
    # ---------------------------------------------------------
    # The save_hypothesis_results function handles the logic of checking
    # if the status is "PASSED" (T-Stat > 2.0 and IC > 0.0)
    save_hypothesis_results(metrics, filepath="output/hypothesis_results.csv")


if __name__ == "__main__":
    # Create necessary output folders if they don't exist
    os.makedirs("output", exist_ok=True)
    run_lab()