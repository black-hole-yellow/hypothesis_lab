import pandas as pd
import numpy as np
from src.utils.data_loader import load_and_prep_data
from src.library.htf_features import add_previous_boundaries, calculate_multi_tf_fvgs, find_major_sr, get_confirmed_swings
from src.library.features import add_williams_fractals

def run_htf_test():
    file_path = "data/gbpusd_data.csv"
    
    # We use a broad range to ensure Weekly S&Rs have time to build
    start_date = '2020-01-01'
    end_date = '2026-02-27'
    tf = '1h'
    
    print(f"Loading data...")
    df = load_and_prep_data(file_path, start_date, end_date, tf)

    print("1. Calculating Boundaries & Multi-TF FVGs...")
    df = add_previous_boundaries(df)
    df = calculate_multi_tf_fvgs(df)

    print("2. Calculating Weekly Swings & S&R...")
    # Resample to Weekly
    weekly_df = df.resample('W-SUN').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    # Add fractals to the weekly view
    weekly_df = add_williams_fractals(weekly_df, timeframe='1W', n=2) # n=1 means 3-bar pattern
    
    # Calculate S&R from the last 4 years of Weekly data
    major_srs = find_major_sr(weekly_df, tolerance_pips=5.0, min_touches=3)

    print("\n" + "="*60)
    print("--- 1. DAILY & WEEKLY BOUNDARIES (Last 5 Rows) ---")
    # Using .tail() without dropna to see exactly what's mapped
    print(df[['Close', 'PDH', 'PDL', 'PWH', 'PWL']].tail(5))

    print("\n" + "="*60)
    print("--- 2. MULTI-TIMEFRAME FVGs (Current State) ---")
    fvg_cols = [c for c in df.columns if 'FVG' in c and ('Type' in c or 'Mid' in c)]
    print(df[fvg_cols].tail(1).T)

    # Assuming your df ends on '2026-02-27'
    test_date = df.index[-1] 
    
    # Get all confirmed swings for the last 5 years relative to the end date
    swings_5yr = get_confirmed_swings(weekly_df, current_date=test_date, n=1, lookback_years=3)
    
    print("\n" + "="*60)
    print(f"--- 5-YEAR CONFIRMED HTF SWINGS (As of {test_date}) ---")
    print(f"Total Confirmed W1 Highs: {len(swings_5yr['Highs'])}")
    print(f"Total Confirmed W1 Lows: {len(swings_5yr['Lows'])}")
    if swings_5yr['Highs']:
        print(f"Last 3 Confirmed Highs: {swings_5yr['Highs']}")
    if swings_5yr['Lows']:
        print(f"Last 3 Confirmed Lows: {swings_5yr['Lows']}")
    print("="*60 + "\n")

    print("\n" + "="*60)
    print(f"--- 4. MAJOR W1 S&R ZONES (>= 3 Touches) ---")
    if major_srs:
        for i, level in enumerate(major_srs):
            print(f"Zone {i+1}: {level:.5f}")
    else:
        print("No Major S&R zones found. Try increasing 'tolerance_pips' or check data history.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_htf_test()