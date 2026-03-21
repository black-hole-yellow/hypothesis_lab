import pandas as pd
from src.utils.data_loader import load_and_prep_data
from library.structure_features import (
    add_previous_boundaries, 
    calculate_multi_tf_fvgs, 
    get_confirmed_swings, 
    find_major_sr
)
from src.library.features import add_williams_fractals

def run_comprehensive_test():
    file_path = "data/gbpusd_data.csv"
    
    # Load enough history for Weekly math to work
    df = load_and_prep_data(file_path, '2024-01-01', '2026-03-10', '1h')
    
    # --- 1. Fix Timezone First ---
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern').tz_convert('Europe/Kyiv')
    
    print("--- 1. Testing Boundaries (Ukraine Session) ---")
    df = add_previous_boundaries(df)
    # Check the very last day in the data
    last_date = df.index[-1].date()
    daily = df[df.index.date == last_date].iloc[0]
    print(f"Date: {last_date} | PDH: {daily['PDH']:.5f} | PDL: {daily['PDL']:.5f} | PWH: {daily['PWH']:.5f}")

    print("\n--- 2. Testing Multi-TF FVGs ---")
    df = calculate_multi_tf_fvgs(df)
    # Print the most recent active gaps
    fvg_cols = [c for c in df.columns if 'FVG' in c and 'Type' in c]
    print(df[fvg_cols].tail(1).T.to_string(header=False))

    print("\n--- 3. Testing Confirmed 5-Year Swings ---")
    # Resample to weekly for fractal calc
    # Prepare different timeframes
    # Inside your test script...
    daily_df = df.resample('1D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    weekly_df = df.resample('W-SUN').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    
    n_swings = 3 
    weekly_df = add_williams_fractals(weekly_df, timeframe='1W', n=n_swings)
    
    # Pass BOTH weekly_df and daily_df
    swings = get_confirmed_swings(
        weekly_df, 
        daily_df, 
        current_date=df.index[-1], 
        n=n_swings, 
        lookback_years=5,
        tolerance_pips=50.0  
    )
    
    print(f"--- Confirmed Highs (n={n_swings}): {len(swings['Highs'])} ---")
    for lvl in swings['Highs']:
        print(f"High: {lvl:.5f}")
if __name__ == "__main__":
    run_comprehensive_test()