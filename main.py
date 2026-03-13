import os
import pandas as pd

# Import our custom Lab modules
from src.utils.data_loader import load_and_prep_data, add_session_tags
from src.library.features import add_williams_fractals, add_volatility_zscore, add_normalized_slope
from src.library.htf_features import add_previous_boundaries, calculate_fvgs  # <--- NEW IMPORTS
from src.hypotheses.london_fake_move import LondonFakeMove
from src.core.base_hypothesis import State

def run_lab():
    # 1. Configuration
    data_file = "data/gbpusd_data.csv"
    output_dir = "output"
    output_file = f"{output_dir}/hypothesis_results.csv"
    
    start_date = '2025-01-01' # Adjust to your desired testing range
    end_date = '2026-02-27'
    timeframe = '1h'
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Hypothesis Lab Execution Started ---")
    print(f"Loading data from {start_date} to {end_date} ({timeframe})...")

    # 2. Data Pipeline
    try:
        # Load Base Data
        df = load_and_prep_data(data_file, start_date, end_date, timeframe)
        
        # Apply Intrinsic Features (DNA)
        df = add_session_tags(df)
        df = add_williams_fractals(df, timeframe=timeframe, n=2)
        df = add_volatility_zscore(df, lookback=50)
        df = add_normalized_slope(df, lookback=20, atr_lookback=14)
        
        # Apply HTF / Environmental Features (Weather) <--- NEW PIPELINE STEP
        print("Calculating HTF Boundaries and FVGs...")
        df = add_previous_boundaries(df)
        df = calculate_fvgs(df)
        
    except Exception as e:
        print(f"Data Pipeline Error: {e}")
        return

    print(f"Data ready. Total candles: {len(df)}. Running hypotheses...")

    # 3. Engine Setup
    active_hypotheses = [LondonFakeMove()]
    results_database = []

    # 4. The Event Loop
    for index, row in df.iterrows():
        for hypo in active_hypotheses:
            hypo.process_candle(index, row)
            
            if hypo.state == State.COMPLETED:
                results_database.append(hypo.get_csv_row())
                hypo.reset()

    # 5. Export Results
    if results_database:
        results_df = pd.DataFrame(results_database)
        
        # Convert NY Time to Ukraine Time for the CSV
        results_df['Trigger_Time'] = pd.to_datetime(results_df['Trigger_Time'])
        results_df['Trigger_Time'] = (results_df['Trigger_Time']
                                      .dt.tz_localize('US/Eastern')
                                      .dt.tz_convert('Europe/Kyiv')
                                      .dt.tz_localize(None))

        results_df.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Logged {len(results_df)} events.")
        print(f"Results saved to: {output_file}")
        
        win_rate = (results_df['Result'] == True).mean() * 100
        print(f"\n--- Quick Stat: {active_hypotheses[0].hypothesis_id} ---")
        print(f"Total Occurrences: {len(results_df)}")
        print(f"True (Target Hit): {results_df['Result'].sum()}")
        print(f"Win Rate: {win_rate:.2f}%")
    else:
        print("\nNo events triggered during this date range.")

if __name__ == "__main__":
    run_lab()