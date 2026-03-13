import pandas as pd
from src.utils.data_loader import load_and_prep_data
from src.library.htf_features import add_previous_boundaries

def run_clean_boundaries_test():
    file_path = "data/gbpusd_data.csv"
    
    start_date = '2026-02-01'
    end_date = '2026-02-27'
    tf = '1h'
    
    df = load_and_prep_data(file_path, start_date, end_date, tf)
    
    # 1. TIMEZONE FIX: Convert to Ukraine time BEFORE calculating boundaries
    if df.index.tz is None:
        df.index = df.index.tz_localize('US/Eastern').tz_convert('Europe/Kyiv')
    else:
        df.index = df.index.tz_convert('Europe/Kyiv')

    # 2. Calculate boundaries (Now based on Kyiv Midnight to Midnight)
    df = add_previous_boundaries(df)
    
    df['Date'] = df.index.date

    # 3. Print Daily Boundaries
    print("\n--- PDH & PDL (Ukraine Timezone Date) ---")
    daily_summary = df.groupby('Date').first().dropna(subset=['PDH', 'PDL'])
    for date_obj, row in daily_summary.tail(10).iterrows():
        print(f"{date_obj} {row['PDH']:.5f} {row['PDL']:.5f}")

    # 4. Print Weekly Boundaries
    print("\n--- PWH & PWL (Ukraine Timezone Date) ---")
    weekly_summary = df.groupby(['PWH', 'PWL']).first().reset_index().sort_values('Date')
    for _, row in weekly_summary.tail(5).iterrows():
        print(f"{row['Date']} {row['PWH']:.5f} {row['PWL']:.5f}")
    print("\n")

if __name__ == "__main__":
    run_clean_boundaries_test()