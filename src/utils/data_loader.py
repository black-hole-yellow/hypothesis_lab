import pandas as pd
import numpy as np
import pytz
import os

def load_and_prep_data(file_path: str, start_date: str, end_date: str, timeframe: str = '1h') -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Missing data file at {file_path}")
        return None

    print(f"Reading {file_path}...")
    cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        df = pd.read_csv(file_path, sep='\t', names=cols, index_col=False, low_memory=False)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # --- TIMEZONE MAGIC ---
        # 1. Tell Pandas the raw data is in US Eastern Time (America/New_York)
        # Using ambiguous='NaT' handles the weird 1-hour overlap during DST shifts
        df.index = df.index.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT')
        
        # 2. Create a dedicated column strictly for Ukraine Time logging
        # This automatically handles all historical DST shifts!
        df['UA_Time'] = df.index.tz_convert('Europe/Kiev')
        # ----------------------

        df = df[['Open', 'High', 'Low', 'Close', 'UA_Time']].dropna(subset=['Close'])
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
        
    except Exception as e:
        print(f"❌ ERROR parsing CSV: {e}")
        return None

    # Slice the requested date range
    sliced_df = df.loc[start_date:end_date]
    
    if sliced_df.empty:
        print(f"❌ ERROR: Slicing failed! No data found between {start_date} and {end_date}.")
        print(f"Available data runs from {df.index[0]} to {df.index[-1]}")
        return None

    # Standardize the timeframe string for pandas (e.g., '1h', '15min')
    tf_pandas = timeframe.replace('H', 'h').replace('m', 'min').replace('M', 'min')
    if tf_pandas.endswith('minh'): 
        tf_pandas = tf_pandas.replace('minh', 'min')

    # Resample to the requested timeframe
    resample_map = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'UA_Time': 'last'}
    resampled_df = sliced_df.resample(tf_pandas).agg(resample_map).dropna()

    print(f"✅ Successfully loaded {len(resampled_df)} candles for {timeframe} timeframe.")
    return resampled_df

def add_session_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags each row with the active Session based on the candle's hour.
    Asia: 02:00 - 10:00
    London: 10:00 - 18:00
    NY: 15:00 - 23:00
    """
    # Look at the Ukraine Time column we created, fallback to index if missing
    if 'UA_Time' in df.columns:
        hours = df['UA_Time'].dt.hour
    else:
        hours = df.index.hour
    
    # Define conditions for each session
    conditions = [
        (hours >= 2) & (hours < 10),   # Asia: 02:00 to 09:59
        (hours >= 10) & (hours < 15),  # London Only: 10:00 to 14:59
        (hours >= 15) & (hours < 18),  # London/NY Overlap: 15:00 to 17:59
        (hours >= 18) & (hours < 23)   # NY Only: 18:00 to 22:59
    ]
    
    choices = ['Asia', 'London', 'London/NY', 'NY']
    
    # Apply conditions, default to 'None' (e.g., the 23:00 to 01:59 dead zone)
    df['Session'] = np.select(conditions, choices, default='None')
    
    return df


df = pd.read_csv("data/gbpusd_data.csv", sep='\t', names=['Date','O','H','L','C','X'])
print(f"Data Starts: {df['Date'].iloc[0]}")
print(f"Data Ends: {df['Date'].iloc[-1]}")