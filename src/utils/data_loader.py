import pandas as pd
import numpy as np

import pandas as pd
import os

def load_and_prep_data(file_path: str, start_date: str, end_date: str, timeframe: str = '1h') -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Missing data file at {file_path}")
        return None

    print(f"Reading {file_path}...")

    # We know the exact structure now: Datetime, Open, High, Low, Close, Volume
    cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        # Load the raw data
        df = pd.read_csv(file_path, sep='\t', names=cols, index_col=False, low_memory=False)
        
        # Convert Datetime and set as index
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Keep only price columns and force them to be numbers
        df = df[['Open', 'High', 'Low', 'Close']].astype(float)
        
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
    resample_map = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    resampled_df = sliced_df.resample(tf_pandas).agg(resample_map).dropna()

    print(f"✅ Successfully loaded {len(resampled_df)} candles for {timeframe} timeframe.")
    return resampled_df

def add_session_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags each row with the active NY Session based on the candle's hour.
    Asia: 19:00 - 03:00
    London: 03:00 - 09:00
    NY: 09:00 - 16:00
    """
    hours = df.index.hour
    
    # Define conditions for each session
    conditions = [
        (hours >= 18) | (hours < 3),   # Asia crosses midnight
        (hours >= 2) & (hours < 8),    # London
        (hours >= 8) & (hours < 15)    # New York
    ]
    
    choices = ['Asia', 'London', 'NY']
    
    # Apply conditions, default to 'None' (e.g., the 16:00-19:00 dead zone)
    df['Session'] = np.select(conditions, choices, default='None')
    
    return df


df = pd.read_csv("data/gbpusd_data.csv", sep='\t', names=['Date','O','H','L','C','X'])
print(f"Data Starts: {df['Date'].iloc[0]}")
print(f"Data Ends: {df['Date'].iloc[-1]}")