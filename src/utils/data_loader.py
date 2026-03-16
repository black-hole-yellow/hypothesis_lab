import pandas as pd
import numpy as np

import pandas as pd
import os

def load_and_prep_data(file_path: str, start_date: str, end_date: str, timeframe: str = '1h') -> pd.DataFrame:
    """
    Robust data loader. Auto-detects delimiters, handles missing headers, 
    and standardizes date parsing so data slices correctly.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing data file: {file_path}")

    print(f"Reading {file_path}...")

    # 1. Sniff the first line to figure out the delimiter (tab or comma)
    with open(file_path, 'r') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','

    # 2. Load the CSV. First, try assuming it has headers.
    try:
        df = pd.read_csv(file_path, sep=delimiter, low_memory=False)
        # Clean column names (strip spaces, make uppercase for uniform searching)
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # Check if standard headers exist. If not, reload assuming no headers.
        if not any(col in df.columns for col in ['OPEN', 'HIGH', 'CLOSE']):
            df = pd.read_csv(file_path, sep=delimiter, header=None)
            
            # Typical Forex export format: Date, Time, Open, High, Low, Close, Volume
            if len(df.columns) >= 6:
                df.columns = ['DATE', 'TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE'] + [f'EXTRA_{i}' for i in range(len(df.columns)-6)]
            elif len(df.columns) == 5: # Datetime combined
                df.columns = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE']

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()

    # 3. Handle Datetime parsing
    if 'DATETIME' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    elif 'DATE' in df.columns and 'TIME' in df.columns:
        # Combine separate Date and Time columns
        df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
    elif 'DATE' in df.columns:
        df['DATETIME'] = pd.to_datetime(df['DATE'])

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