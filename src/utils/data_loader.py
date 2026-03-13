import pandas as pd
import numpy as np

def load_and_prep_data(file_path: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
    """
    Loads raw CSV data, formats the Datetime, slices by date, and resamples.
    Automatically handles comma or space delimited files.
    """
    # 1. Detect separator
    with open(file_path, 'r') as f:
        first_line = f.readline()
        
    if ',' in first_line:
        df = pd.read_csv(file_path, header=None, sep=',', engine='python', 
                         names=['Datetime', 'Open', 'High', 'Low', 'Close'])
    else:
        df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python', 
                         names=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
        df['Datetime'] = df['Date'] + ' ' + df['Time']
        df.drop(columns=['Date', 'Time'], inplace=True)
    
    # 2. Parse Datetime & Sort
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # 3. Slice Dates
    if start_date in df.index or not df.loc[start_date:end_date].empty:
        df = df.loc[start_date:end_date]
    else:
        raise ValueError(f"Date range {start_date} to {end_date} not found in data.")
    
    # 4. Resample
    resample_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    df_resampled = df.resample(timeframe).agg(resample_dict).dropna()
    
    return df_resampled

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