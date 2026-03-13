import pandas as pd
import numpy as np
from datetime import timedelta

PIP = 0.0001 # Standard pip size for GBPUSD

def add_previous_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Static session-based boundaries for PDH/PDL and PWH/PWL.
    Ensures levels update ONLY when a new session begins.
    """
    # 1. DAILY BOUNDARIES
    # Group by the date part of the index
    daily_stats = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'})
    # Shift so today sees 'yesterday's' values
    daily_stats['PDH'] = daily_stats['High'].shift(1)
    daily_stats['PDL'] = daily_stats['Low'].shift(1)
    
    # Map back to 1h timeframe
    df['date_key'] = df.index.date
    df = df.join(daily_stats[['PDH', 'PDL']], on='date_key')

    # 2. WEEKLY BOUNDARIES
    # We use 'isocalendar' weeks to ensure Monday-Sunday alignment
    df['week_key'] = df.index.isocalendar().week
    df['year_key'] = df.index.isocalendar().year
    
    # Calculate weekly extremes
    weekly_stats = df.groupby(['year_key', 'week_key']).agg({'High': 'max', 'Low': 'min'})
    # Shift within the group to get PREVIOUS week
    weekly_stats['PWH'] = weekly_stats['High'].shift(1)
    weekly_stats['PWL'] = weekly_stats['Low'].shift(1)
    
    # Map back to 1h timeframe
    df = df.join(weekly_stats[['PWH', 'PWL']], on=['year_key', 'week_key'])

    # Cleanup temporary keys
    df.drop(columns=['date_key', 'week_key', 'year_key'], inplace=True)
    
    # Forward fill to catch any gaps from weekends
    df[['PDH', 'PDL', 'PWH', 'PWL']] = df[['PDH', 'PDL', 'PWH', 'PWL']].ffill()
    
    return df

def calculate_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies ALL FVGs (Bullish/Bearish) and calculates their Middle Line.
    (Closure tracking removed per user request for performance and scope).
    """
    df['FVG_Type'] = None
    df['FVG_Top'] = np.nan
    df['FVG_Bottom'] = np.nan
    df['FVG_Mid'] = np.nan

    # FVG Creation Logic (Shifted by 2 to compare Candle 1 and Candle 3)
    bull_gap = df['Low'] > df['High'].shift(2)
    bear_gap = df['High'] < df['Low'].shift(2)
    
    # Populate Bullish FVGs
    df.loc[bull_gap, 'FVG_Type'] = 'BULL'
    df.loc[bull_gap, 'FVG_Top'] = df['Low']
    df.loc[bull_gap, 'FVG_Bottom'] = df['High'].shift(2)
    
    # Populate Bearish FVGs
    df.loc[bear_gap, 'FVG_Type'] = 'BEAR'
    df.loc[bear_gap, 'FVG_Top'] = df['Low'].shift(2)
    df.loc[bear_gap, 'FVG_Bottom'] = df['High']
    
    # Midpoint
    df['FVG_Mid'] = (df['FVG_Top'] + df['FVG_Bottom']) / 2

    return df

def get_htf_fvgs(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples data to a higher timeframe, calculates FVGs, and shifts them 
    forward to prevent lookahead bias.
    """
    # 1. Resample to HTF
    htf_df = df.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    
    # 2. Identify Gaps
    bull_gap = htf_df['Low'] > htf_df['High'].shift(2)
    bear_gap = htf_df['High'] < htf_df['Low'].shift(2)
    
    # 3. Store FVG Data
    htf_df[f'FVG_{timeframe}_Type'] = None
    htf_df.loc[bull_gap, f'FVG_{timeframe}_Type'] = 'BULL'
    htf_df.loc[bear_gap, f'FVG_{timeframe}_Type'] = 'BEAR'
    
    htf_df[f'FVG_{timeframe}_Top'] = np.nan
    htf_df.loc[bull_gap, f'FVG_{timeframe}_Top'] = htf_df['Low']
    htf_df.loc[bear_gap, f'FVG_{timeframe}_Top'] = htf_df['Low'].shift(2)
    
    htf_df[f'FVG_{timeframe}_Bottom'] = np.nan
    htf_df.loc[bull_gap, f'FVG_{timeframe}_Bottom'] = htf_df['High'].shift(2)
    htf_df.loc[bear_gap, f'FVG_{timeframe}_Bottom'] = htf_df['High']
    
    htf_df[f'FVG_{timeframe}_Mid'] = (htf_df[f'FVG_{timeframe}_Top'] + htf_df[f'FVG_{timeframe}_Bottom']) / 2
    
    # 4. Shift forward to prevent lookahead bias (only valid AFTER the candle closes)
    result_cols = [f'FVG_{timeframe}_Type', f'FVG_{timeframe}_Top', f'FVG_{timeframe}_Bottom', f'FVG_{timeframe}_Mid']
    return htf_df[result_cols].shift(1)

def calculate_multi_tf_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates FVGs across 1h, 4h, and 1D timeframes and maps them 
    back to the base 1-hour dataframe.
    """
    # 1. Base Timeframe (1h) FVGs
    df_1h = get_htf_fvgs(df, '1h')
    df = df.join(df_1h)
    
    # 2. 4-Hour FVGs
    df_4h = get_htf_fvgs(df, '4h')
    # Reindex maps the 4h data to the 1h timestamps, ffill carries the last known 4h FVG forward
    df_4h_mapped = df_4h.reindex(df.index).ffill() 
    df = df.join(df_4h_mapped)
    
    # 3. Daily FVGs
    df_1d = get_htf_fvgs(df, '1D')
    df_1d_mapped = df_1d.reindex(df.index).ffill()
    df = df.join(df_1d_mapped)
    
    return df

def find_major_sr(weekly_df: pd.DataFrame, daily_df: pd.DataFrame, tolerance_pips: float = 10.0, min_touches: int = 5) -> list:
    """
    Identifies Major S&R levels.
    1. Uses Weekly Fractals as 'Candidate' levels.
    2. Uses Daily (1D) candles to count touches for validation.
    """
    tolerance = tolerance_pips * PIP
    
    # Get candidate levels from Weekly Fractals
    highs = weekly_df[weekly_df['Fractal_High'] == True]['High'].tolist()
    lows = weekly_df[weekly_df['Fractal_Low'] == True]['Low'].tolist()
    all_candidates = sorted(highs + lows)
    
    if not all_candidates:
        return []

    # Cluster candidates that are near each other
    clusters = []
    if all_candidates:
        current_cluster = [all_candidates[0]]
        for level in all_candidates[1:]:
            if abs(level - np.mean(current_cluster)) <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
    
    major_sr = []
    for cluster in clusters:
        mean_level = np.mean(cluster)
        
        # Count how many DAILY candles touched this level
        # A touch is defined as the Daily High >= Level - Tol AND Daily Low <= Level + Tol
        touches = len(daily_df[
            (daily_df['High'] >= mean_level - tolerance) & 
            (daily_df['Low'] <= mean_level + tolerance)
        ])
        
        if touches >= min_touches:
            major_sr.append(mean_level)
            
    return major_sr

def filter_clustered_swings(swing_prices: list, daily_df: pd.DataFrame, tolerance_pips: float = 15.0) -> list:
    """
    Groups swings that are close to each other. 
    Uses 1D (Daily) candles to count historical touches, and keeps 
    only the swing with the highest Daily touch count in that cluster.
    """
    if not swing_prices:
        return []
        
    tolerance = tolerance_pips * PIP
    
    # 1. Sort and Cluster the prices
    sorted_swings = sorted(swing_prices)
    clusters = []
    current_cluster = [sorted_swings[0]]
    
    for price in sorted_swings[1:]:
        if abs(price - np.mean(current_cluster)) <= tolerance:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]
    clusters.append(current_cluster)
    
    # 2. Filter clusters based on 1D touch count
    filtered_swings = []
    for cluster in clusters:
        if len(cluster) == 1:
            filtered_swings.append(cluster[0])
        else:
            best_swing = cluster[0]
            max_touches = -1
            
            for swing in cluster:
                # Count touches using DAILY candles
                touches = len(daily_df[
                    (daily_df['Low'] <= swing + tolerance) & 
                    (daily_df['High'] >= swing - tolerance)
                ])
                
                if touches > max_touches:
                    max_touches = touches
                    best_swing = swing
                    
            filtered_swings.append(best_swing)
            
    return filtered_swings

def get_confirmed_swings(weekly_df: pd.DataFrame, daily_df: pd.DataFrame, current_date: pd.Timestamp, n: int = 1, lookback_years: int = 5, tolerance_pips: float = 15.0) -> dict:
    """
    Extracts 100% confirmed Weekly Swings (5-year rolling lookback) and filters 
    out redundant clustered swings based on 1D historical touches.
    """
    cutoff_date = current_date - pd.DateOffset(years=lookback_years)
    
    # 1. Slice Weekly Data for swings
    history_df = weekly_df.loc[:current_date].copy()
    if history_df.empty:
        return {'Highs': [], 'Lows': []}

    history_df['Confirmed_High_Signal'] = history_df['Fractal_High'].shift(n)
    history_df['Confirmed_Low_Signal'] = history_df['Fractal_Low'].shift(n)
    history_df['Swing_High_Price'] = history_df['High'].shift(n)
    history_df['Swing_Low_Price'] = history_df['Low'].shift(n)
    
    valid_history = history_df[history_df.index >= cutoff_date]
    raw_highs = valid_history[valid_history['Confirmed_High_Signal'] == True]['Swing_High_Price'].tolist()
    raw_lows = valid_history[valid_history['Confirmed_Low_Signal'] == True]['Swing_Low_Price'].tolist()
    
    # 2. Slice Daily Data for touch validation
    valid_daily = daily_df[(daily_df.index >= cutoff_date) & (daily_df.index <= current_date)]
    
    # 3. Apply the Clustered Touch Filter using the Daily Data
    final_highs = filter_clustered_swings(raw_highs, valid_daily, tolerance_pips=tolerance_pips)
    final_lows = filter_clustered_swings(raw_lows, valid_daily, tolerance_pips=tolerance_pips)
    
    return {
        'Highs': final_highs,
        'Lows': final_lows
    }