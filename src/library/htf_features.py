import pandas as pd
import numpy as np
from scipy.stats import linregress

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

def add_htf_trend_probability(df: pd.DataFrame, htf: str = '4h', lookback: int = 120) -> pd.DataFrame:
    """
    Calculates a 0-100% Bullish Trend Probability using HTF Confluence.
    50% based on Market Structure (Higher Highs / Lower Lows).
    50% based on Statistical Vector (Linear Regression Slope * R-Squared).
    """
    # 1. Resample to Higher Timeframe
    # We use 'h' for pandas timeframe formatting compatibility
    htf_df = pd.DataFrame()
    htf_df['HTF_Close'] = df['Close'].resample(htf).last()
    htf_df['HTF_High'] = df['High'].resample(htf).max()
    htf_df['HTF_Low'] = df['Low'].resample(htf).min()
    htf_df.dropna(inplace=True)

    # ---------------------------------------------------------
    # PART 1: STATISTICAL VECTOR (Linear Regression)
    # ---------------------------------------------------------
    slopes = []
    r_squareds = []
    
    for i in range(len(htf_df)):
        if i < lookback:
            slopes.append(0)
            r_squareds.append(0)
            continue
            
        y = htf_df['HTF_Close'].iloc[i-lookback:i].values
        x = np.arange(lookback)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slopes.append(slope)
        r_squareds.append(r_value**2)

    htf_df['Slope'] = slopes
    htf_df['R2'] = r_squareds

    # Stat Score: Max 50%. 
    # If slope is Up: 50 * R^2. If slope is Down: 0.
    htf_df['Stat_Score'] = np.where(htf_df['Slope'] > 0, 50 * htf_df['R2'], 0)

    # ---------------------------------------------------------
    # PART 2: MARKET STRUCTURE (Fractals)
    # ---------------------------------------------------------
    n = 2
    htf_df['Fractal_Up'] = False
    htf_df['Fractal_Down'] = False
    
    # Calculate HTF Fractals
    for i in range(2*n, len(htf_df)):
        window_high = htf_df['HTF_High'].iloc[i - 2*n : i + 1]
        window_low = htf_df['HTF_Low'].iloc[i - 2*n : i + 1]
        mid_idx = i - n
        
        if htf_df['HTF_High'].iloc[mid_idx] == window_high.max():
            htf_df.iat[mid_idx, htf_df.columns.get_loc('Fractal_Up')] = True
        if htf_df['HTF_Low'].iloc[mid_idx] == window_low.min():
            htf_df.iat[mid_idx, htf_df.columns.get_loc('Fractal_Down')] = True

    # Trace HH/HL Logic without lookahead bias
    struct_scores = []
    last_up_1, last_up_2 = None, None
    last_down_1, last_down_2 = None, None

    for i in range(len(htf_df)):
        # To avoid lookahead, we only acknowledge a fractal 'n' periods after it formed
        check_idx = i - n
        if check_idx >= 0:
            if htf_df['Fractal_Up'].iloc[check_idx]:
                last_up_2 = last_up_1
                last_up_1 = htf_df['HTF_High'].iloc[check_idx]
            if htf_df['Fractal_Down'].iloc[check_idx]:
                last_down_2 = last_down_1
                last_down_1 = htf_df['HTF_Low'].iloc[check_idx]

        score = 25 # Default Neutral
        if last_up_1 and last_up_2 and last_down_1 and last_down_2:
            hh = last_up_1 > last_up_2
            hl = last_down_1 > last_down_2
            lh = last_up_1 < last_up_2
            ll = last_down_1 < last_down_2

            if hh and hl:
                score = 50  # Perfect Bullish Structure
            elif lh and ll:
                score = 0   # Perfect Bearish Structure
        
        struct_scores.append(score)

    htf_df['Struct_Score'] = struct_scores

    # ---------------------------------------------------------
    # MERGE & ALIGN WITH 1H CHART
    # ---------------------------------------------------------
    htf_df['HTF_Bullish_Prob'] = htf_df['Stat_Score'] + htf_df['Struct_Score']
    
    # SHIFT BY 1 HTF BAR! This guarantees the algorithm only knows 
    # the 4H trend AFTER the 4H candle has completely closed.
    htf_df['HTF_Bullish_Prob'] = htf_df['HTF_Bullish_Prob'].shift(1)

    # Map back to your main dataframe
    df = df.join(htf_df[['HTF_Bullish_Prob']])
    
    # Forward fill the 4H value across the four 1H candles
    df['HTF_Bullish_Prob'] = df['HTF_Bullish_Prob'].ffill().fillna(50.0) 
    
    # Round to 1 decimal place for clean logging
    df['HTF_Bullish_Prob'] = df['HTF_Bullish_Prob'].round(1)

    return df