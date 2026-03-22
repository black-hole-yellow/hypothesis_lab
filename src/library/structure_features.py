import pandas as pd
import numpy as np
from scipy.stats import linregress

PIP = 0.0001

def add_previous_boundaries(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
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

def calculate_fvgs(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
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

def calculate_multi_tf_fvgs(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
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

def add_fvg_order_flow_context(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
    """
    Identifies if price is inside a 4H FVG and if a 1H Order Flow Flip occurred.
    Requires calculate_multi_tf_fvgs and add_volume_zscore to be run first.
    """
    
    # --- 1. FVG Zone Detection ---
    # Bullish FVG: Price dips into the zone, but hasn't closed below it
    df['In_4h_Bull_FVG'] = (df['FVG_4h_Type'] == 'BULL') & \
                           (df['Low'] <= df['FVG_4h_Top']) & \
                           (df['Close'] >= df['FVG_4h_Bottom'])
                           
    # Bearish FVG: Price spikes into the zone, but hasn't closed above it
    df['In_4h_Bear_FVG'] = (df['FVG_4h_Type'] == 'BEAR') & \
                           (df['High'] >= df['FVG_4h_Bottom']) & \
                           (df['Close'] <= df['FVG_4h_Top'])

    # --- 2. 1H Order Flow Flip ---
    # Prev candle was Red, Current is Green WITH decent volume
    prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)
    curr_bullish = df['Close'] > df['Open']
    vol_expansion = df['Volume_ZScore'] > 0.5 
    
    df['1h_Bullish_Flip'] = prev_bearish & curr_bullish & vol_expansion
    
    # Prev candle was Green, Current is Red WITH decent volume
    prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
    curr_bearish = df['Close'] < df['Open']
    
    df['1h_Bearish_Flip'] = prev_bullish & curr_bearish & vol_expansion

    # Convert booleans to 1/0 so the JSON parser can easily read them
    df['In_4h_Bull_FVG'] = df['In_4h_Bull_FVG'].astype(int)
    df['In_4h_Bear_FVG'] = df['In_4h_Bear_FVG'].astype(int)
    df['1h_Bullish_Flip'] = df['1h_Bullish_Flip'].astype(int)
    df['1h_Bearish_Flip'] = df['1h_Bearish_Flip'].astype(int)
    
    return df

def add_weekly_swing_context(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
    """
    Calculates the Weekly Trend (Swing) using a 168-hour lookback.
    Returns 1 for Bullish (Price > SMA), -1 for Bearish.
    """
    # 168 hours = 1 Week of 1H data
    lookback = 168
    df['Weekly_SMA'] = df['Close'].rolling(window=lookback).mean()
    
    # Define Swing based on Price vs SMA
    df['Weekly_Swing'] = 0
    df.loc[df['Close'] > df['Weekly_SMA'], 'Weekly_Swing'] = 1
    df.loc[df['Close'] < df['Weekly_SMA'], 'Weekly_Swing'] = -1
    
    return df

def add_1w_swing_context(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
    """
    Determines the 1W HTF Swing direction based on the last broken weekly extreme.
    Guaranteed zero lookahead bias and stateful tracking.
    """
    # 1. Define the 1-Week (120 trading hours) rolling extremes
    df['1W_High'] = df['High'].rolling(window=120).max().shift(1)
    df['1W_Low'] = df['Low'].rolling(window=120).min().shift(1)

    # 2. Track structural breaks
    df['Broke_1W_High'] = df['Close'] > df['1W_High']
    df['Broke_1W_Low'] = df['Close'] < df['1W_Low']

    # 3. Create a stateful "Swing Signal" (1 for Bullish, -1 for Bearish)
    df['Swing_Signal'] = 0
    df.loc[df['Broke_1W_High'], 'Swing_Signal'] = 1
    df.loc[df['Broke_1W_Low'], 'Swing_Signal'] = -1
    
    # Forward fill the state so it holds 'Bullish' until a 'Bearish' break occurs
    df['1W_Swing_State'] = df['Swing_Signal'].replace(0, np.nan).ffill().fillna(0)

    # 4. Binary features for the JSON Parser
    df['1W_Swing_Bullish'] = (df['1W_Swing_State'] == 1).astype(int)
    df['1W_Swing_Bearish'] = (df['1W_Swing_State'] == -1).astype(int)

    # Clean up intermediate columns
    df.drop(columns=['1W_High', '1W_Low', 'Broke_1W_High', 'Broke_1W_Low', 'Swing_Signal', '1W_Swing_State'], inplace=True)
    
    return df

def add_1d_swing_context(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
    """
    Determines the 1D Swing direction based on the last broken daily extreme.
    Tracks structural breaks over a rolling 24-hour period.
    """
    
    # 1. 1-Day (24 hours) rolling extremes
    df['1D_High'] = df['High'].rolling(window=24).max().shift(1)
    df['1D_Low'] = df['Low'].rolling(window=24).min().shift(1)

    # 2. Track structural breaks
    df['Broke_1D_High'] = df['Close'] > df['1D_High']
    df['Broke_1D_Low'] = df['Close'] < df['1D_Low']

    # 3. Stateful Swing Signal
    df['1D_Swing_Signal'] = 0
    df.loc[df['Broke_1D_High'], '1D_Swing_Signal'] = 1
    df.loc[df['Broke_1D_Low'], '1D_Swing_Signal'] = -1
    
    df['1D_Swing_State'] = df['1D_Swing_Signal'].replace(0, np.nan).ffill().fillna(0)

    # 4. Binary features
    df['1D_Swing_Bullish'] = (df['1D_Swing_State'] == 1).astype(int)
    df['1D_Swing_Bearish'] = (df['1D_Swing_State'] == -1).astype(int)

    df.drop(columns=['1D_High', '1D_Low', 'Broke_1D_High', 'Broke_1D_Low', '1D_Swing_Signal', '1D_Swing_State'], inplace=True)
    return df

def add_weekly_floor_context(df: pd.DataFrame, events: list = None,) -> pd.DataFrame:
    """
    Detects if a London Fractal aligns with the Weekly Swing direction.
    Uses an expanded time window (09:00 - 14:00) to account for n=2 confirmation lag.
    """
    # 1. Time Window
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    
    # 2. Extract Data Safely (Converting to boolean to prevent 1/0 matching errors)
    bull_swing = df['1W_Swing_Bullish'].fillna(0) == 1
    bear_swing = df['1W_Swing_Bearish'].fillna(0) == 1
    
    fractal_low = df['Confirmed_Fractal_Low'].fillna(0) == 1
    fractal_high = df['Confirmed_Fractal_High'].fillna(0) == 1
    
    # 3. Match Logic: Bullish Swing + Fractal Low (Support)
    df['LDN_Weekly_Bull_Floor'] = is_london_eval & bull_swing & fractal_low
    
    # Match Logic: Bearish Swing + Fractal High (Resistance)
    df['LDN_Weekly_Bear_Floor'] = is_london_eval & bear_swing & fractal_high
    
    # 4. Ensure strictly ONE trigger per day (The Bulletproof Fix)
    df['Date'] = df.index.date
    df['Floor_Trigger_Count'] = (df['LDN_Weekly_Bull_Floor'] | df['LDN_Weekly_Bear_Floor']).groupby(df['Date']).cumsum()
    
    # 5. Final Engine Output Columns
    df['First_LDN_Weekly_Bull'] = (df['LDN_Weekly_Bull_Floor'] & (df['Floor_Trigger_Count'] == 1)).astype(int)
    df['First_LDN_Weekly_Bear'] = (df['LDN_Weekly_Bear_Floor'] & (df['Floor_Trigger_Count'] == 1)).astype(int)
    
    # Clean up intermediate data
    df.drop(columns=['Date', 'Floor_Trigger_Count', 'LDN_Weekly_Bull_Floor', 'LDN_Weekly_Bear_Floor'], inplace=True)
    
    return df

def add_fvg_sr_confluence_context(df: pd.DataFrame,events: list = None, max_dist_pips: int = 30) -> pd.DataFrame:
    """
    Detects when a 1D FVG forms immediately adjacent to a Major S&R level (Confluence).
    Triggers on the strictly FIRST touch of this confluence zone per day.
    """
    PIP = 0.0001
    dist = max_dist_pips * PIP

    # 1. Corrected Geometry: Major Level is just outside the FVG, acting as a wall
    # Res: Major Resistance is within 30 pips above the FVG Top
    df['Res_Confluence_Zone'] = (df['FVG_1D_Type'] == 'BEAR') & \
                                (df['Major_Resistance'] >= df['FVG_1D_Top']) & \
                                ((df['Major_Resistance'] - df['FVG_1D_Top']) <= dist)

    # Sup: Major Support is within 30 pips below the FVG Bottom
    df['Sup_Confluence_Zone'] = (df['FVG_1D_Type'] == 'BULL') & \
                                (df['Major_Support'] <= df['FVG_1D_Bottom']) & \
                                ((df['FVG_1D_Bottom'] - df['Major_Support']) <= dist)

    # 2. Detect Raw Touches (Price taps the FVG opening)
    df['Touch_Res_Zone'] = df['Res_Confluence_Zone'] & (df['High'] >= df['FVG_1D_Bottom'])
    df['Touch_Sup_Zone'] = df['Sup_Confluence_Zone'] & (df['Low'] <= df['FVG_1D_Top'])

    # 3. The Bulletproof Cumsum Fix (One touch per day)
    df['Date'] = df.index.date
    df['Res_Zone_Touches'] = df.groupby('Date')['Touch_Res_Zone'].cumsum()
    df['Sup_Zone_Touches'] = df.groupby('Date')['Touch_Sup_Zone'].cumsum()

    df['First_Touch_Res_Zone'] = df['Touch_Res_Zone'] & (df['Res_Zone_Touches'] == 1)
    df['First_Touch_Sup_Zone'] = df['Touch_Sup_Zone'] & (df['Sup_Zone_Touches'] == 1)

    # 4. Clean up
    df.drop(columns=['Date', 'Res_Zone_Touches', 'Sup_Zone_Touches'], inplace=True)
    df['First_Touch_Res_Zone'] = df['First_Touch_Res_Zone'].astype(int)
    df['First_Touch_Sup_Zone'] = df['First_Touch_Sup_Zone'].astype(int)

    return df

def add_1w_level_rejection_context(df: pd.DataFrame, events: list = None, max_dist_pips: int = 20) -> pd.DataFrame:
    """
    Detects if a counter-trend pullback taps a Major Weekly Level (PWH/PWL) 
    and prints a confirmed fractal, signaling macro trend continuation.
    Requires add_previous_boundaries and add_confirmed_fractals.
    """
    PIP = 0.0001
    tol = max_dist_pips * PIP
    
    # 1. Safely extract fractal signals
    fractal_low = (df['Confirmed_Fractal_Low'].fillna(0) == 1)
    fractal_high = (df['Confirmed_Fractal_High'].fillna(0) == 1)
    
    # 2. Check if the fractal wick tapped or swept the Weekly Extremes
    # We allow a 20-pip tolerance above/below the exact level
    tap_pwl = df['Confirmed_Fractal_Low_Price'] <= (df['PWL'] + tol)
    tap_pwh = df['Confirmed_Fractal_High_Price'] >= (df['PWH'] - tol)
              
    # 3. Active Trading Hours (London & NY: 10:00 - 21:00 Kyiv Time)
    is_active = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 21)
    
    # 4. Trigger Logic (Fractal forms at the weekly level during active hours)
    df['Reject_PWL_Long'] = is_active & fractal_low & tap_pwl
    df['Reject_PWH_Short'] = is_active & fractal_high & tap_pwh
    
    # 5. Bulletproof: Strictly ONE trigger per day
    df['Date'] = df.index.date
    df['1W_Rej_Trigger_Count'] = (df['Reject_PWL_Long'] | df['Reject_PWH_Short']).groupby(df['Date']).cumsum()
    
    df['First_1W_Rej_Long'] = (df['Reject_PWL_Long'] & (df['1W_Rej_Trigger_Count'] == 1)).astype(int)
    df['First_1W_Rej_Short'] = (df['Reject_PWH_Short'] & (df['1W_Rej_Trigger_Count'] == 1)).astype(int)
    
    # Clean up
    df.drop(columns=['Date', '1W_Rej_Trigger_Count', 'Reject_PWL_Long', 'Reject_PWH_Short'], inplace=True)
    return df

def add_htf_trend_probability(df: pd.DataFrame, events: list = None, htf: str = '4h', lookback: int = 60) -> pd.DataFrame:
    """
    Calculates a 0-100% Bullish Trend Probability using HTF Confluence.
    50% based on Market Structure (Higher Highs / Lower Lows).
    50% based on Statistical Vector (Linear Regression Slope * R-Squared).
    """

    # --- ЗАЩИТА ОТ ДВОЙНОГО РАСЧЕТА ---
    if 'HTF_Bullish_Prob' in df.columns:
        # Если колонка уже есть, просто возвращаем DF
        return df
    
    df = df.copy()

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
    if 'HTF_Bullish_Prob' not in df.columns:
        df = df.join(htf_df[['HTF_Bullish_Prob']])
    
    # Forward fill the 4H value across the four 1H candles
    df['HTF_Bullish_Prob'] = df['HTF_Bullish_Prob'].ffill().fillna(50.0) 
    
    # Round to 1 decimal place for clean logging
    df['HTF_Bullish_Prob'] = df['HTF_Bullish_Prob'].round(1)

    return df

def add_structure_flip_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    
    # Detect 3-bar swings
    df['Swing_High'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
    df['Swing_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
    
    df['Last_Swing_High'] = df['High'].where(df['Swing_High']).ffill()
    df['Last_Swing_Low'] = df['Low'].where(df['Swing_Low']).ffill()
    
    bullish_break = (df['Close'] > df['Last_Swing_High'].shift(1))
    bearish_break = (df['Close'] < df['Last_Swing_Low'].shift(1))
    
    df['1h_Bullish_Flip'] = bullish_break.rolling(3, min_periods=1).max().fillna(0).astype(int)
    df['1h_Bearish_Flip'] = bearish_break.rolling(3, min_periods=1).max().fillna(0).astype(int)
    
    df.drop(columns=['Swing_High', 'Swing_Low', 'Last_Swing_High', 'Last_Swing_Low'], inplace=True, errors='ignore')
    return df

def add_asia_fvg_protection_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    """
    Определяет, находится ли Азиатский Хай/Лоу внутри 4H FVG.
    Запускает триггер на продолжение тренда ровно в 10:00 (London Open).
    """
    # --- ДОБАВИТЬ ЭТОТ БЛОК: Локальный расчет Азии ---
    if 'Asia_Low' not in df.columns or 'Asia_High' not in df.columns:
        df['Day_Key'] = df.index.normalize()
        asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
        asia_stats = df[asia_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
        df['Asia_High'] = df['Day_Key'].map(asia_stats['High'])
        df['Asia_Low'] = df['Day_Key'].map(asia_stats['Low'])
    # --------------------------------------------------

    # 1. Проверяем, находится ли Азиатский Лоу внутри бычьего 4H FVG
    df['Asia_Low'] = (df['FVG_4h_Type'] == 'BULL') & \
                               (df['Asia_Low'] <= df['FVG_4h_Top']) & \
                               (df['Asia_Low'] >= df['FVG_4h_Bottom'])

    # 2. Проверяем, находится ли Азиатский Хай внутри медвежьего 4H FVG
    df['Asia_High'] = (df['FVG_4h_Type'] == 'BEAR') & \
                                (df['Asia_High'] <= df['FVG_4h_Top']) & \
                                (df['Asia_High'] >= df['FVG_4h_Bottom'])

    # 3. Триггер ровно на первой свече открытия Лондона (10:00 Киев)
    is_london_open = (df['UA_Hour'] == 10)

    # Лонг, если Азиатский Лоу защищен
    df['LDN_Protected_AL_Long'] = is_london_open & df['Asia_Low']
    
    # Шорт, если Азиатский Хай защищен
    df['LDN_Protected_AH_Short'] = is_london_open & df['Asia_High']

    # Очистка и форматирование (1/0 для парсера)
    df['LDN_Protected_AL_Long'] = df['LDN_Protected_AL_Long'].fillna(False).astype(int)
    df['LDN_Protected_AH_Short'] = df['LDN_Protected_AH_Short'].fillna(False).astype(int)

    df.drop(columns=['Day_Key'], inplace=True, errors='ignore')
    return df

def add_asian_sweep_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Ensure Asia boundaries exist
    if 'Asia_High' not in df.columns:
        df['Date_Key_Tmp'] = df.index.date
        is_asia = (df['UA_Hour'] >= 0) & (df['UA_Hour'] <= 8)
        asia_stats = df[is_asia].groupby('Date_Key_Tmp').agg({'High': 'max', 'Low': 'min'})
        df['Asia_High'] = df['Date_Key_Tmp'].map(asia_stats['High'])
        df['Asia_Low'] = df['Date_Key_Tmp'].map(asia_stats['Low'])
        df.drop(columns=['Date_Key_Tmp'], inplace=True)
    
    # 2. Setup the missing columns with defaults
    df['Swept_AL_Into_FVG'] = 0
    df['Swept_AH_Into_FVG'] = 0

    # 3. Calculate sweeps if HTF FVG exists
    if 'FVG_4h_Type' in df.columns:
        sweep_al = (df['FVG_4h_Type'] == 'BULL') & (df['Low'] < df['Asia_Low']) & (df['Low'] <= df['FVG_4h_Top'])
        sweep_ah = (df['FVG_4h_Type'] == 'BEAR') & (df['High'] > df['Asia_High']) & (df['High'] >= df['FVG_4h_Bottom'])
        
        df['Swept_AL_Into_FVG'] = sweep_al.rolling(3, min_periods=1).max().fillna(0).astype(int)
        df['Swept_AH_Into_FVG'] = sweep_ah.rolling(3, min_periods=1).max().fillna(0).astype(int)
        
    return df