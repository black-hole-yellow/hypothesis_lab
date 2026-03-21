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

def add_asian_sweep_context(df: pd.DataFrame, max_dist_pips: int = 15) -> pd.DataFrame:
    """
    Calculates Asian Session Extremes and detects sweeps into 4H FVGs.
    Requires calculate_multi_tf_fvgs to be run first.
    """
    PIP = 0.0001
    tolerance = max_dist_pips * PIP
    
    # 1. Define Asian Session (00:00 to 08:00 Kyiv Time)
    is_asia = (df['UA_Hour'] >= 0) & (df['UA_Hour'] <= 10)
    
    # 2. Get daily Asian High/Low
    df['Date_Key'] = df.index.date
    asia_stats = df[is_asia].groupby('Date_Key').agg({'High': 'max', 'Low': 'min'})
    asia_stats.rename(columns={'High': 'Asia_High', 'Low': 'Asia_Low'}, inplace=True)
    
    # 3. Merge back to the main dataframe
    df = df.join(asia_stats, on='Date_Key')
    
    # 4. Define the Setup: FVG must be "immediately outside" the Asian extreme
    # Bullish: 4H FVG Top is just below the Asian Low
    df['Bull_FVG_Below_AL'] = (df['FVG_4h_Type'] == 'BULL') & \
                              (df['Asia_Low'] >= df['FVG_4h_Top']) & \
                              ((df['Asia_Low'] - df['FVG_4h_Top']) <= tolerance)
                              
    # Bearish: 4H FVG Bottom is just above the Asian High
    df['Bear_FVG_Above_AH'] = (df['FVG_4h_Type'] == 'BEAR') & \
                              (df['Asia_High'] <= df['FVG_4h_Bottom']) & \
                              ((df['FVG_4h_Bottom'] - df['Asia_High']) <= tolerance)
                              
    # 5. Detect the Sweep
    # The sweep happens if the current candle breaks the Asian extreme AND enters the FVG
    df['Swept_AL_Into_FVG'] = df['Bull_FVG_Below_AL'] & (df['Low'] < df['Asia_Low']) & (df['Low'] <= df['FVG_4h_Top'])
    df['Swept_AH_Into_FVG'] = df['Bear_FVG_Above_AH'] & (df['High'] > df['Asia_High']) & (df['High'] >= df['FVG_4h_Bottom'])
    
    # Clean up and format
    df.drop(columns=['Date_Key'], inplace=True)
    df['Swept_AL_Into_FVG'] = df['Swept_AL_Into_FVG'].fillna(False).astype(int)
    df['Swept_AH_Into_FVG'] = df['Swept_AH_Into_FVG'].fillna(False).astype(int)
    
    return df

def add_london_pdh_pdl_sweep_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects if the London session sweeps the Previous Day High/Low 
    and prints a confirmed fractal at that exact sweep.
    Requires add_previous_boundaries and add_confirmed_fractals.
    """
    # 1. London Evaluation Window (10:00 - 14:00 Kyiv Time)
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    
    # 2. Safely extract fractal signals
    fractal_low = (df['Confirmed_Fractal_Low'].fillna(0) == 1)
    fractal_high = (df['Confirmed_Fractal_High'].fillna(0) == 1)
    
    # 3. Check if the fractal price actually swept the PDL/PDH
    # Confirmed_Fractal_Low_Price stores the exact price of the wick that formed the valley
    sweep_pdl = df['Confirmed_Fractal_Low_Price'] < df['PDL']
    sweep_pdh = df['Confirmed_Fractal_High_Price'] > df['PDH']
    
    # 4. Trap Logic
    # We LONG if London sweeps the PDL and forms a Fractal Low (Fake breakdown)
    df['LDN_Sweep_PDL_Long'] = is_london_eval & fractal_low & sweep_pdl
    
    # We SHORT if London sweeps the PDH and forms a Fractal High (Fake breakout)
    df['LDN_Sweep_PDH_Short'] = is_london_eval & fractal_high & sweep_pdh
    
    # 5. Strictly ONE trigger per day (Bulletproof Cumulative Sum)
    df['Date'] = df.index.date
    df['Sweep_Trigger_Count'] = (df['LDN_Sweep_PDL_Long'] | df['LDN_Sweep_PDH_Short']).groupby(df['Date']).cumsum()
    
    df['First_LDN_PDL_Long'] = (df['LDN_Sweep_PDL_Long'] & (df['Sweep_Trigger_Count'] == 1)).astype(int)
    df['First_LDN_PDH_Short'] = (df['LDN_Sweep_PDH_Short'] & (df['Sweep_Trigger_Count'] == 1)).astype(int)
    
    # Clean up intermediate tracking columns
    df.drop(columns=['Date', 'Sweep_Trigger_Count', 'LDN_Sweep_PDL_Long', 'LDN_Sweep_PDH_Short'], inplace=True)
    return df

def add_fvg_order_flow_context(df: pd.DataFrame) -> pd.DataFrame:
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

def add_weekly_swing_context(df: pd.DataFrame) -> pd.DataFrame:
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

def add_ny_sr_touch_context(df: pd.DataFrame, tolerance_pips: int = 10) -> pd.DataFrame:
    """
    Calculates Major S&R (20-Day Extremes) and detects the strictly FIRST touch 
    during the New York session, guaranteed once per day.
    """
    PIP = 0.0001
    tol = tolerance_pips * PIP
    
    # 1. Define Major S&R
    df['Major_Resistance'] = df['High'].rolling(window=24*20).max().shift(1)
    df['Major_Support'] = df['Low'].rolling(window=24*20).min().shift(1)
    
    # 2. Isolate NY Session
    is_ny = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 22)
    
    # 3. Detect Raw Touches
    df['NY_Touch_Res'] = is_ny & (df['High'] >= (df['Major_Resistance'] - tol))
    df['NY_Touch_Sup'] = is_ny & (df['Low'] <= (df['Major_Support'] + tol))
    
    # 4. THE BULLETPROOF FIX: Count touches per day
    df['Date'] = df.index.date
    df['Res_Daily_Touches'] = df.groupby('Date')['NY_Touch_Res'].cumsum()
    df['Sup_Daily_Touches'] = df.groupby('Date')['NY_Touch_Sup'].cumsum()
    
    # 5. Trigger ONLY on the exact 1st touch of the day
    df['NY_First_Touch_Res'] = df['NY_Touch_Res'] & (df['Res_Daily_Touches'] == 1)
    df['NY_First_Touch_Sup'] = df['NY_Touch_Sup'] & (df['Sup_Daily_Touches'] == 1)
    
    # 6. Clean up
    df.drop(columns=['Date', 'Res_Daily_Touches', 'Sup_Daily_Touches'], inplace=True)
    df['NY_First_Touch_Res'] = df['NY_First_Touch_Res'].astype(int)
    df['NY_First_Touch_Sup'] = df['NY_First_Touch_Sup'].astype(int)
    
    return df

def add_ny_expansion_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies if the NY session opened inside the Asian Range.
    Requires add_asian_sweep_context to be run first.
    """
    # 1. Identify NY Open (15:00 Kyiv time)
    # We use a ffill to carry the 15:00 price through the rest of the day
    df['NY_Open_Price'] = df.where(df['UA_Hour'] == 15)['Open']
    df['NY_Open_Price'] = df.groupby(df.index.date)['NY_Open_Price'].ffill()

    # 2. Check if NY Open is inside Asian Range
    # We use the Asia_High and Asia_Low calculated in add_asian_sweep_context
    df['NY_Opened_In_Asia_Range'] = (df['NY_Open_Price'] < df['Asia_High']) & \
                                     (df['NY_Open_Price'] > df['Asia_Low'])
    
    # 3. Detect the Sweep (Current High/Low breaks Asian Extreme during NY)
    is_ny_session = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 23)
    
    # Change the sweep detection to only trigger on the CROSS
    # This checks if the CURRENT candle is below Asia_Low, but the PREVIOUS one was not.
    df['NY_Sweep_Asia_Low'] = is_ny_session & \
                            (df['Low'] < df['Asia_Low']) & \
                            (df['Low'].shift(1) >= df['Asia_Low'].shift(1))

    df['NY_Sweep_Asia_High'] = is_ny_session & \
                            (df['High'] > df['Asia_High']) & \
                            (df['High'].shift(1) <= df['Asia_High'].shift(1))

    # Convert to 1/0 for the Parser
    df['NY_Opened_In_Asia_Range'] = df['NY_Opened_In_Asia_Range'].fillna(False).astype(int)
    df['NY_Sweep_Asia_High'] = df['NY_Sweep_Asia_High'].astype(int)
    df['NY_Sweep_Asia_Low'] = df['NY_Sweep_Asia_Low'].astype(int)
    
    return df

def add_1w_swing_context(df: pd.DataFrame) -> pd.DataFrame:
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

def add_1d_swing_context(df: pd.DataFrame) -> pd.DataFrame:
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

def add_weekly_floor_context(df: pd.DataFrame) -> pd.DataFrame:
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

def add_london_counter_fractal_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects when a London fractal forms AGAINST the 1D trend (a Liquidity Trap).
    """
    # 1. London Evaluation Window
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    
    # 2. Extract states securely
    bull_trend = (df['1D_Swing_Bullish'] == 1)
    bear_trend = (df['1D_Swing_Bearish'] == 1)
    
    fractal_low = (df['Confirmed_Fractal_Low'] == 1)
    fractal_high = (df['Confirmed_Fractal_High'] == 1)

    # 3. Trap Logic
    df['LDN_Counter_Low_Trap'] = is_london_eval & bear_trend & fractal_low
    df['LDN_Counter_High_Trap'] = is_london_eval & bull_trend & fractal_high

    # 4. Strictly one trigger per day
    df['Date'] = df.index.date
    df['Trap_Trigger_Count'] = (df['LDN_Counter_Low_Trap'] | df['LDN_Counter_High_Trap']).groupby(df['Date']).cumsum()
    
    df['First_LDN_Counter_Low'] = (df['LDN_Counter_Low_Trap'] & (df['Trap_Trigger_Count'] == 1)).astype(int)
    df['First_LDN_Counter_High'] = (df['LDN_Counter_High_Trap'] & (df['Trap_Trigger_Count'] == 1)).astype(int)
    
    # Clean up AFTER printing
    df.drop(columns=['Date', 'Trap_Trigger_Count', 'LDN_Counter_Low_Trap', 'LDN_Counter_High_Trap'], inplace=True)
    return df

def add_fvg_sr_confluence_context(df: pd.DataFrame, max_dist_pips: int = 30) -> pd.DataFrame:
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

def add_asian_sr_alignment_context(df: pd.DataFrame, max_dist_pips: int = 15) -> pd.DataFrame:
    """
    Определяет слияние Азиатских экстремумов и Макро-уровней (S&R).
    Фиксирует первый ложный пробой этой зоны после открытия Лондона.
    Требует выполнения add_asian_sweep_context и add_ny_sr_touch_context до нее.
    """
    PIP = 0.0001
    dist = max_dist_pips * PIP

    # 1. Проверяем "Идеальное совпадение" (Alignment)
    # Азиатский Хай совпадает с Макро-Сопротивлением
    df['Asia_Res_Aligned'] = (abs(df['Asia_High'] - df['Major_Resistance']) <= dist)
    
    # Азиатский Лоу совпадает с Макро-Поддержкой
    df['Asia_Sup_Aligned'] = (abs(df['Asia_Low'] - df['Major_Support']) <= dist)

    # 2. Ищем пробой (только в активные часы: Лондон и Нью-Йорк, с 10:00 до 21:00)
    is_active_session = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 21)

    # Цена пробивает сдвоенный Хай вверх (готовимся шортить)
    df['Break_Aligned_Res'] = is_active_session & df['Asia_Res_Aligned'] & (df['High'] > df['Asia_High'])
    
    # Цена пробивает сдвоенный Лоу вниз (готовимся лонговать)
    df['Break_Aligned_Sup'] = is_active_session & df['Asia_Sup_Aligned'] & (df['Low'] < df['Asia_Low'])

    # 3. Бронебойный фикс: берем только ПЕРВЫЙ пробой за день
    df['Date'] = df.index.date
    df['Res_Break_Count'] = df.groupby('Date')['Break_Aligned_Res'].cumsum()
    df['Sup_Break_Count'] = df.groupby('Date')['Break_Aligned_Sup'].cumsum()

    df['First_False_Break_Res'] = df['Break_Aligned_Res'] & (df['Res_Break_Count'] == 1)
    df['First_False_Break_Sup'] = df['Break_Aligned_Sup'] & (df['Sup_Break_Count'] == 1)

    # 4. Очистка
    df.drop(columns=['Date', 'Res_Break_Count', 'Sup_Break_Count'], inplace=True)
    df['First_False_Break_Res'] = df['First_False_Break_Res'].fillna(False).astype(int)
    df['First_False_Break_Sup'] = df['First_False_Break_Sup'].fillna(False).astype(int)

    return df

def add_asia_fvg_protection_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Определяет, находится ли Азиатский Хай/Лоу внутри 4H FVG.
    Запускает триггер на продолжение тренда ровно в 10:00 (London Open).
    """
    # 1. Проверяем, находится ли Азиатский Лоу внутри бычьего 4H FVG
    df['Asia_Low_Protected'] = (df['FVG_4h_Type'] == 'BULL') & \
                               (df['Asia_Low'] <= df['FVG_4h_Top']) & \
                               (df['Asia_Low'] >= df['FVG_4h_Bottom'])

    # 2. Проверяем, находится ли Азиатский Хай внутри медвежьего 4H FVG
    df['Asia_High_Protected'] = (df['FVG_4h_Type'] == 'BEAR') & \
                                (df['Asia_High'] <= df['FVG_4h_Top']) & \
                                (df['Asia_High'] >= df['FVG_4h_Bottom'])

    # 3. Триггер ровно на первой свече открытия Лондона (10:00 Киев)
    is_london_open = (df['UA_Hour'] == 10)

    # Лонг, если Азиатский Лоу защищен
    df['LDN_Protected_AL_Long'] = is_london_open & df['Asia_Low_Protected']
    
    # Шорт, если Азиатский Хай защищен
    df['LDN_Protected_AH_Short'] = is_london_open & df['Asia_High_Protected']

    # Очистка и форматирование (1/0 для парсера)
    df['LDN_Protected_AL_Long'] = df['LDN_Protected_AL_Long'].fillna(False).astype(int)
    df['LDN_Protected_AH_Short'] = df['LDN_Protected_AH_Short'].fillna(False).astype(int)

    df.drop(columns=['Asia_Low_Protected', 'Asia_High_Protected'], inplace=True)
    return df

def add_1w_level_rejection_context(df: pd.DataFrame, max_dist_pips: int = 20) -> pd.DataFrame:
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

def add_geopolitical_shock_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Сканирует реестр макро-событий и ставит триггер '1' на той свече,
    когда новость ударила по рынку (строго в UTC).
    """
    df['Geo_Shock_Short'] = 0
    
    # Фильтруем только геополитические шоки
    shock_events = [e for e in events if e.get('category') == 'Geopolitical_Shock']
    
    for event in shock_events:
        try:
            dt = event['start_date']
            
            # Мастер-индекс (df.index) находится в UTC!
            # Локализуем время события в UTC для идеального совпадения.
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
            
            # Округляем до часа, чтобы совпало со свечой 1H
            rounded_dt = dt.floor('h')
            
            # Ищем точное совпадение
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'Geo_Shock_Short'] = 1
        except Exception as e:
            print(f"Skipping event {event.get('name')} due to error: {e}")
            
    # Принудительно конвертируем в int, чтобы JSON парсер (== 1) сработал идеально
    df['Geo_Shock_Short'] = df['Geo_Shock_Short'].fillna(0).astype(int)
            
    return df

def add_election_volatility_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Рассчитывает реализованную волатильность (4H ATR) и ставит триггер на Шорт 
    в момент публикации экзит-полов (Volatility Crush).
    """
    # 1. Считаем True Range (TR) и 4H Реализованную Волатильность
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['Realized_Vol'] = df['TR'].rolling(window=4).mean() * 10000

    df['Election_Vol_Crush_Short'] = 0
    
    # 2. Ищем выборы в реестре
    elections = [e for e in events if e.get('category') == 'Elections']
    
    for event in elections:
        try:
            dt = pd.to_datetime(event['start_date'])
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
                
            rounded_dt = dt.floor('h')
            
            # Ставим сигнал на T=0
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'Election_Vol_Crush_Short'] = 1
        except Exception as e:
            print(f"Skipping event {event.get('name')} due to error: {e}")
            
    df['Election_Vol_Crush_Short'] = df['Election_Vol_Crush_Short'].fillna(0).astype(int)
    return df

def add_uk_political_shock_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Отслеживает негативные политические шоки UK.
    Ждет 1 час (пока паника retail-трейдеров утихнет) и дает сигнал
    на продолжение доминирующего 4H тренда.
    """
    df['UK_Shock_T0'] = 0
    
    # 1. Фильтруем события политических шоков
    shock_events = [e for e in events if e.get('category') == 'UK_Political_Shock']
    
    for event in shock_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
                
            rounded_dt = dt.floor('h')
            
            # Ставим маркер на час выхода новости
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'UK_Shock_T0'] = 1
        except Exception as e:
            print(f"Skipping event {event.get('name')} due to error: {e}")
            
    # 2. СДВИГ ВРЕМЕНИ (Магия гипотезы):
    # Мы ждем ровно 1 час (T+1), пока первая свеча закроется.
    df['UK_Shock_T1'] = df['UK_Shock_T0'].shift(1).fillna(0)
    
    # 3. Фильтрация по тренду (с запасом для Trend Guard)
    # Заходим в Лонг на T+1, если 4H тренд был Бычьим
    df['UK_Shock_Cont_Long'] = ((df['UK_Shock_T1'] == 1) & (df['HTF_Bullish_Prob'] >= 55)).astype(int)
    
    # Заходим в Шорт на T+1, если 4H тренд был Медвежьим
    df['UK_Shock_Cont_Short'] = ((df['UK_Shock_T1'] == 1) & (df['HTF_Bullish_Prob'] <= 45)).astype(int)
    
    # Убираем временные колонки, оставляем только готовые сигналы
    df.drop(columns=['UK_Shock_T0', 'UK_Shock_T1'], inplace=True)
    
    return df

def add_boe_hawkish_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Отслеживает 'ястребиные' сюрпризы Банка Англии (BoE).
    Генерирует сигнал на покупку (Long) для торговли Post-Announcement Drift.
    """
    df['BoE_Hawkish_T0'] = 0
    
    hawkish_events = [e for e in events if e.get('category') == 'BoE_Hawkish_Shock']
    
    for event in hawkish_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'BoE_Hawkish_T0'] = 1
        except Exception as e:
            print(f"Skipping event {event.get('name')} due to error: {e}")
            
    # Конвертируем в Int для парсера
    df['BoE_Hawkish_Long'] = df['BoE_Hawkish_T0'].fillna(0).astype(int)
    
    # Очистка
    df.drop(columns=['BoE_Hawkish_T0'], inplace=True)
    return df

def add_uk_cpi_momentum_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Отслеживает шоки UK CPI с дельтой > 0.3%.
    Ждет закрытия первой 1-часовой свечи (T=0). 
    Если T=0 бычья -> сигнал Long на T+1. Если T=0 медвежья -> сигнал Short на T+1.
    """
    df['CPI_Release_T0'] = 0
    
    cpi_events = [e for e in events if e.get('category') == 'UK_CPI_Shock']
    
    for event in cpi_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'CPI_Release_T0'] = 1
        except Exception as e:
            pass
            
    # 1. We look 4 hours back to see if the news happened 4 hours ago
    df['CPI_T4_Active'] = df['CPI_Release_T0'].shift(4).fillna(0)
    
    # 2. Determine direction: Compare current Close (after 4h) 
    # to the Open price when the news actually broke (4 hours ago)
    df['T4_Direction'] = np.where(df['Close'] >= df['Open'].shift(3), 1, -1)
    
    # 3. Signals now trigger only after the 4th hour closes
    df['CPI_Momentum_Long'] = ((df['CPI_T4_Active'] == 1) & (df['T4_Direction'] == 1)).astype(int)
    df['CPI_Momentum_Short'] = ((df['CPI_T4_Active'] == 1) & (df['T4_Direction'] == -1)).astype(int)
    
    # Очистка
    df.drop(columns=['CPI_Release_T0', 'CPI_T4_Active', 'T4_Direction'], inplace=True)
    return df

def add_sovereign_risk_proxy_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Ищет режим Sovereign Risk (Идиосинкразическая паника).
    Условие: Политический шок + 4H ATR превышает 30-дневную норму в 2.5 раза.
    Действие: Продаем после того, как сформировался 4-часовой восходящий отскок (Fade the 4H bounce).
    """
    # ФИКС ПАМЯТИ: Дефрагментируем DataFrame перед добавлением кучи новых колонок
    df = df.copy()
    
    # 1. Считаем True Range (TR) и 4H ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['ATR_4H'] = df['TR'].rolling(window=4).mean()
    df['ATR_30D_Avg'] = df['ATR_4H'].rolling(window=720).mean()
    df['Vol_Anomaly'] = df['ATR_4H'] > (df['ATR_30D_Avg'] * 2.5)
    
    # 2. Размечаем окно Политического Шока
    df['Pol_Shock_Active'] = 0
    uk_shocks = [e for e in events if e.get('category') == 'UK_Political_Shock']
    
    for event in uk_shocks:
        try:
            start_dt = pd.to_datetime(event['start_date'])
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            
            end_dt = pd.to_datetime(event['end_date'])
            end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
            
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            df.loc[mask, 'Pol_Shock_Active'] = 1
        except Exception as e:
            pass
            
    # 3. ЛОГИКА 4H ОТСКОКА: Вместо одной зеленой свечи, мы смотрим,
    # выросла ли цена за последние 4 часа (Текущий Close > Open 3 свечи назад)
    df['Is_4H_Bounce'] = df['Close'] > df['Open'].shift(3)
    
    # 4. ГЕНЕРАЦИЯ СИГНАЛА НА T+1
    trigger_condition = (
        (df['Pol_Shock_Active'].shift(1) == 1) & 
        (df['Vol_Anomaly'].shift(1) == True) & 
        (df['Is_4H_Bounce'].shift(1) == True)
    )
    
    df['Sovereign_Risk_Short'] = trigger_condition.astype(int)
    
    # Очистка мусора
    df.drop(columns=['TR', 'ATR_4H', 'ATR_30D_Avg', 'Vol_Anomaly', 'Pol_Shock_Active', 'Is_4H_Bounce'], inplace=True, errors='ignore')
    
    return df

def add_weekend_gap_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит выходные дни (разрыв между свечами > 24 часов).
    Считает размер гэпа открытия в пипсах.
    Генерирует сигналы на возврат к среднему (Gap Fill) для гэпов > 40 пипсов.
    """
    # 1. Находим разницу во времени между свечами
    time_diff = df.index.to_series().diff()
    
    # 2. Любой разрыв больше 24 часов - это открытие недели (Sunday/Monday Open)
    is_week_open = time_diff > pd.Timedelta(hours=24)
    
    # 3. Считаем размер гэпа в пипсах (Open текущей свечи минус Close пятницы)
    gap_pips = (df['Open'] - df['Close'].shift(1)) * 10000
    
    # 4. ТРИГГЕРЫ (Торгуем ПРОТИВ гэпа на его закрытие)
    # Если гэп ВВЕРХ > 40 пипсов, цена перекуплена на панике -> ШОРТИМ
    df['Gap_Up_Fade_Short'] = ((is_week_open) & (gap_pips >= 40)).astype(int)
    
    # Если гэп ВНИЗ > 40 пипсов (gap_pips <= -40) -> ЛОНГУЕМ
    df['Gap_Down_Fade_Long'] = ((is_week_open) & (gap_pips <= -40)).astype(int)
    
    return df

def add_boe_tone_shift_proxy_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    NLP Прокси: Ищет расхождение между текстом релиза и риторикой Губернатора.
    Условие: T=0 (Стейтмент) - зеленая свеча. T+1 (Пресс-конференция) - красная свеча.
    Действие: Шортим на закрытии T+1 (Fade the rally).
    """
    df = df.copy()
    df['BoE_Release_T0'] = 0
    
    # 1. Используем наш существующий список ястребиных шоков
    boe_events = [e for e in events if e.get('category') == 'BoE_Hawkish_Shock']
    
    for event in boe_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'BoE_Release_T0'] = 1
        except Exception as e:
            pass
            
    # 2. Логика Tone Shift (Отпечаток NLP)
    # Направление текущей свечи
    df['Is_Green'] = df['Close'] > df['Open']
    df['Is_Red'] = df['Close'] < df['Open']
    
    # Смещаем данные, чтобы смотреть в прошлое на момент закрытия свечи пресс-конференции
    df['Was_Release_T1'] = df['BoE_Release_T0'].shift(1).fillna(0)
    df['Was_T1_Green'] = df['Is_Green'].shift(1).fillna(False)
    
    # 3. ТРИГГЕР: Час назад был релиз и он был зеленым, но ТЕКУЩАЯ свеча (выступление) красная!
    trigger_condition = (
        (df['Was_Release_T1'] == 1) & 
        (df['Was_T1_Green'] == True) & 
        (df['Is_Red'] == True)
    )
    
    df['BoE_Tone_Shift_Short'] = trigger_condition.astype(int)
    
    # Очистка
    df.drop(columns=['BoE_Release_T0', 'Is_Green', 'Is_Red', 'Was_Release_T1', 'Was_T1_Green'], inplace=True, errors='ignore')
    
    return df

def add_macro_shock_inside_bar_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    ГИПОТЕЗА А (Макро): Ищет спайк >3 сигмы строго в первые 24 часа после внезапной новости.
    """
    df = df.copy()
    
    # Считаем метрики волатильности
    df['TR_Local'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['TR_Mean'] = df['TR_Local'].rolling(window=720).mean()
    df['TR_Std'] = df['TR_Local'].rolling(window=720).std()
    df['Is_3SD_Spike'] = df['TR_Local'] > (df['TR_Mean'] + 3 * df['TR_Std'])
    
    # Жесткий фильтр: ТОЛЬКО 24 часа после старта события (Игнорируем end_date)
    df['Strict_Macro_Window'] = 0
    unscheduled_categories = ['Geopolitical_Shock', 'UK_Political_Shock']
    target_events = [e for e in events if e.get('category') in unscheduled_categories]
    
    for event in target_events:
        try:
            start_dt = pd.to_datetime(event['start_date'])
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            
            # ФИКС БАГА: Принудительно задаем окно ровно в 24 часа
            strict_end_dt = start_dt + pd.Timedelta(hours=24)
            
            mask = (df.index >= start_dt) & (df.index <= strict_end_dt)
            df.loc[mask, 'Strict_Macro_Window'] = 1
        except Exception as e:
            pass
            
    if 'Realized_Vol' not in df.columns:
        df['Realized_Vol'] = df['TR_Local'].rolling(window=4).mean() * 10000
            
    # Триггер: В окне макро-события случился аномальный спайк
    trigger_condition = (
        (df['Strict_Macro_Window'].shift(1) == 1) & 
        (df['Is_3SD_Spike'].shift(1) == True)
    )
    
    df['Macro_Inside_Bar_Short'] = trigger_condition.astype(int)
    
    df.drop(columns=['TR_Local', 'TR_Mean', 'TR_Std', 'Is_3SD_Spike', 'Strict_Macro_Window'], inplace=True, errors='ignore')
    return df

def add_pure_algo_vol_crush_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    ГИПОТЕЗА Б (Algo): Ищет спайк >3 сигмы в ЛЮБОЙ день (без привязки к макро-календарю)
    и шортит волатильность на возврат к среднему (Mean Reversion).
    """
    df = df.copy()
    
    df['TR_Local'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['TR_Mean'] = df['TR_Local'].rolling(window=720).mean()
    df['TR_Std'] = df['TR_Local'].rolling(window=720).std()
    
    # Триггер: Просто любой аномальный спайк
    df['Is_3SD_Spike'] = df['TR_Local'] > (df['TR_Mean'] + 3 * df['TR_Std'])
    
    if 'Realized_Vol' not in df.columns:
        df['Realized_Vol'] = df['TR_Local'].rolling(window=4).mean() * 10000
        
    df['Algo_Vol_Crush_Short'] = df['Is_3SD_Spike'].shift(1).fillna(0).astype(int)

    df.drop(columns=['TR_Local', 'TR_Mean', 'TR_Std', 'Is_3SD_Spike'], inplace=True, errors='ignore')
    return df

def add_nfp_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df = df.copy()
    df['NFP_Level'] = np.nan
    
    # ФИКС ЗДЕСЬ: Было 0, стало np.nan. Теперь ffill сможет протянуть данные!
    df['NFP_Initial_Dir'] = np.nan 

    nfp_events = [e for e in events if e.get('category') == 'US_NFP_Divergence']
    matched_events = 0

    for event in nfp_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            if rounded_dt in df.index:
                matched_events += 1
                df.loc[rounded_dt, 'NFP_Level'] = df.loc[rounded_dt, 'Open']
                # Сохраняем 1 (рост) или -1 (падение)
                df.loc[rounded_dt, 'NFP_Initial_Dir'] = 1 if df.loc[rounded_dt, 'Close'] > df.loc[rounded_dt, 'Open'] else -1
        except Exception as e:
            pass

    # Протягиваем уровень и направление вперед на 5 часов (теперь это сработает!)
    df['NFP_Level'] = df['NFP_Level'].ffill(limit=5)
    df['NFP_Initial_Dir'] = df['NFP_Initial_Dir'].ffill(limit=5)

    # УСЛОВИЕ ВХОДА (Fade подтвержден пробоем уровня открытия)
    df['NFP_Fade_Long'] = ((df['NFP_Initial_Dir'] == -1) & (df['Close'] > df['NFP_Level'])).astype(int)
    df['NFP_Fade_Short'] = ((df['NFP_Initial_Dir'] == 1) & (df['Close'] < df['NFP_Level'])).astype(int)

    # Берем только ПЕРВОЕ пересечение уровня
    df['NFP_Fade_Long'] = ((df['NFP_Fade_Long'] == 1) & (df['NFP_Fade_Long'].shift(1) == 0)).astype(int)
    df['NFP_Fade_Short'] = ((df['NFP_Fade_Short'] == 1) & (df['NFP_Fade_Short'].shift(1) == 0)).astype(int)

    # Очистка
    df.drop(columns=['NFP_Level', 'NFP_Initial_Dir'], inplace=True, errors='ignore')
    return df

def add_nfp_revision_trap_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Торгует ловушки NFP (Beat + Negative Revision).
    Ждет возврата цены к уровню открытия новости в течение 4 часов.
    Оценивает максимальный профит (MFE) за последующие 6 часов.
    """
    df = df.copy()
    df['Trap_Level'] = np.nan
    df['Trap_Initial_Dir'] = np.nan 

    trap_events = [e for e in events if e.get('category') == 'US_NFP_Revision_Trap']
    matched_events = 0

    for event in trap_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            if rounded_dt in df.index:
                matched_events += 1
                df.loc[rounded_dt, 'Trap_Level'] = df.loc[rounded_dt, 'Open']
                df.loc[rounded_dt, 'Trap_Initial_Dir'] = 1 if df.loc[rounded_dt, 'Close'] > df.loc[rounded_dt, 'Open'] else -1
        except Exception as e:
            pass

    # Протягиваем уровень и направление вперед ровно на 4 ЧАСА
    df['Trap_Level'] = df['Trap_Level'].ffill(limit=4)
    df['Trap_Initial_Dir'] = df['Trap_Initial_Dir'].ffill(limit=4)

    # ПЕРЕВЕРНУТАЯ ЛОГИКА (TREND RESUMPTION)
    # Исходный импульс ВНИЗ (-1) + Откат вверх (Close > Level) = ШОРТ
    df['NFP_Resumption_Short'] = ((df['Trap_Initial_Dir'] == -1) & (df['Close'] > df['Trap_Level'])).astype(int)

    # Исходный импульс ВВЕРХ (1) + Откат вниз (Close < Level) = ЛОНГ
    df['NFP_Resumption_Long'] = ((df['Trap_Initial_Dir'] == 1) & (df['Close'] < df['Trap_Level'])).astype(int)

    # Защита от дублей (берем только первый пробой)
    df['NFP_Resumption_Short'] = ((df['NFP_Resumption_Short'] == 1) & (df['NFP_Resumption_Short'].shift(1) == 0)).astype(int)
    df['NFP_Resumption_Long'] = ((df['NFP_Resumption_Long'] == 1) & (df['NFP_Resumption_Long'].shift(1) == 0)).astype(int)

    # Очистка
    df.drop(columns=['Trap_Level', 'Trap_Initial_Dir'], inplace=True, errors='ignore')
    
    return df

def add_cpi_match_mean_reversion_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Отрабатывает гипотезу "Volatility Crush" на CPI.
    Если CPI совпадает с прогнозом, цена возвращается к открытию Лондона (08:00 UTC).
    """
    df = df.copy()
    df['CPI_Match_Fade_Short'] = 0
    df['CPI_Match_Fade_Long'] = 0
    
    cpi_events = [e for e in events if e.get('category') == 'US_CPI_Match']
    matched_events = 0

    for event in cpi_events:
        try:
            # Время релиза CPI
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            # Находим время открытия Лондона (08:00 UTC) в ТОТ ЖЕ ДЕНЬ
            date_str = rounded_dt.strftime('%Y-%m-%d')
            london_open_dt = pd.to_datetime(f"{date_str} 08:00:00").tz_localize('UTC')
            
            if rounded_dt in df.index and london_open_dt in df.index:
                matched_events += 1
                london_anchor_price = df.loc[london_open_dt, 'Open']
                pre_cpi_price = df.loc[rounded_dt, 'Open']
                
                # ЛОГИКА ВОЗВРАТА:
                # Если цена выросла с открытия Лондона до CPI -> Шортим обратно к якорю
                if pre_cpi_price > london_anchor_price:
                    df.loc[rounded_dt, 'CPI_Match_Fade_Short'] = 1
                
                # Если цена упала с открытия Лондона до CPI -> Лонгуем обратно к якорю
                elif pre_cpi_price < london_anchor_price:
                    df.loc[rounded_dt, 'CPI_Match_Fade_Long'] = 1
                    
        except Exception as e:
            pass
            

    return df

def add_cb_divergence_state_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df = df.copy()
    df['Fed_State'] = np.nan
    df['BoE_State'] = np.nan
    df['Divergence_Trigger'] = 0
    
    # 1. Парсим ФРС
    fed_events = [e for e in events if e.get('category') == 'Fed_Significant_Probability_Shift']
    for event in fed_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            dt = dt.floor('h')
            if dt in df.index:
                # Читаем контекст из названия
                name = event['name'].lower()
                if 'cut prob from' in name and 'drops' not in name or 'emergency cut' in name or 'pivot' in name:
                    df.loc[dt, 'Fed_State'] = -1 # Dovish
                else:
                    df.loc[dt, 'Fed_State'] = 1  # Hawkish
        except: pass

    # 2. Парсим Банк Англии
    boe_events = [e for e in events if e.get('category') == 'BoE_Significant_Probability_Shift']
    for event in boe_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            dt = dt.floor('h')
            if dt in df.index:
                name = event['name'].lower()
                if 'cut prob' in name and 'collapses' not in name or 'dovish' in name:
                    df.loc[dt, 'BoE_State'] = -1 # Dovish
                else:
                    df.loc[dt, 'BoE_State'] = 1  # Hawkish
        except: pass

    # 3. State Machine: Протягиваем состояния вперед (память рынка)
    df['Fed_State'] = df['Fed_State'].ffill()
    df['BoE_State'] = df['BoE_State'].ffill()

    # Создаем базовые "блоки" расхождений (1 = есть расхождение, 0 = нет)
    df['Is_Bullish_Div'] = ((df['BoE_State'] == 1) & (df['Fed_State'] == -1)).astype(int)
    df['Is_Bearish_Div'] = ((df['BoE_State'] == -1) & (df['Fed_State'] == 1)).astype(int)

    # 4. ТРИГГЕР (Защита от спама): 
    # Входим ТОЛЬКО в тот самый 1-й час, когда состояние переключилось с 0 на 1.
    df['CB_Divergence_Long'] = ((df['Is_Bullish_Div'] == 1) & (df['Is_Bullish_Div'].shift(1) == 0)).astype(int)
    df['CB_Divergence_Short'] = ((df['Is_Bearish_Div'] == 1) & (df['Is_Bearish_Div'].shift(1) == 0)).astype(int)

    # Очистка
    df.drop(columns=['Fed_State', 'BoE_State', 'Is_Bullish_Div', 'Is_Bearish_Div'], inplace=True, errors='ignore')
    return df

def add_fomc_sell_the_news_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Паттерн: Buy the Rumor, Sell the News (FOMC).
    Если ФРС повышает ставку ровно так, как ожидалось (Priced In),
    институционалы фиксируют лонги по USD. GBP/USD растет.
    """
    df = df.copy()
    df['FOMC_Sell_News_Long'] = 0
    
    fomc_events = [e for e in events if e.get('category') == 'US_FOMC_InLine_Hike']
    matched_events = 0

    for event in fomc_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            
            if rounded_dt in df.index:
                matched_events += 1
                # Триггер: Покупаем GBP/USD (продаем USD) на закрытии часа релиза
                df.loc[rounded_dt, 'FOMC_Sell_News_Long'] = 1
        except Exception as e:
            pass
    
    return df

def add_uk_us_cpi_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df = df.copy()
    
    # Создаем колонки только для МОМЕНТА выхода новости (будут заполнены только в 1 час)
    df['US_CPI_Release'] = np.nan
    df['UK_CPI_Release'] = np.nan
    
    # 1. Отмечаем ТОЛЬКО час выхода US CPI (Cold)
    us_events = [e for e in events if e.get('category') == 'US_CPI_Cold']
    for event in us_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'US_CPI_Release'] = -1
        except: pass

    # 2. Отмечаем ТОЛЬКО час выхода UK CPI (Hot)
    uk_events = [e for e in events if e.get('category') == 'UK_CPI_Hot']
    for event in uk_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'UK_CPI_Release'] = 1
        except: pass

    # 3. Создаем "Память" рынка на 30 дней (720 часов)
    # Эти колонки просто хранят статус последнего отчета, но НЕ используются как триггеры
    df['US_Cold_Memory'] = df['US_CPI_Release'].ffill(limit=720)
    df['UK_Hot_Memory'] = df['UK_CPI_Release'].ffill(limit=720)

    # 4. ЖЕЛЕЗОБЕТОННЫЙ ТРИГГЕР ВХОДА
    df['Macro_CPI_Div_Long'] = 0

    # Сценарий А: Сейчас вышел UK CPI (Hot), проверяем, был ли прошлый US CPI холодным
    uk_trigger_mask = (df['UK_CPI_Release'] == 1) & (df['US_Cold_Memory'] == -1)
    
    # Сценарий Б: Сейчас вышел US CPI (Cold), проверяем, был ли прошлый UK CPI горячим
    us_trigger_mask = (df['US_CPI_Release'] == -1) & (df['UK_Hot_Memory'] == 1)

    # Записываем сигнал только в час публикации новости
    df.loc[uk_trigger_mask | us_trigger_mask, 'Macro_CPI_Div_Long'] = 1

    # Очистка рабочего мусора
    df.drop(columns=['US_CPI_Release', 'UK_CPI_Release', 'US_Cold_Memory', 'UK_Hot_Memory'], inplace=True, errors='ignore')
    
    return df

def add_unemp_fakeout_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Торгует ловушку (Fakeout) на данных по безработице.
    Если безработица США растет, но GBP/USD падает в первый час (DXY rally),
    это ложное движение. Входим в лонг под пробой 4H максимума.
    """
    df = df.copy()
    
    # Вычисляем максимум за предыдущие 4 часа
    df['Prev_4H_High'] = df['High'].rolling(window=4, min_periods=1).max().shift(1)
    df['Unemp_Fakeout_Long'] = 0

    target_events = [e for e in events if e.get('category') == 'US_Unemp_Rise_UK_Stable']
    matched_events = 0

    for event in target_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')

            if rounded_dt in df.index:
                matched_events += 1
                
                # ЛОГИКА FAKEOUT: 
                # Новость для доллара плохая, но свеча GBP/USD закрылась в МИНУСЕ (Close < Open).
                # Это и есть та самая иррациональная защитная реакция рынка, которую мы выкупаем.
                if df.loc[rounded_dt, 'Close'] < df.loc[rounded_dt, 'Open']:
                    df.loc[rounded_dt, 'Unemp_Fakeout_Long'] = 1
                    
        except Exception as e:
            pass

    
    # Очистка
    df.drop(columns=['Prev_4H_High'], inplace=True, errors='ignore') 
    return df

def add_retail_sales_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    """
    Торгует дивергенцию розничных продаж (Retail Sales).
    Условие: US Retail Miss (потребитель США слаб) + UK Retail Beat (потребитель UK силен).
    """
    df = df.copy()
    
    # Колонки для точного часа релиза
    df['US_Retail_Release'] = np.nan
    df['UK_Retail_Release'] = np.nan
    
    us_events = [e for e in events if e.get('category') == 'US_Retail_Miss']
    for event in us_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index:
                df.loc[dt.floor('h'), 'US_Retail_Release'] = -1
        except: pass

    uk_events = [e for e in events if e.get('category') == 'UK_Retail_Beat']
    for event in uk_events:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index:
                df.loc[dt.floor('h'), 'UK_Retail_Release'] = 1
        except: pass

    # Память рынка (держим статус в уме 30 дней = 720 часов)
    df['US_Retail_Memory'] = df['US_Retail_Release'].ffill(limit=720)
    df['UK_Retail_Memory'] = df['UK_Retail_Release'].ffill(limit=720)

    # ЖЕЛЕЗОБЕТОННЫЙ ТРИГГЕР ВХОДА
    df['Retail_Div_Long'] = 0

    # Сценарий А: Сейчас вышел UK Beat, проверяем, был ли US Miss недавно
    uk_trigger_mask = (df['UK_Retail_Release'] == 1) & (df['US_Retail_Memory'] == -1)
    
    # Сценарий Б: Сейчас вышел US Miss, проверяем, был ли UK Beat недавно
    us_trigger_mask = (df['US_Retail_Release'] == -1) & (df['UK_Retail_Memory'] == 1)

    # Записываем сигнал ТОЛЬКО в час выхода новости
    df.loc[uk_trigger_mask | us_trigger_mask, 'Retail_Div_Long'] = 1

    # Очистка
    df.drop(columns=['US_Retail_Release', 'UK_Retail_Release', 'US_Retail_Memory', 'UK_Retail_Memory'], inplace=True, errors='ignore')
    
    return df

def add_friday_reversal_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    Торгует пятничную фиксацию прибыли (Institutional Risk-Off).
    Если с понедельника пара выросла более чем на 2 дневных ATR, 
    в пятницу в 16:00 UTC институционалы начнут закрывать лонги (шорт).
    Для симметрии добавлено условие и на лонг (выкуп сильного падения).
    """
    df = df.copy()
    df['Friday_Reversal_Short'] = 0
    df['Friday_Reversal_Long'] = 0
    
    # 1. Считаем грубый дневной ATR на часовиках
    # Берем размах (High - Low) за 24 часа и усредняем за 14 дней (336 часов)
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Находим цену открытия недели (Понедельник)
    # На рынке Форекс от понедельника 00:00 до пятницы 16:00 проходит ровно 112 часов торгов (без учета праздников).
    # Используем shift(112) как надежный прокси для старта недели.
    df['Week_Open'] = df['Open'].shift(112)
    
    # 3. Фильтр времени: Пятница (dayofweek == 4), 16:00 UTC
    is_friday_16 = (df.index.dayofweek == 4) & (df.index.hour == 16)
    
    # 4. Считаем пройденное расстояние за неделю
    weekly_rally = df['Close'] - df['Week_Open']
    weekly_dump = df['Week_Open'] - df['Close']
    
    # 5. ТРИГГЕРЫ: Движение должно быть больше 2 * ATR_14D
    short_condition = is_friday_16 & (weekly_rally > (2 * df['ATR_14D']))
    long_condition = is_friday_16 & (weekly_dump > (2 * df['ATR_14D']))
    
    df.loc[short_condition, 'Friday_Reversal_Short'] = 1
    df.loc[long_condition, 'Friday_Reversal_Long'] = 1
    
    # Очистка рабочего мусора
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Week_Open'], inplace=True, errors='ignore')
    
    return df

def add_monday_gap_reversion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_037: Monday Opening Gap Reversion.
    Если прошлая неделя закрылась с аномальным отклонением (> 2 ATR),
    открываем позицию на возврат в первый час понедельника.
    """
    df = df.copy()
    df['Monday_Reversion_Short'] = 0
    df['Monday_Reversion_Long'] = 0
    
    # 1. ATR для измерения "нормальности" движения (24ч * 14д)
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Результат недели: Цена сейчас (Пт 21:00 / Пн 00:00) минус Цена 5 дней назад
    # 120 часов — это полная торговая неделя
    df['Weekly_Delta'] = df['Close'] - df['Open'].shift(120)
    
    # 3. Условие: Понедельник, 00:00 UTC
    is_monday_open = (df.index.dayofweek == 0) & (df.index.hour == 0)
    
    # 4. ТРИГГЕРЫ (смотрим на Weekly_Delta относительно ATR на момент открытия)
    # Если за неделю выросли > 2 ATR -> Шортим в Пн 00:00
    short_cond = is_monday_open & (df['Weekly_Delta'] > (2 * df['ATR_14D']))
    # Если за неделю упали > 2 ATR -> Лонгуем в Пн 00:00
    long_cond = is_monday_open & (df['Weekly_Delta'] < -(2 * df['ATR_14D']))
    
    df.loc[short_cond, 'Monday_Reversion_Short'] = 1
    df.loc[long_cond, 'Monday_Reversion_Long'] = 1
    
    # Чистим колонки
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Weekly_Delta'], inplace=True, errors='ignore')
    
    return df

def add_turnaround_tuesday_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_038: Turnaround Tuesday (Trend Resumption).
    Если прошлая неделя была бычьей, а понедельник - медвежьим (откат),
    мы покупаем во вторник в 08:00 UTC (Лондон) на возобновление тренда.
    """
    df = df.copy()
    df['Tuesday_Resumption_Long'] = 0
    df['Tuesday_Resumption_Short'] = 0
    
    # 1. Измеряем тренд прошлой недели (120 часов)
    # Используем сдвиг на 144 часа (120ч неделя + 24ч понедельник), чтобы получить цену открытия прошлой недели
    df['Prev_Week_Open'] = df['Open'].shift(144)
    df['Prev_Week_Close'] = df['Close'].shift(24) # Закрытие прошлой пятницы = 24 часа назад от утра вторника
    
    # 2. Измеряем тренд Понедельника (последние 24 часа)
    df['Monday_Open'] = df['Open'].shift(24)
    df['Monday_Close'] = df['Close'].shift(1)
    
    # 3. Триггер времени: Вторник, 08:00 UTC (Открытие Лондона)
    is_tuesday_london_open = (df.index.dayofweek == 1) & (df.index.hour == 8)
    
    # 4. ЛОГИКА ВХОДА
    # ЛОНГ: Прошлая неделя бычья (Close > Open) И Понедельник медвежий (Close < Open) -> Покупаем откат
    long_cond = is_tuesday_london_open & (df['Prev_Week_Close'] > df['Prev_Week_Open']) & (df['Monday_Close'] < df['Monday_Open'])
    
    # ШОРТ: Прошлая неделя медвежья (Close < Open) И Понедельник бычий (Close > Open) -> Продаем отскок
    short_cond = is_tuesday_london_open & (df['Prev_Week_Close'] < df['Prev_Week_Open']) & (df['Monday_Close'] > df['Monday_Open'])
    
    df.loc[long_cond, 'Tuesday_Resumption_Long'] = 1
    df.loc[short_cond, 'Tuesday_Resumption_Short'] = 1
    
    df.drop(columns=['Prev_Week_Open', 'Prev_Week_Close', 'Monday_Open', 'Monday_Close'], inplace=True, errors='ignore')
    return df

def add_wednesday_fakeout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_039: Mid-Week Fakeout (Liquidity Sweep).
    Среда часто сбивает стопы за максимумами/минимумами Вторника.
    Если цена пробивает экстремум Вторника, но часовая свеча закрывается откатом (пинбар/поглощение) — входим на разворот.
    """
    df = df.copy()
    df['Wed_Fakeout_Long'] = 0
    df['Wed_Fakeout_Short'] = 0
    
    # 1. Вычисляем дневные максимумы и минимумы
    df['Date'] = df.index.date
    daily_hl = df.groupby('Date').agg({'High': 'max', 'Low': 'min'})
    
    # Сдвигаем на 1 день, чтобы получить данные "Вчерашнего дня"
    prev_day_hl = daily_hl.shift(1)
    
    # Мапим данные вчерашнего дня на каждый час текущего дня
    df['Prev_Day_High'] = df['Date'].map(prev_day_hl['High'])
    df['Prev_Day_Low'] = df['Date'].map(prev_day_hl['Low'])
    
    # 2. Изолируем Среду (dayofweek == 2)
    is_wednesday = df.index.dayofweek == 2
    
    # 3. ЛОГИКА ШОРТА (Sweep of High):
    # Цена поднималась ВЫШЕ максимума вторника, но часовая свеча закрылась НИЖЕ него, 
    # и сама свеча медвежья (Close < Open)
    sweep_high = (df['High'] > df['Prev_Day_High']) & (df['Close'] < df['Prev_Day_High']) & (df['Close'] < df['Open'])
    short_cond = is_wednesday & sweep_high
    
    # 4. ЛОГИКА ЛОНГА (Sweep of Low):
    # Цена падала НИЖЕ минимума вторника, но часовая свеча закрылась ВЫШЕ него,
    # и сама свеча бычья (Close > Open)
    sweep_low = (df['Low'] < df['Prev_Day_Low']) & (df['Close'] > df['Prev_Day_Low']) & (df['Close'] > df['Open'])
    long_cond = is_wednesday & sweep_low
    
    df.loc[short_cond, 'Wed_Fakeout_Short'] = 1
    df.loc[long_cond, 'Wed_Fakeout_Long'] = 1
    
    # Очистка рабочего мусора
    df.drop(columns=['Date', 'Prev_Day_High', 'Prev_Day_Low'], inplace=True, errors='ignore')
    return df

def add_thursday_expansion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_040: Thursday Trend Expansion.
    Если с Понедельника по Среду сформировался четкий тренд,
    Четверг на открытии Лондона (08:00 UTC) продолжит этот импульс.
    """
    df = df.copy()
    df['Thursday_Trend_Long'] = 0
    df['Thursday_Trend_Short'] = 0
    
    # 1. Считаем дневной ATR (для понимания волатильности)
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Находим цены Открытия Недели и Закрытия Среды
    # В четверг в 08:00 UTC открытие недели (понедельник 00:00) было ровно 80 часов назад
    df['Week_Open'] = df['Open'].shift(80)
    # Закрытие среды (23:00) было 9 часов назад относительно 08:00 четверга
    df['Wed_Close'] = df['Close'].shift(9) 
    
    # 3. Триггер времени: Четверг, 08:00 UTC
    is_thursday_london_open = (df.index.dayofweek == 3) & (df.index.hour == 8)
    
    # 4. ЛОГИКА ВХОДА: Тренд должен быть больше 0.5 * ATR (отсеиваем флэт)
    trend_up = (df['Wed_Close'] - df['Week_Open']) > (0.5 * df['ATR_14D'])
    trend_down = (df['Week_Open'] - df['Wed_Close']) > (0.5 * df['ATR_14D'])
    
    long_cond = is_thursday_london_open & trend_up
    short_cond = is_thursday_london_open & trend_down
    
    df.loc[long_cond, 'Thursday_Trend_Long'] = 1
    df.loc[short_cond, 'Thursday_Trend_Short'] = 1
    
    df.drop(columns=['Week_Open', 'Wed_Close', 'Daily_Range', 'ATR_14D'], inplace=True, errors='ignore')
    return df

def add_london_fix_fade_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_041: Friday London Fix Fade.
    В 16:00 UTC в пятницу происходит пик институциональной активности (Fix).
    Если неделя была экстремальной (> 1.5 ATR), после 16:00 происходит 
    откат из-за закрытия позиций перед выходными.
    """
    df = df.copy()
    df['Fix_Fade_Short'] = 0
    df['Fix_Fade_Long'] = 0
    
    # 1. Измеряем ATR за 2 недели
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Результат недели на текущий момент (Пт 16:00)
    # Считаем разницу между Пн 00:00 (112 часов назад) и сейчас
    df['Week_Performance'] = df['Close'] - df['Open'].shift(112)
    
    # 3. Триггер времени: Пятница, 16:00 UTC
    is_friday_fix = (df.index.dayofweek == 4) & (df.index.hour == 16)
    
    # 4. ЛОГИКА ВХОДА:
    # Если неделя выросла более чем на 1.5 ATR -> Шортим в 16:00
    short_cond = is_friday_fix & (df['Week_Performance'] > (1.5 * df['ATR_14D']))
    
    # Если неделя упала более чем на 1.5 ATR -> Лонгуем в 16:00
    long_cond = is_friday_fix & (df['Week_Performance'] < -(1.5 * df['ATR_14D']))
    
    df.loc[short_cond, 'Fix_Fade_Short'] = 1
    df.loc[long_cond, 'Fix_Fade_Long'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Week_Performance'], inplace=True, errors='ignore')
    return df

def add_tokyo_trap_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_043: The Tokyo Trap.
    Если Азия (00:00-07:00) создала направленный тренд > 0.5 ATR,
    Лондон (08:00) развернет это движение.
    """
    df = df.copy()
    df['Tokyo_Trap_Short'] = 0
    df['Tokyo_Trap_Long'] = 0
    
    # 1. ATR для фильтра силы движения
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Цена открытия Азии (00:00) и закрытия (07:00)
    # В 08:00 (Лондон) открытие Азии было 8 часов назад, закрытие - 1 час назад
    df['Asia_Open'] = df['Open'].shift(8)
    df['Asia_Close'] = df['Close'].shift(1)
    
    # 3. Триггер времени: Лондон, 08:00 UTC
    is_london_open = (df.index.hour == 8)
    
    # 4. ЛОГИКА ТРЕНДА АЗИИ
    asia_rally = (df['Asia_Close'] - df['Asia_Open']) > (0.5 * df['ATR_14D'])
    asia_dump = (df['Asia_Open'] - df['Asia_Close']) > (0.5 * df['ATR_14D'])
    
    # 5. СИГНАЛ (Входим ПРОТИВ Азии)
    df.loc[is_london_open & asia_rally, 'Tokyo_Trap_Short'] = 1
    df.loc[is_london_open & asia_dump, 'Tokyo_Trap_Long'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Asia_Open', 'Asia_Close'], inplace=True, errors='ignore')
    return df

def add_asian_box_breakout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Asian_Box_Long'] = 0
    df['Asian_Box_Short'] = 0
    
    # 1. ATR для измерения волатильности
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Границы Азии (00:00 - 08:00)
    # Используем окно 8 часов. Для проверки в 09:00 нам нужны данные, закончившиеся в 08:00.
    df['Asia_High'] = df['High'].rolling(window=8).max().shift(1)
    df['Asia_Low'] = df['Low'].rolling(window=8).min().shift(1)
    df['Asia_Range'] = df['Asia_High'] - df['Asia_Low']
    
    # 3. Характеристики сигнальной свечи (08:00 UTC)
    # Нам нужно посмотреть, как закрылась свеча, которая началась в 08:00 и закончилась в 09:00
    df['Signal_Candle_Close'] = df['Close']
    df['Signal_Candle_Open'] = df['Open']
    
    # 4. Условие входа: Понедельник-Пятница, 09:00 UTC (сразу после сигнальной свечи)
    is_entry_time = (df.index.hour == 9)
    
    # 5. Условие сжатой пружины (проверяем по состоянию на 08:00)
    is_coiled = df['Asia_Range'].shift(1) < (0.25 * df['ATR_14D'].shift(1))
    
    # 6. ОПРЕДЕЛЕНИЕ НАПРАВЛЕНИЯ (Импульс + Пробой)
    # LONG: Свеча 08:00 закрылась выше хая Азии И она бычья
    long_breakout = (df['Signal_Candle_Close'].shift(1) > df['Asia_High'].shift(1)) & \
                    (df['Signal_Candle_Close'].shift(1) > df['Signal_Candle_Open'].shift(1))
    
    # SHORT: Свеча 08:00 закрылась ниже лоу Азии И она медвежья
    short_breakout = (df['Signal_Candle_Close'].shift(1) < df['Asia_Low'].shift(1)) & \
                     (df['Signal_Candle_Close'].shift(1) < df['Signal_Candle_Open'].shift(1))
    
    df.loc[is_entry_time & is_coiled & long_breakout, 'Asian_Box_Long'] = 1
    df.loc[is_entry_time & is_coiled & short_breakout, 'Asian_Box_Short'] = 1
    
    # Очистка
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Asia_High', 'Asia_Low', 'Asia_Range', 
                     'Signal_Candle_Close', 'Signal_Candle_Open'], inplace=True, errors='ignore')
    return df

def add_london_true_trend_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_045: London True Trend (10:00 AM Entry).
    Пропускаем первые 2 токсичных часа Лондона (08:00-10:00).
    В 10:00 измеряем чистый вектор. Если тренд сформирован, входим по нему.
    """
    df = df.copy()
    df['LO_True_Trend_Long'] = 0
    df['LO_True_Trend_Short'] = 0
    
    # 1. Измеряем дневной ATR (для понимания нормы волатильности)
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    # 2. Триггер времени: 10:00 UTC (прошло 2 часа с открытия Лондона)
    is_10_am = (df.index.hour == 9)
    
    # 3. Находим цену открытия Лондона (08:00)
    # Если мы стоим на свече 10:00, то открытие 08:00 было ровно 2 свечи назад
    df['LO_Open_Price'] = df['Open'].shift(2)
    
    # Текущая цена на момент 10:00
    df['Current_10AM_Price'] = df['Open'] 
    
    # 4. Вычисляем Вектор (Displacement) за первые 2 часа
    # Фильтр: цена должна уйти хотя бы на 0.3 * ATR, чтобы мы поверили, что это тренд, а не флэт
    trend_up = is_10_am & ((df['Current_10AM_Price'] - df['LO_Open_Price']) > (0.3 * df['ATR_14D']))
    trend_down = is_10_am & ((df['LO_Open_Price'] - df['Current_10AM_Price']) > (0.3 * df['ATR_14D']))
    
    # 5. СИГНАЛЫ
    df.loc[trend_up, 'LO_True_Trend_Long'] = 1
    df.loc[trend_down, 'LO_True_Trend_Short'] = 1
    
    # Очистка
    df.drop(columns=['Daily_Range', 'ATR_14D', 'LO_Open_Price', 'Current_10AM_Price'], inplace=True, errors='ignore')
    return df

def add_judas_swing_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_047: The Judas Swing (Standard 1:2 RR Version).
    Фильтры: Без Декабря, Без Пятниц.
    Окно: 10:00 - 18:00 (Kyiv).
    """
    df = df.copy()
    df['Judas_Short'] = 0
    df['Judas_Short_SL'] = np.nan
    df['Judas_Long'] = 0
    df['Judas_Long_SL'] = np.nan
    
    # 1. Границы Азии (00:00 - 06:59 UTC)
    df['Day_Key'] = df.index.normalize() 
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    
    daily_stats = df[asia_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['Asia_High_Day'] = df['Day_Key'].map(daily_stats['High'])
    df['Asia_Low_Day'] = df['Day_Key'].map(daily_stats['Low'])
    
    # 2. ФИЛЬТРЫ ВРЕМЕНИ
    is_hunting_zone = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 18)
    is_not_friday = (df.index.dayofweek != 4)
    is_not_december = (df.index.month != 12)
    
    global_time_filter = is_hunting_zone & is_not_friday & is_not_december
    
    # 3. ЛОГИКА SWEEP (Пинбар или Возврат)
    # ШОРТ
    just_returned_high = (df['Close'] < df['Asia_High_Day']) & (df['Close'].shift(1) >= df['Asia_High_Day'])
    pinbar_high = (df['High'] > df['Asia_High_Day']) & (df['Close'] < df['Asia_High_Day'])
    sweep_high = (just_returned_high | pinbar_high) & (df['Close'] < df['Open'])
    
    # ЛОНГ
    just_returned_low = (df['Close'] > df['Asia_Low_Day']) & (df['Close'].shift(1) <= df['Asia_Low_Day'])
    pinbar_low = (df['Low'] < df['Asia_Low_Day']) & (df['Close'] > df['Asia_Low_Day'])
    sweep_low = (just_returned_low | pinbar_low) & (df['Close'] > df['Open'])
    
    # 4. СТОП-ЛОСС
    recent_highest = df['High'].rolling(3, min_periods=1).max()
    recent_lowest = df['Low'].rolling(3, min_periods=1).min()
    
    # 5. ЗАПИСЬ СИГНАЛОВ
    df.loc[global_time_filter & sweep_high, 'Judas_Short'] = 1
    df.loc[global_time_filter & sweep_high, 'Judas_Short_SL'] = recent_highest
    
    df.loc[global_time_filter & sweep_low, 'Judas_Long'] = 1
    df.loc[global_time_filter & sweep_low, 'Judas_Long_SL'] = recent_lowest
    
    df.drop(columns=['Day_Key', 'Asia_High_Day', 'Asia_Low_Day'], inplace=True, errors='ignore')
    return df

def add_ny_continuation_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_048: NY Overlap Continuation (Нью-Йоркский Локомотив).
    Входим в 16:00 UA в сторону сильного лондонского тренда.
    """
    df = df.copy()
    df['NY_Cont_Long'] = 0
    df['NY_Cont_Long_SL'] = np.nan
    df['NY_Cont_Short'] = 0
    df['NY_Cont_Short_SL'] = np.nan

    # 1. Считаем локальный ATR (защита от KeyError)
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()

    # 2. Безопасная работа с датами
    df['Day_Key'] = df.index.normalize()

    # 3. Фиксируем цены открытия Лондона (10:00) и закрытия (15:00)
    open_10 = df[df['UA_Hour'] == 10].groupby('Day_Key')['Open'].first()
    close_15 = df[df['UA_Hour'] == 15].groupby('Day_Key')['Close'].last()

    df['LDN_Open_10'] = df['Day_Key'].map(open_10)
    df['LDN_Close_15'] = df['Day_Key'].map(close_15)

    # 4. Вычисляем "Вектор Лондона"
    df['LDN_Vector'] = df['LDN_Close_15'] - df['LDN_Open_10']
    df['LDN_Distance'] = df['LDN_Vector'].abs()

    # ФИЛЬТР: Вектор должен быть сильным (> 0.5 ATR)
    strong_trend = df['LDN_Distance'] > (0.5 * df['ATR_14D'])

    # 5. Время входа: 16:00 UA (После открытия акций в США и новостей)
    is_entry_time = (df['UA_Hour'] == 16)

    # 6. Стоп-лосс (Прячем за откат на открытии Нью-Йорка)
    # rolling(3) в 16:00 покроет свечи 14:00, 15:00 и 16:00
    recent_highest = df['High'].rolling(3, min_periods=1).max()
    recent_lowest = df['Low'].rolling(3, min_periods=1).min()

    # 7. ЛОГИКА ВХОДА
    # Long: Лондон шел вверх (Vector > 0), тренд сильный, время 16:00
    long_cond = is_entry_time & strong_trend & (df['LDN_Vector'] > 0)
    # Short: Лондон шел вниз (Vector < 0), тренд сильный, время 16:00
    short_cond = is_entry_time & strong_trend & (df['LDN_Vector'] < 0)

    # 8. Запись сигналов
    df.loc[long_cond, 'NY_Cont_Long'] = 1
    df.loc[long_cond, 'NY_Cont_Long_SL'] = recent_lowest

    df.loc[short_cond, 'NY_Cont_Short'] = 1
    df.loc[short_cond, 'NY_Cont_Short_SL'] = recent_highest

    # Очистка
    df.drop(columns=['Day_Key', 'LDN_Open_10', 'LDN_Close_15', 'LDN_Vector', 'LDN_Distance', 'Daily_Range', 'ATR_14D'], inplace=True, errors='ignore')
    return df

def add_htf_trend_probability(df: pd.DataFrame, htf: str = '4h', lookback: int = 60) -> pd.DataFrame:
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