import pandas as pd
import numpy as np
from scipy.stats import linregress
from src.utils.decorators import provides

PIP = 0.0001 # Standard pip size for GBPUSD

@provides('PDH', 'PDL', 'PWH', 'PWL')
def add_previous_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    daily_stats = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'})
    daily_stats['PDH'] = daily_stats['High'].shift(1)
    daily_stats['PDL'] = daily_stats['Low'].shift(1)
    
    df['date_key'] = df.index.date
    df = df.join(daily_stats[['PDH', 'PDL']], on='date_key')

    df['week_key'] = df.index.isocalendar().week
    df['year_key'] = df.index.isocalendar().year
    
    weekly_stats = df.groupby(['year_key', 'week_key']).agg({'High': 'max', 'Low': 'min'})
    weekly_stats['PWH'] = weekly_stats['High'].shift(1)
    weekly_stats['PWL'] = weekly_stats['Low'].shift(1)
    
    df = df.join(weekly_stats[['PWH', 'PWL']], on=['year_key', 'week_key'])
    df.drop(columns=['date_key', 'week_key', 'year_key'], inplace=True)
    df[['PDH', 'PDL', 'PWH', 'PWL']] = df[['PDH', 'PDL', 'PWH', 'PWL']].ffill()
    return df

# Helper function (doesn't need @provides as it's called internally)
def get_htf_fvgs(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    htf_df = df.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    bull_gap = htf_df['Low'] > htf_df['High'].shift(2)
    bear_gap = htf_df['High'] < htf_df['Low'].shift(2)
    
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
    
    result_cols = [f'FVG_{timeframe}_Type', f'FVG_{timeframe}_Top', f'FVG_{timeframe}_Bottom', f'FVG_{timeframe}_Mid']
    return htf_df[result_cols].shift(1)

@provides(
    'FVG_1h_Type', 'FVG_1h_Top', 'FVG_1h_Bottom', 'FVG_1h_Mid',
    'FVG_4h_Type', 'FVG_4h_Top', 'FVG_4h_Bottom', 'FVG_4h_Mid',
    'FVG_1D_Type', 'FVG_1D_Top', 'FVG_1D_Bottom', 'FVG_1D_Mid'
)
def calculate_multi_tf_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    df_1h = get_htf_fvgs(df, '1h')
    df = df.join(df_1h)
    
    df_4h = get_htf_fvgs(df, '4h')
    df_4h_mapped = df_4h.reindex(df.index).ffill() 
    df = df.join(df_4h_mapped)
    
    df_1d = get_htf_fvgs(df, '1D')
    df_1d_mapped = df_1d.reindex(df.index).ffill()
    df = df.join(df_1d_mapped)
    return df

@provides('Swept_AL_Into_FVG', 'Swept_AH_Into_FVG')
def add_asian_sweep_context(df: pd.DataFrame, max_dist_pips: int = 15) -> pd.DataFrame:
    # --- SELF-HEALING: Ensure FVG data exists ---
    if 'FVG_4h_Type' not in df.columns:
        df = calculate_multi_tf_fvgs(df)

    tolerance = max_dist_pips * PIP
    is_asia = (df['UA_Hour'] >= 0) & (df['UA_Hour'] <= 10)
    
    df['Date_Key'] = df.index.date
    asia_stats = df[is_asia].groupby('Date_Key').agg({'High': 'max', 'Low': 'min'})
    asia_stats.rename(columns={'High': 'Asia_High', 'Low': 'Asia_Low'}, inplace=True)
    df = df.join(asia_stats, on='Date_Key')
    
    df['Bull_FVG_Below_AL'] = (df['FVG_4h_Type'] == 'BULL') & (df['Asia_Low'] >= df['FVG_4h_Top']) & ((df['Asia_Low'] - df['FVG_4h_Top']) <= tolerance)
    df['Bear_FVG_Above_AH'] = (df['FVG_4h_Type'] == 'BEAR') & (df['Asia_High'] <= df['FVG_4h_Bottom']) & ((df['FVG_4h_Bottom'] - df['Asia_High']) <= tolerance)
                              
    df['Swept_AL_Into_FVG'] = df['Bull_FVG_Below_AL'] & (df['Low'] < df['Asia_Low']) & (df['Low'] <= df['FVG_4h_Top'])
    df['Swept_AH_Into_FVG'] = df['Bear_FVG_Above_AH'] & (df['High'] > df['Asia_High']) & (df['High'] >= df['FVG_4h_Bottom'])
    
    df.drop(columns=['Date_Key'], inplace=True)
    df['Swept_AL_Into_FVG'] = df['Swept_AL_Into_FVG'].fillna(False).astype(int)
    df['Swept_AH_Into_FVG'] = df['Swept_AH_Into_FVG'].fillna(False).astype(int)
    return df

@provides('First_LDN_PDL_Long', 'First_LDN_PDH_Short')
def add_london_pdh_pdl_sweep_context(df: pd.DataFrame) -> pd.DataFrame:
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    fractal_low = (df['Confirmed_Fractal_Low'].fillna(0) == 1)
    fractal_high = (df['Confirmed_Fractal_High'].fillna(0) == 1)
    
    sweep_pdl = df['Confirmed_Fractal_Low_Price'] < df['PDL']
    sweep_pdh = df['Confirmed_Fractal_High_Price'] > df['PDH']
    
    ldn_sweep_pdl_long = is_london_eval & fractal_low & sweep_pdl
    ldn_sweep_pdh_short = is_london_eval & fractal_high & sweep_pdh
    
    df['Date'] = df.index.date
    sweep_count = (ldn_sweep_pdl_long | ldn_sweep_pdh_short).groupby(df['Date']).cumsum()
    
    df['First_LDN_PDL_Long'] = (ldn_sweep_pdl_long & (sweep_count == 1)).astype(int)
    df['First_LDN_PDH_Short'] = (ldn_sweep_pdh_short & (sweep_count == 1)).astype(int)
    
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('In_4h_Bull_FVG', 'In_4h_Bear_FVG', '1h_Bullish_Flip', '1h_Bearish_Flip')
def add_fvg_order_flow_context(df: pd.DataFrame) -> pd.DataFrame:
    # --- SELF-HEALING: Ensure FVG data exists ---
    if 'FVG_4h_Type' not in df.columns:
        df = calculate_multi_tf_fvgs(df)
    
    # --- SELF-HEALING: Ensure Volume Z-Score exists ---
    if 'Volume_ZScore' not in df.columns:
        from src.library.features import add_volume_zscore
        df = add_volume_zscore(df)

    df['In_4h_Bull_FVG'] = (df['FVG_4h_Type'] == 'BULL') & (df['Low'] <= df['FVG_4h_Top']) & (df['Close'] >= df['FVG_4h_Bottom'])
    df['In_4h_Bear_FVG'] = (df['FVG_4h_Type'] == 'BEAR') & (df['High'] >= df['FVG_4h_Bottom']) & (df['Close'] <= df['FVG_4h_Top'])

    prev_bearish = df['Close'].shift(1) < df['Open'].shift(1)
    curr_bullish = df['Close'] > df['Open']
    vol_expansion = df['Volume_ZScore'] > 0.5 
    df['1h_Bullish_Flip'] = prev_bearish & curr_bullish & vol_expansion
    
    prev_bullish = df['Close'].shift(1) > df['Open'].shift(1)
    curr_bearish = df['Close'] < df['Open']
    df['1h_Bearish_Flip'] = prev_bullish & curr_bearish & vol_expansion

    df['In_4h_Bull_FVG'] = df['In_4h_Bull_FVG'].astype(int)
    df['In_4h_Bear_FVG'] = df['In_4h_Bear_FVG'].astype(int)
    df['1h_Bullish_Flip'] = df['1h_Bullish_Flip'].astype(int)
    df['1h_Bearish_Flip'] = df['1h_Bearish_Flip'].astype(int)
    return df

@provides('Weekly_SMA', 'Weekly_Swing')
def add_weekly_swing_context(df: pd.DataFrame) -> pd.DataFrame:
    lookback = 168
    df['Weekly_SMA'] = df['Close'].rolling(window=lookback).mean()
    df['Weekly_Swing'] = 0
    df.loc[df['Close'] > df['Weekly_SMA'], 'Weekly_Swing'] = 1
    df.loc[df['Close'] < df['Weekly_SMA'], 'Weekly_Swing'] = -1
    return df

@provides('Major_Resistance', 'Major_Support', 'NY_First_Touch_Res', 'NY_First_Touch_Sup')
def add_ny_sr_touch_context(df: pd.DataFrame, tolerance_pips: int = 10) -> pd.DataFrame:
    tol = tolerance_pips * PIP
    df['Major_Resistance'] = df['High'].rolling(window=24*20).max().shift(1)
    df['Major_Support'] = df['Low'].rolling(window=24*20).min().shift(1)
    
    is_ny = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 22)
    ny_touch_res = is_ny & (df['High'] >= (df['Major_Resistance'] - tol))
    ny_touch_sup = is_ny & (df['Low'] <= (df['Major_Support'] + tol))
    
    df['Date'] = df.index.date
    res_touches = df.groupby('Date')[ny_touch_res.name if hasattr(ny_touch_res, 'name') else 'Date'].cumsum() if False else ny_touch_res.groupby(df['Date']).cumsum()
    sup_touches = ny_touch_sup.groupby(df['Date']).cumsum()
    
    df['NY_First_Touch_Res'] = (ny_touch_res & (res_touches == 1)).astype(int)
    df['NY_First_Touch_Sup'] = (ny_touch_sup & (sup_touches == 1)).astype(int)
    
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('NY_Open_Price', 'NY_Opened_In_Asia_Range', 'NY_Sweep_Asia_Low', 'NY_Sweep_Asia_High')
def add_ny_expansion_context(df: pd.DataFrame) -> pd.DataFrame:
    df['NY_Open_Price'] = df.where(df['UA_Hour'] == 15)['Open']
    df['NY_Open_Price'] = df.groupby(df.index.date)['NY_Open_Price'].ffill()

    df['NY_Opened_In_Asia_Range'] = (df['NY_Open_Price'] < df['Asia_High']) & (df['NY_Open_Price'] > df['Asia_Low'])
    
    is_ny_session = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 23)
    df['NY_Sweep_Asia_Low'] = is_ny_session & (df['Low'] < df['Asia_Low']) & (df['Low'].shift(1) >= df['Asia_Low'].shift(1))
    df['NY_Sweep_Asia_High'] = is_ny_session & (df['High'] > df['Asia_High']) & (df['High'].shift(1) <= df['Asia_High'].shift(1))

    df['NY_Opened_In_Asia_Range'] = df['NY_Opened_In_Asia_Range'].fillna(False).astype(int)
    df['NY_Sweep_Asia_High'] = df['NY_Sweep_Asia_High'].astype(int)
    df['NY_Sweep_Asia_Low'] = df['NY_Sweep_Asia_Low'].astype(int)
    return df

@provides('1W_Swing_Bullish', '1W_Swing_Bearish')
def add_1w_swing_context(df: pd.DataFrame) -> pd.DataFrame:
    df['1W_High'] = df['High'].rolling(window=120).max().shift(1)
    df['1W_Low'] = df['Low'].rolling(window=120).min().shift(1)

    broke_high = df['Close'] > df['1W_High']
    broke_low = df['Close'] < df['1W_Low']

    swing_signal = pd.Series(0, index=df.index)
    swing_signal.loc[broke_high] = 1
    swing_signal.loc[broke_low] = -1
    
    swing_state = swing_signal.replace(0, np.nan).ffill().fillna(0)

    df['1W_Swing_Bullish'] = (swing_state == 1).astype(int)
    df['1W_Swing_Bearish'] = (swing_state == -1).astype(int)
    df.drop(columns=['1W_High', '1W_Low'], inplace=True)
    return df

@provides('1D_Swing_Bullish', '1D_Swing_Bearish')
def add_1d_swing_context(df: pd.DataFrame) -> pd.DataFrame:
    df['1D_High'] = df['High'].rolling(window=24).max().shift(1)
    df['1D_Low'] = df['Low'].rolling(window=24).min().shift(1)

    broke_high = df['Close'] > df['1D_High']
    broke_low = df['Close'] < df['1D_Low']

    swing_signal = pd.Series(0, index=df.index)
    swing_signal.loc[broke_high] = 1
    swing_signal.loc[broke_low] = -1
    
    swing_state = swing_signal.replace(0, np.nan).ffill().fillna(0)

    df['1D_Swing_Bullish'] = (swing_state == 1).astype(int)
    df['1D_Swing_Bearish'] = (swing_state == -1).astype(int)
    df.drop(columns=['1D_High', '1D_Low'], inplace=True)
    return df

@provides('First_LDN_Weekly_Bull', 'First_LDN_Weekly_Bear')
def add_weekly_floor_context(df: pd.DataFrame) -> pd.DataFrame:
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    bull_swing = df['1W_Swing_Bullish'].fillna(0) == 1
    bear_swing = df['1W_Swing_Bearish'].fillna(0) == 1
    
    fractal_low = df['Confirmed_Fractal_Low'].fillna(0) == 1
    fractal_high = df['Confirmed_Fractal_High'].fillna(0) == 1
    
    bull_floor = is_london_eval & bull_swing & fractal_low
    bear_floor = is_london_eval & bear_swing & fractal_high
    
    df['Date'] = df.index.date
    trigger_count = (bull_floor | bear_floor).groupby(df['Date']).cumsum()
    
    df['First_LDN_Weekly_Bull'] = (bull_floor & (trigger_count == 1)).astype(int)
    df['First_LDN_Weekly_Bear'] = (bear_floor & (trigger_count == 1)).astype(int)
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('First_LDN_Counter_Low', 'First_LDN_Counter_High')
def add_london_counter_fractal_context(df: pd.DataFrame) -> pd.DataFrame:
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    bull_trend = (df['1D_Swing_Bullish'] == 1)
    bear_trend = (df['1D_Swing_Bearish'] == 1)
    fractal_low = (df['Confirmed_Fractal_Low'] == 1)
    fractal_high = (df['Confirmed_Fractal_High'] == 1)

    low_trap = is_london_eval & bear_trend & fractal_low
    high_trap = is_london_eval & bull_trend & fractal_high

    df['Date'] = df.index.date
    trap_count = (low_trap | high_trap).groupby(df['Date']).cumsum()
    
    df['First_LDN_Counter_Low'] = (low_trap & (trap_count == 1)).astype(int)
    df['First_LDN_Counter_High'] = (high_trap & (trap_count == 1)).astype(int)
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('Res_Confluence_Zone', 'Sup_Confluence_Zone', 'First_Touch_Res_Zone', 'First_Touch_Sup_Zone')
def add_fvg_sr_confluence_context(df: pd.DataFrame, max_dist_pips: int = 30) -> pd.DataFrame:
    dist = max_dist_pips * PIP
    df['Res_Confluence_Zone'] = (df['FVG_1D_Type'] == 'BEAR') & (df['Major_Resistance'] >= df['FVG_1D_Top']) & ((df['Major_Resistance'] - df['FVG_1D_Top']) <= dist)
    df['Sup_Confluence_Zone'] = (df['FVG_1D_Type'] == 'BULL') & (df['Major_Support'] <= df['FVG_1D_Bottom']) & ((df['FVG_1D_Bottom'] - df['Major_Support']) <= dist)

    touch_res = df['Res_Confluence_Zone'] & (df['High'] >= df['FVG_1D_Bottom'])
    touch_sup = df['Sup_Confluence_Zone'] & (df['Low'] <= df['FVG_1D_Top'])

    df['Date'] = df.index.date
    res_touches = touch_res.groupby(df['Date']).cumsum()
    sup_touches = touch_sup.groupby(df['Date']).cumsum()

    df['First_Touch_Res_Zone'] = (touch_res & (res_touches == 1)).astype(int)
    df['First_Touch_Sup_Zone'] = (touch_sup & (sup_touches == 1)).astype(int)
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('Asia_Res_Aligned', 'Asia_Sup_Aligned', 'First_False_Break_Res', 'First_False_Break_Sup')
def add_asian_sr_alignment_context(df: pd.DataFrame, max_dist_pips: int = 15) -> pd.DataFrame:
    dist = max_dist_pips * PIP
    df['Asia_Res_Aligned'] = (abs(df['Asia_High'] - df['Major_Resistance']) <= dist)
    df['Asia_Sup_Aligned'] = (abs(df['Asia_Low'] - df['Major_Support']) <= dist)

    is_active = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 21)
    break_res = is_active & df['Asia_Res_Aligned'] & (df['High'] > df['Asia_High'])
    break_sup = is_active & df['Asia_Sup_Aligned'] & (df['Low'] < df['Asia_Low'])

    df['Date'] = df.index.date
    res_count = break_res.groupby(df['Date']).cumsum()
    sup_count = break_sup.groupby(df['Date']).cumsum()

    df['First_False_Break_Res'] = (break_res & (res_count == 1)).fillna(False).astype(int)
    df['First_False_Break_Sup'] = (break_sup & (sup_count == 1)).fillna(False).astype(int)
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('LDN_Protected_AL_Long', 'LDN_Protected_AH_Short')
def add_asia_fvg_protection_context(df: pd.DataFrame) -> pd.DataFrame:
    asia_low_protected = (df['FVG_4h_Type'] == 'BULL') & (df['Asia_Low'] <= df['FVG_4h_Top']) & (df['Asia_Low'] >= df['FVG_4h_Bottom'])
    asia_high_protected = (df['FVG_4h_Type'] == 'BEAR') & (df['Asia_High'] <= df['FVG_4h_Top']) & (df['Asia_High'] >= df['FVG_4h_Bottom'])

    is_london_open = (df['UA_Hour'] == 10)
    df['LDN_Protected_AL_Long'] = (is_london_open & asia_low_protected).fillna(False).astype(int)
    df['LDN_Protected_AH_Short'] = (is_london_open & asia_high_protected).fillna(False).astype(int)
    return df

@provides('First_1W_Rej_Long', 'First_1W_Rej_Short')
def add_1w_level_rejection_context(df: pd.DataFrame, max_dist_pips: int = 20) -> pd.DataFrame:
    tol = max_dist_pips * PIP
    fractal_low = (df['Confirmed_Fractal_Low'].fillna(0) == 1)
    fractal_high = (df['Confirmed_Fractal_High'].fillna(0) == 1)
    
    tap_pwl = df['Confirmed_Fractal_Low_Price'] <= (df['PWL'] + tol)
    tap_pwh = df['Confirmed_Fractal_High_Price'] >= (df['PWH'] - tol)
              
    is_active = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 21)
    
    rej_long = is_active & fractal_low & tap_pwl
    rej_short = is_active & fractal_high & tap_pwh
    
    df['Date'] = df.index.date
    rej_count = (rej_long | rej_short).groupby(df['Date']).cumsum()
    
    df['First_1W_Rej_Long'] = (rej_long & (rej_count == 1)).astype(int)
    df['First_1W_Rej_Short'] = (rej_short & (rej_count == 1)).astype(int)
    df.drop(columns=['Date'], inplace=True)
    return df

@provides('Geo_Shock_Short')
def add_geopolitical_shock_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['Geo_Shock_Short'] = 0
    shock_events = [e for e in events if e.get('category') == 'Geopolitical_Shock']
    
    for event in shock_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'Geo_Shock_Short'] = 1
        except: pass
    df['Geo_Shock_Short'] = df['Geo_Shock_Short'].fillna(0).astype(int)
    return df

@provides('Election_Vol_Crush_Short')
def add_election_volatility_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['Election_Vol_Crush_Short'] = 0
    elections = [e for e in events if e.get('category') == 'Elections']
    
    for event in elections:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'Election_Vol_Crush_Short'] = 1
        except: pass
    df['Election_Vol_Crush_Short'] = df['Election_Vol_Crush_Short'].fillna(0).astype(int)
    return df

@provides('UK_Shock_Cont_Long', 'UK_Shock_Cont_Short')
def add_uk_political_shock_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['UK_Shock_T0'] = 0
    shock_events = [e for e in events if e.get('category') == 'UK_Political_Shock']
    
    for event in shock_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'UK_Shock_T0'] = 1
        except: pass
            
    df['UK_Shock_T1'] = df['UK_Shock_T0'].shift(1).fillna(0)
    df['UK_Shock_Cont_Long'] = ((df['UK_Shock_T1'] == 1) & (df['HTF_Bullish_Prob'] >= 55)).astype(int)
    df['UK_Shock_Cont_Short'] = ((df['UK_Shock_T1'] == 1) & (df['HTF_Bullish_Prob'] <= 45)).astype(int)
    df.drop(columns=['UK_Shock_T0', 'UK_Shock_T1'], inplace=True)
    return df

@provides('BoE_Hawkish_Long')
def add_boe_hawkish_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['BoE_Hawkish_T0'] = 0
    hawkish_events = [e for e in events if e.get('category') == 'BoE_Hawkish_Shock']
    for event in hawkish_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'BoE_Hawkish_T0'] = 1
        except: pass
    df['BoE_Hawkish_Long'] = df['BoE_Hawkish_T0'].fillna(0).astype(int)
    df.drop(columns=['BoE_Hawkish_T0'], inplace=True)
    return df

@provides('CPI_Momentum_Long', 'CPI_Momentum_Short')
def add_uk_cpi_momentum_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['CPI_Release_T0'] = 0
    cpi_events = [e for e in events if e.get('category') == 'UK_CPI_Shock']
    for event in cpi_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'CPI_Release_T0'] = 1
        except: pass
            
    df['CPI_T4_Active'] = df['CPI_Release_T0'].shift(4).fillna(0)
    t4_direction = np.where(df['Close'] >= df['Open'].shift(3), 1, -1)
    
    df['CPI_Momentum_Long'] = ((df['CPI_T4_Active'] == 1) & (t4_direction == 1)).astype(int)
    df['CPI_Momentum_Short'] = ((df['CPI_T4_Active'] == 1) & (t4_direction == -1)).astype(int)
    df.drop(columns=['CPI_Release_T0', 'CPI_T4_Active'], inplace=True)
    return df

@provides('Sovereign_Risk_Short')
def add_sovereign_risk_proxy_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    tr = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    atr_4h = tr.rolling(window=4).mean()
    atr_30d_avg = atr_4h.rolling(window=720).mean()
    vol_anomaly = atr_4h > (atr_30d_avg * 2.5)
    
    pol_shock_active = pd.Series(0, index=df.index)
    uk_shocks = [e for e in events if e.get('category') == 'UK_Political_Shock']
    for event in uk_shocks:
        try:
            start_dt = pd.to_datetime(event['start_date'])
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            end_dt = pd.to_datetime(event['end_date'])
            end_dt = end_dt.tz_localize('UTC') if end_dt.tz is None else end_dt.tz_convert('UTC')
            mask = (df.index >= start_dt) & (df.index <= end_dt)
            pol_shock_active.loc[mask] = 1
        except: pass
            
    is_4h_bounce = df['Close'] > df['Open'].shift(3)
    df['Sovereign_Risk_Short'] = ((pol_shock_active.shift(1) == 1) & (vol_anomaly.shift(1) == True) & (is_4h_bounce.shift(1) == True)).astype(int)
    return df

@provides('Gap_Up_Fade_Short', 'Gap_Down_Fade_Long')
def add_weekend_gap_context(df: pd.DataFrame) -> pd.DataFrame:
    time_diff = df.index.to_series().diff()
    is_week_open = time_diff > pd.Timedelta(hours=24)
    gap_pips = (df['Open'] - df['Close'].shift(1)) * 10000
    
    df['Gap_Up_Fade_Short'] = ((is_week_open) & (gap_pips >= 40)).astype(int)
    df['Gap_Down_Fade_Long'] = ((is_week_open) & (gap_pips <= -40)).astype(int)
    return df

@provides('BoE_Tone_Shift_Short')
def add_boe_tone_shift_proxy_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    boe_release_t0 = pd.Series(0, index=df.index)
    boe_events = [e for e in events if e.get('category') == 'BoE_Hawkish_Shock']
    for event in boe_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                boe_release_t0.loc[rounded_dt] = 1
        except: pass
            
    is_green = df['Close'] > df['Open']
    is_red = df['Close'] < df['Open']
    
    df['BoE_Tone_Shift_Short'] = ((boe_release_t0.shift(1) == 1) & (is_green.shift(1) == True) & (is_red == True)).astype(int)
    return df

@provides('Macro_Inside_Bar_Short')
def add_macro_shock_inside_bar_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    tr_local = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    tr_mean = tr_local.rolling(window=720).mean()
    tr_std = tr_local.rolling(window=720).std()
    is_3sd_spike = tr_local > (tr_mean + 3 * tr_std)
    
    strict_macro_window = pd.Series(0, index=df.index)
    target_events = [e for e in events if e.get('category') in ['Geopolitical_Shock', 'UK_Political_Shock']]
    for event in target_events:
        try:
            start_dt = pd.to_datetime(event['start_date'])
            start_dt = start_dt.tz_localize('UTC') if start_dt.tz is None else start_dt.tz_convert('UTC')
            strict_end_dt = start_dt + pd.Timedelta(hours=24)
            mask = (df.index >= start_dt) & (df.index <= strict_end_dt)
            strict_macro_window.loc[mask] = 1
        except: pass
            
    df['Macro_Inside_Bar_Short'] = ((strict_macro_window.shift(1) == 1) & (is_3sd_spike.shift(1) == True)).astype(int)
    return df

@provides('Algo_Vol_Crush_Short')
def add_pure_algo_vol_crush_context(df: pd.DataFrame) -> pd.DataFrame:
    tr_local = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    tr_mean = tr_local.rolling(window=720).mean()
    tr_std = tr_local.rolling(window=720).std()
    is_3sd_spike = tr_local > (tr_mean + 3 * tr_std)
        
    df['Algo_Vol_Crush_Short'] = is_3sd_spike.shift(1).fillna(0).astype(int)
    return df

@provides('NFP_Fade_Long', 'NFP_Fade_Short')
def add_nfp_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    nfp_level = pd.Series(np.nan, index=df.index)
    nfp_initial_dir = pd.Series(np.nan, index=df.index) 

    nfp_events = [e for e in events if e.get('category') == 'US_NFP_Divergence']
    for event in nfp_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                nfp_level.loc[rounded_dt] = df.loc[rounded_dt, 'Open']
                nfp_initial_dir.loc[rounded_dt] = 1 if df.loc[rounded_dt, 'Close'] > df.loc[rounded_dt, 'Open'] else -1
        except: pass

    nfp_level = nfp_level.ffill(limit=5)
    nfp_initial_dir = nfp_initial_dir.ffill(limit=5)

    fade_long = ((nfp_initial_dir == -1) & (df['Close'] > nfp_level)).astype(int)
    fade_short = ((nfp_initial_dir == 1) & (df['Close'] < nfp_level)).astype(int)

    df['NFP_Fade_Long'] = ((fade_long == 1) & (fade_long.shift(1) == 0)).astype(int)
    df['NFP_Fade_Short'] = ((fade_short == 1) & (fade_short.shift(1) == 0)).astype(int)
    return df

@provides('NFP_Resumption_Short', 'NFP_Resumption_Long')
def add_nfp_revision_trap_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    trap_level = pd.Series(np.nan, index=df.index)
    trap_initial_dir = pd.Series(np.nan, index=df.index) 

    trap_events = [e for e in events if e.get('category') == 'US_NFP_Revision_Trap']
    for event in trap_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                trap_level.loc[rounded_dt] = df.loc[rounded_dt, 'Open']
                trap_initial_dir.loc[rounded_dt] = 1 if df.loc[rounded_dt, 'Close'] > df.loc[rounded_dt, 'Open'] else -1
        except: pass

    trap_level = trap_level.ffill(limit=4)
    trap_initial_dir = trap_initial_dir.ffill(limit=4)

    resumption_short = ((trap_initial_dir == -1) & (df['Close'] > trap_level)).astype(int)
    resumption_long = ((trap_initial_dir == 1) & (df['Close'] < trap_level)).astype(int)

    df['NFP_Resumption_Short'] = ((resumption_short == 1) & (resumption_short.shift(1) == 0)).astype(int)
    df['NFP_Resumption_Long'] = ((resumption_long == 1) & (resumption_long.shift(1) == 0)).astype(int)
    return df

@provides('CPI_Match_Fade_Short', 'CPI_Match_Fade_Long')
def add_cpi_match_mean_reversion_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['CPI_Match_Fade_Short'] = 0
    df['CPI_Match_Fade_Long'] = 0
    cpi_events = [e for e in events if e.get('category') == 'US_CPI_Match']

    for event in cpi_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            date_str = rounded_dt.strftime('%Y-%m-%d')
            london_open_dt = pd.to_datetime(f"{date_str} 08:00:00").tz_localize('UTC')
            
            if rounded_dt in df.index and london_open_dt in df.index:
                london_anchor_price = df.loc[london_open_dt, 'Open']
                pre_cpi_price = df.loc[rounded_dt, 'Open']
                if pre_cpi_price > london_anchor_price:
                    df.loc[rounded_dt, 'CPI_Match_Fade_Short'] = 1
                elif pre_cpi_price < london_anchor_price:
                    df.loc[rounded_dt, 'CPI_Match_Fade_Long'] = 1
        except: pass
    return df

@provides('CB_Divergence_Long', 'CB_Divergence_Short')
def add_cb_divergence_state_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    fed_state = pd.Series(np.nan, index=df.index)
    boe_state = pd.Series(np.nan, index=df.index)
    
    for event in [e for e in events if e.get('category') == 'Fed_Significant_Probability_Shift']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            dt = dt.floor('h')
            if dt in df.index:
                name = event['name'].lower()
                fed_state.loc[dt] = -1 if ('cut prob from' in name and 'drops' not in name) or 'emergency cut' in name or 'pivot' in name else 1
        except: pass

    for event in [e for e in events if e.get('category') == 'BoE_Significant_Probability_Shift']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            dt = dt.floor('h')
            if dt in df.index:
                name = event['name'].lower()
                boe_state.loc[dt] = -1 if ('cut prob' in name and 'collapses' not in name) or 'dovish' in name else 1
        except: pass

    fed_state = fed_state.ffill()
    boe_state = boe_state.ffill()

    is_bullish_div = ((boe_state == 1) & (fed_state == -1)).astype(int)
    is_bearish_div = ((boe_state == -1) & (fed_state == 1)).astype(int)

    df['CB_Divergence_Long'] = ((is_bullish_div == 1) & (is_bullish_div.shift(1) == 0)).astype(int)
    df['CB_Divergence_Short'] = ((is_bearish_div == 1) & (is_bearish_div.shift(1) == 0)).astype(int)
    return df

@provides('FOMC_Sell_News_Long')
def add_fomc_sell_the_news_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['FOMC_Sell_News_Long'] = 0
    fomc_events = [e for e in events if e.get('category') == 'US_FOMC_InLine_Hike']
    for event in fomc_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                df.loc[rounded_dt, 'FOMC_Sell_News_Long'] = 1
        except: pass
    return df

@provides('Macro_CPI_Div_Long')
def add_uk_us_cpi_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    us_cpi_release = pd.Series(np.nan, index=df.index)
    uk_cpi_release = pd.Series(np.nan, index=df.index)
    
    for event in [e for e in events if e.get('category') == 'US_CPI_Cold']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index: us_cpi_release.loc[dt.floor('h')] = -1
        except: pass

    for event in [e for e in events if e.get('category') == 'UK_CPI_Hot']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index: uk_cpi_release.loc[dt.floor('h')] = 1
        except: pass

    us_cold_memory = us_cpi_release.ffill(limit=720)
    uk_hot_memory = uk_cpi_release.ffill(limit=720)

    df['Macro_CPI_Div_Long'] = 0
    uk_trigger_mask = (uk_cpi_release == 1) & (us_cold_memory == -1)
    us_trigger_mask = (us_cpi_release == -1) & (uk_hot_memory == 1)

    df.loc[uk_trigger_mask | us_trigger_mask, 'Macro_CPI_Div_Long'] = 1
    return df

@provides('Unemp_Fakeout_Long')
def add_unemp_fakeout_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    df['Unemp_Fakeout_Long'] = 0
    target_events = [e for e in events if e.get('category') == 'US_Unemp_Rise_UK_Stable']
    for event in target_events:
        try:
            dt = pd.to_datetime(event['start_date'])
            dt = dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC')
            rounded_dt = dt.floor('h')
            if rounded_dt in df.index:
                if df.loc[rounded_dt, 'Close'] < df.loc[rounded_dt, 'Open']:
                    df.loc[rounded_dt, 'Unemp_Fakeout_Long'] = 1
        except: pass
    return df

@provides('Retail_Div_Long')
def add_retail_sales_divergence_context(df: pd.DataFrame, events: list) -> pd.DataFrame:
    us_retail_release = pd.Series(np.nan, index=df.index)
    uk_retail_release = pd.Series(np.nan, index=df.index)
    
    for event in [e for e in events if e.get('category') == 'US_Retail_Miss']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index: us_retail_release.loc[dt.floor('h')] = -1
        except: pass

    for event in [e for e in events if e.get('category') == 'UK_Retail_Beat']:
        try:
            dt = pd.to_datetime(event['start_date']).tz_localize('UTC') if pd.to_datetime(event['start_date']).tz is None else pd.to_datetime(event['start_date']).tz_convert('UTC')
            if dt.floor('h') in df.index: uk_retail_release.loc[dt.floor('h')] = 1
        except: pass

    us_retail_memory = us_retail_release.ffill(limit=720)
    uk_retail_memory = uk_retail_release.ffill(limit=720)

    df['Retail_Div_Long'] = 0
    uk_trigger_mask = (uk_retail_release == 1) & (us_retail_memory == -1)
    us_trigger_mask = (us_retail_release == -1) & (uk_retail_memory == 1)
    df.loc[uk_trigger_mask | us_trigger_mask, 'Retail_Div_Long'] = 1
    return df

@provides('Friday_Reversal_Short', 'Friday_Reversal_Long')
def add_friday_reversal_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    week_open = df['Open'].shift(112)
    
    is_friday_16 = (df.index.dayofweek == 4) & (df.index.hour == 16)
    weekly_rally = df['Close'] - week_open
    weekly_dump = week_open - df['Close']
    
    df['Friday_Reversal_Short'] = (is_friday_16 & (weekly_rally > (2 * atr_14d))).astype(int)
    df['Friday_Reversal_Long'] = (is_friday_16 & (weekly_dump > (2 * atr_14d))).astype(int)
    return df

@provides('Monday_Reversion_Short', 'Monday_Reversion_Long')
def add_monday_gap_reversion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    weekly_delta = df['Close'] - df['Open'].shift(120)
    is_monday_open = (df.index.dayofweek == 0) & (df.index.hour == 0)
    
    df['Monday_Reversion_Short'] = (is_monday_open & (weekly_delta > (2 * atr_14d))).astype(int)
    df['Monday_Reversion_Long'] = (is_monday_open & (weekly_delta < -(2 * atr_14d))).astype(int)
    return df

@provides('Tuesday_Resumption_Long', 'Tuesday_Resumption_Short')
def add_turnaround_tuesday_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    prev_week_open = df['Open'].shift(144)
    prev_week_close = df['Close'].shift(24) 
    monday_open = df['Open'].shift(24)
    monday_close = df['Close'].shift(1)
    
    is_tuesday_london_open = (df.index.dayofweek == 1) & (df.index.hour == 8)
    
    long_cond = is_tuesday_london_open & (prev_week_close > prev_week_open) & (monday_close < monday_open)
    short_cond = is_tuesday_london_open & (prev_week_close < prev_week_open) & (monday_close > monday_open)
    
    df['Tuesday_Resumption_Long'] = long_cond.astype(int)
    df['Tuesday_Resumption_Short'] = short_cond.astype(int)
    return df

@provides('Wed_Fakeout_Long', 'Wed_Fakeout_Short')
def add_wednesday_fakeout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_hl = df.groupby(df.index.date).agg({'High': 'max', 'Low': 'min'})
    prev_day_hl = daily_hl.shift(1)
    prev_day_high = df.index.date.map(lambda d: prev_day_hl.loc[d, 'High'] if d in prev_day_hl.index else np.nan)
    prev_day_low = df.index.date.map(lambda d: prev_day_hl.loc[d, 'Low'] if d in prev_day_hl.index else np.nan)
    
    is_wednesday = df.index.dayofweek == 2
    sweep_high = (df['High'] > prev_day_high) & (df['Close'] < prev_day_high) & (df['Close'] < df['Open'])
    sweep_low = (df['Low'] < prev_day_low) & (df['Close'] > prev_day_low) & (df['Close'] > df['Open'])
    
    df['Wed_Fakeout_Short'] = (is_wednesday & sweep_high).astype(int)
    df['Wed_Fakeout_Long'] = (is_wednesday & sweep_low).astype(int)
    return df

@provides('Thursday_Trend_Long', 'Thursday_Trend_Short')
def add_thursday_expansion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    week_open = df['Open'].shift(80)
    wed_close = df['Close'].shift(9) 
    
    is_thursday_london_open = (df.index.dayofweek == 3) & (df.index.hour == 8)
    trend_up = (wed_close - week_open) > (0.5 * atr_14d)
    trend_down = (week_open - wed_close) > (0.5 * atr_14d)
    
    df['Thursday_Trend_Long'] = (is_thursday_london_open & trend_up).astype(int)
    df['Thursday_Trend_Short'] = (is_thursday_london_open & trend_down).astype(int)
    return df

# Renamed to prevent conflict with the daily fixing strategy below
@provides('Fix_Fade_Short_Weekly', 'Fix_Fade_Long_Weekly')
def add_weekly_london_fix_fade_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    week_performance = df['Close'] - df['Open'].shift(112)
    is_friday_fix = (df.index.dayofweek == 4) & (df.index.hour == 16)
    
    df['Fix_Fade_Short_Weekly'] = (is_friday_fix & (week_performance > (1.5 * atr_14d))).astype(int)
    df['Fix_Fade_Long_Weekly'] = (is_friday_fix & (week_performance < -(1.5 * atr_14d))).astype(int)
    return df

@provides('Tokyo_Trap_Short', 'Tokyo_Trap_Long')
def add_tokyo_trap_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    asia_open = df['Open'].shift(8)
    asia_close = df['Close'].shift(1)
    
    is_london_open = (df.index.hour == 8)
    asia_rally = (asia_close - asia_open) > (0.5 * atr_14d)
    asia_dump = (asia_open - asia_close) > (0.5 * atr_14d)
    
    df['Tokyo_Trap_Short'] = (is_london_open & asia_rally).astype(int)
    df['Tokyo_Trap_Long'] = (is_london_open & asia_dump).astype(int)
    return df

@provides('Asian_Box_Long', 'Asian_Box_Short')
def add_asian_box_breakout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    
    asia_high = df['High'].rolling(window=8).max().shift(1)
    asia_low = df['Low'].rolling(window=8).min().shift(1)
    asia_range = asia_high - asia_low
    
    is_entry_time = (df.index.hour == 9)
    is_coiled = asia_range.shift(1) < (0.25 * atr_14d.shift(1))
    
    long_breakout = (df['Close'].shift(1) > asia_high.shift(1)) & (df['Close'].shift(1) > df['Open'].shift(1))
    short_breakout = (df['Close'].shift(1) < asia_low.shift(1)) & (df['Close'].shift(1) < df['Open'].shift(1))
    
    df['Asian_Box_Long'] = (is_entry_time & is_coiled & long_breakout).astype(int)
    df['Asian_Box_Short'] = (is_entry_time & is_coiled & short_breakout).astype(int)
    return df

@provides('LO_True_Trend_Long', 'LO_True_Trend_Short')
def add_london_true_trend_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    is_10_am = (df.index.hour == 9)
    lo_open_price = df['Open'].shift(2)
    
    trend_up = is_10_am & ((df['Open'] - lo_open_price) > (0.3 * atr_14d))
    trend_down = is_10_am & ((lo_open_price - df['Open']) > (0.3 * atr_14d))
    
    df['LO_True_Trend_Long'] = trend_up.astype(int)
    df['LO_True_Trend_Short'] = trend_down.astype(int)
    return df

@provides('Judas_Short', 'Judas_Long')
def add_judas_swing_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    day_key = df.index.normalize() 
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    daily_stats = df[asia_mask].groupby(day_key).agg({'High': 'max', 'Low': 'min'})
    
    asia_high_day = day_key.map(daily_stats['High'])
    asia_low_day = day_key.map(daily_stats['Low'])
    
    global_time_filter = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 18) & (df.index.dayofweek != 4) & (df.index.month != 12)
    
    sweep_high = ((df['Close'] < asia_high_day) & (df['Close'].shift(1) >= asia_high_day) | 
                  (df['High'] > asia_high_day) & (df['Close'] < asia_high_day)) & (df['Close'] < df['Open'])
                  
    sweep_low = ((df['Close'] > asia_low_day) & (df['Close'].shift(1) <= asia_low_day) | 
                 (df['Low'] < asia_low_day) & (df['Close'] > asia_low_day)) & (df['Close'] > df['Open'])
    
    df['Judas_Short'] = (global_time_filter & sweep_high).astype(int)
    df['Judas_Long'] = (global_time_filter & sweep_low).astype(int)
    return df

@provides('NY_Cont_Long', 'NY_Cont_Short')
def add_ny_continuation_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()
    day_key = df.index.normalize()

    open_10 = df[df['UA_Hour'] == 10].groupby(day_key)['Open'].first()
    close_15 = df[df['UA_Hour'] == 15].groupby(day_key)['Close'].last()

    ldn_vector = day_key.map(close_15) - day_key.map(open_10)
    strong_trend = ldn_vector.abs() > (0.5 * atr_14d)
    is_entry_time = (df['UA_Hour'] == 16)

    df['NY_Cont_Long'] = (is_entry_time & strong_trend & (ldn_vector > 0)).astype(int)
    df['NY_Cont_Short'] = (is_entry_time & strong_trend & (ldn_vector < 0)).astype(int)
    return df

@provides('NY_Sweep_Short', 'NY_Sweep_Long')
def add_ny_news_sweep_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    day_key = df.index.normalize()
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    asia_stats = df[asia_mask].groupby(day_key).agg({'High': 'max', 'Low': 'min'})
    asia_high = day_key.map(asia_stats['High'])
    asia_low = day_key.map(asia_stats['Low'])

    ldn_mask = (df['UA_Hour'] >= 10) & (df['UA_Hour'] < 15)
    ldn_stats = df[ldn_mask].groupby(day_key).agg({'High': 'max', 'Low': 'min'})
    ldn_high = day_key.map(ldn_stats['High'])
    ldn_low = day_key.map(ldn_stats['Low'])

    ldn_swept_asia_high = ldn_high > asia_high
    ldn_swept_asia_low = ldn_low < asia_low

    valid_time = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 17) & (df.index.dayofweek != 4) & (df.index.month != 12)

    sweep_high = ((df['Close'] < ldn_high) & (df['Close'].shift(1) >= ldn_high) | 
                  (df['High'] > ldn_high) & (df['Close'] < ldn_high)) & (df['Close'] < df['Open'])

    sweep_low = ((df['Close'] > ldn_low) & (df['Close'].shift(1) <= ldn_low) | 
                 (df['Low'] < ldn_low) & (df['Close'] > ldn_low)) & (df['Close'] > df['Open'])

    df['NY_Sweep_Short'] = (valid_time & sweep_high & (~ldn_swept_asia_high)).astype(int)
    df['NY_Sweep_Long'] = (valid_time & sweep_low & (~ldn_swept_asia_low)).astype(int)
    return df

@provides('Fix_Fade_Short', 'Fix_Fade_Long')
def add_london_fix_fade_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    day_key = df.index.normalize()
    daily_range = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    atr_14d = daily_range.rolling(336).mean()

    day_open = df.groupby(day_key)['Open'].transform('first')
    day_trend_vector = df['Close'] - day_open
    exhaustion_reached = day_trend_vector.abs() > (0.8 * atr_14d)

    is_fix_time = (df['UA_Hour'] == 18)
    is_valid_day = (df.index.dayofweek != 4) & (df.index.month != 12)

    df['Fix_Fade_Short'] = (is_fix_time & is_valid_day & exhaustion_reached & (day_trend_vector > 0)).astype(int)
    df['Fix_Fade_Long'] = (is_fix_time & is_valid_day & exhaustion_reached & (day_trend_vector < 0)).astype(int)
    return df

@provides('HTF_Bullish_Prob')
def add_htf_trend_probability(df: pd.DataFrame, htf: str = '4h', lookback: int = 60) -> pd.DataFrame:
    htf_df = pd.DataFrame()
    htf_df['HTF_Close'] = df['Close'].resample(htf).last()
    htf_df['HTF_High'] = df['High'].resample(htf).max()
    htf_df['HTF_Low'] = df['Low'].resample(htf).min()
    htf_df.dropna(inplace=True)

    slopes, r_squareds = [], []
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
    htf_df['Stat_Score'] = np.where(htf_df['Slope'] > 0, 50 * htf_df['R2'], 0)

    n = 2
    htf_df['Fractal_Up'] = False
    htf_df['Fractal_Down'] = False
    
    for i in range(2*n, len(htf_df)):
        window_high = htf_df['HTF_High'].iloc[i - 2*n : i + 1]
        window_low = htf_df['HTF_Low'].iloc[i - 2*n : i + 1]
        mid_idx = i - n
        
        if htf_df['HTF_High'].iloc[mid_idx] == window_high.max():
            htf_df.iat[mid_idx, htf_df.columns.get_loc('Fractal_Up')] = True
        if htf_df['HTF_Low'].iloc[mid_idx] == window_low.min():
            htf_df.iat[mid_idx, htf_df.columns.get_loc('Fractal_Down')] = True

    struct_scores = []
    last_up_1, last_up_2, last_down_1, last_down_2 = None, None, None, None

    for i in range(len(htf_df)):
        check_idx = i - n
        if check_idx >= 0:
            if htf_df['Fractal_Up'].iloc[check_idx]:
                last_up_2, last_up_1 = last_up_1, htf_df['HTF_High'].iloc[check_idx]
            if htf_df['Fractal_Down'].iloc[check_idx]:
                last_down_2, last_down_1 = last_down_1, htf_df['HTF_Low'].iloc[check_idx]

        score = 25 
        if last_up_1 and last_up_2 and last_down_1 and last_down_2:
            hh = last_up_1 > last_up_2
            hl = last_down_1 > last_down_2
            lh = last_up_1 < last_up_2
            ll = last_down_1 < last_down_2
            if hh and hl: score = 50 
            elif lh and ll: score = 0 
        struct_scores.append(score)

    htf_df['Struct_Score'] = struct_scores
    htf_df['HTF_Bullish_Prob'] = (htf_df['Stat_Score'] + htf_df['Struct_Score']).shift(1)
    
    df = df.join(htf_df[['HTF_Bullish_Prob']])
    df['HTF_Bullish_Prob'] = df['HTF_Bullish_Prob'].ffill().fillna(50.0).round(1)
    return df