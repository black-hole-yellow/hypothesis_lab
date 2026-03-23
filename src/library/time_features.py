import numpy as np
import pandas as pd

PIP = 0.0001

def add_judas_swing_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Judas_Short' in df.columns: return df
    df = df.copy()
    df['Judas_Short'] = 0
    df['Judas_Short_SL'] = np.nan
    df['Judas_Long'] = 0
    df['Judas_Long_SL'] = np.nan
    
    df['Day_Key'] = df.index.normalize() 
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    
    daily_stats = df[asia_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['Asia_High_Day'] = df['Day_Key'].map(daily_stats['High'])
    df['Asia_Low_Day'] = df['Day_Key'].map(daily_stats['Low'])
    
    is_hunting_zone = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 18)
    is_not_friday = (df.index.dayofweek != 4)
    is_not_december = (df.index.month != 12)
    
    global_time_filter = is_hunting_zone & is_not_friday & is_not_december
    
    just_returned_high = (df['Close'] < df['Asia_High_Day']) & (df['Close'].shift(1) >= df['Asia_High_Day'])
    pinbar_high = (df['High'] > df['Asia_High_Day']) & (df['Close'] < df['Asia_High_Day'])
    sweep_high = (just_returned_high | pinbar_high) & (df['Close'] < df['Open'])
    
    just_returned_low = (df['Close'] > df['Asia_Low_Day']) & (df['Close'].shift(1) <= df['Asia_Low_Day'])
    pinbar_low = (df['Low'] < df['Asia_Low_Day']) & (df['Close'] > df['Asia_Low_Day'])
    sweep_low = (just_returned_low | pinbar_low) & (df['Close'] > df['Open'])
    
    recent_highest = df['High'].rolling(3, min_periods=1).max()
    recent_lowest = df['Low'].rolling(3, min_periods=1).min()
    
    df.loc[global_time_filter & sweep_high, 'Judas_Short'] = 1
    df.loc[global_time_filter & sweep_high, 'Judas_Short_SL'] = recent_highest
    
    df.loc[global_time_filter & sweep_low, 'Judas_Long'] = 1
    df.loc[global_time_filter & sweep_low, 'Judas_Long_SL'] = recent_lowest
    
    df.drop(columns=['Day_Key', 'Asia_High_Day', 'Asia_Low_Day'], inplace=True, errors='ignore')
    return df

def add_asian_sweep_context(df: pd.DataFrame, events=None, max_dist_pips: int = 15) -> pd.DataFrame:
    """Calculates Asian Session Extremes and detects sweeps into 4H FVGs."""
    if 'Swept_AL_Into_FVG' in df.columns: return df
    tolerance = max_dist_pips * PIP
    
    # 1. Считаем Азию только если ее еще нет (Один раз!)
    if 'Asia_High' not in df.columns:
        df['Date_Key_Tmp'] = df.index.date
        is_asia = (df['UA_Hour'] >= 0) & (df['UA_Hour'] <= 8)
        asia_stats = df[is_asia].groupby('Date_Key_Tmp').agg({'High': 'max', 'Low': 'min'})
        df['Asia_High'] = df['Date_Key_Tmp'].map(asia_stats['High'])
        df['Asia_Low'] = df['Date_Key_Tmp'].map(asia_stats['Low'])
        df.drop(columns=['Date_Key_Tmp'], inplace=True)
    
    # 2. Убедимся, что FVG 4H колонки рассчитаны перед этой функцией
    if 'FVG_4h_Type' not in df.columns:
        df['FVG_4h_Type'] = None
        df['FVG_4h_Top'] = np.nan
        df['FVG_4h_Bottom'] = np.nan

    # 3. Условия "Вблизи границы Азии"
    df['Bull_FVG_Below_AL'] = (df['FVG_4h_Type'] == 'BULL') & \
                              (df['Asia_Low'] >= df['FVG_4h_Top']) & \
                              ((df['Asia_Low'] - df['FVG_4h_Top']) <= tolerance)
                              
    df['Bear_FVG_Above_AH'] = (df['FVG_4h_Type'] == 'BEAR') & \
                              (df['Asia_High'] <= df['FVG_4h_Bottom']) & \
                              ((df['FVG_4h_Bottom'] - df['Asia_High']) <= tolerance)
                              
    # 4. Детект самого Sweep
    raw_sweep_al = df['Bull_FVG_Below_AL'] & (df['Low'] < df['Asia_Low']) & (df['Low'] <= df['FVG_4h_Top'])
    raw_sweep_ah = df['Bear_FVG_Above_AH'] & (df['High'] > df['Asia_High']) & (df['High'] >= df['FVG_4h_Bottom'])
    
    # 5. Fix for 0 trades: Даем окну "Sweep" прожить еще 3 свечи, 
    # чтобы структура успела смениться (Flip) на часовиках.
    df['Swept_AL_Into_FVG'] = raw_sweep_al.rolling(3, min_periods=1).max().fillna(0).astype(int)
    df['Swept_AH_Into_FVG'] = raw_sweep_ah.rolling(3, min_periods=1).max().fillna(0).astype(int)
    
    return df

def add_asian_box_breakout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Asian_Box_Long' in df.columns: return df
    df = df.copy()
    df['Asian_Box_Long'] = 0
    df['Asian_Box_Short'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    df['Asia_High_Tmp'] = df['High'].rolling(window=8).max().shift(1)
    df['Asia_Low_Tmp'] = df['Low'].rolling(window=8).min().shift(1)
    df['Asia_Range'] = df['Asia_High_Tmp'] - df['Asia_Low_Tmp']
    
    df['Signal_Candle_Close'] = df['Close']
    df['Signal_Candle_Open'] = df['Open']
    
    is_entry_time = (df.index.hour == 9)
    is_coiled = df['Asia_Range'].shift(1) < (0.25 * df['ATR_14D'].shift(1))
    
    long_breakout = (df['Signal_Candle_Close'].shift(1) > df['Asia_High_Tmp'].shift(1)) & \
                    (df['Signal_Candle_Close'].shift(1) > df['Signal_Candle_Open'].shift(1))
    
    short_breakout = (df['Signal_Candle_Close'].shift(1) < df['Asia_Low_Tmp'].shift(1)) & \
                     (df['Signal_Candle_Close'].shift(1) < df['Signal_Candle_Open'].shift(1))
    
    df.loc[is_entry_time & is_coiled & long_breakout, 'Asian_Box_Long'] = 1
    df.loc[is_entry_time & is_coiled & short_breakout, 'Asian_Box_Short'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Asia_High_Tmp', 'Asia_Low_Tmp', 'Asia_Range', 
                     'Signal_Candle_Close', 'Signal_Candle_Open'], inplace=True, errors='ignore')
    return df

def add_tokyo_trap_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Tokyo_Trap_Short' in df.columns: return df
    df = df.copy()
    df['Tokyo_Trap_Short'] = 0
    df['Tokyo_Trap_Long'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    df['Asia_Open'] = df['Open'].shift(8)
    df['Asia_Close'] = df['Close'].shift(1)
    
    is_london_open = (df.index.hour == 8)
    
    asia_rally = (df['Asia_Close'] - df['Asia_Open']) > (0.5 * df['ATR_14D'])
    asia_dump = (df['Asia_Open'] - df['Asia_Close']) > (0.5 * df['ATR_14D'])
    
    df.loc[is_london_open & asia_rally, 'Tokyo_Trap_Short'] = 1
    df.loc[is_london_open & asia_dump, 'Tokyo_Trap_Long'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Asia_Open', 'Asia_Close'], inplace=True, errors='ignore')
    return df

def add_london_true_trend_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'LO_True_Trend_Long' in df.columns: return df
    df = df.copy()
    df['LO_True_Trend_Long'] = 0
    df['LO_True_Trend_Short'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    is_10_am = (df.index.hour == 9)
    df['LO_Open_Price'] = df['Open'].shift(2)
    df['Current_10AM_Price'] = df['Open'] 
    
    trend_up = is_10_am & ((df['Current_10AM_Price'] - df['LO_Open_Price']) > (0.3 * df['ATR_14D']))
    trend_down = is_10_am & ((df['LO_Open_Price'] - df['Current_10AM_Price']) > (0.3 * df['ATR_14D']))
    
    df.loc[trend_up, 'LO_True_Trend_Long'] = 1
    df.loc[trend_down, 'LO_True_Trend_Short'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'LO_Open_Price', 'Current_10AM_Price'], inplace=True, errors='ignore')
    return df

def add_london_fix_fade_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Fix_Fade_Short' in df.columns: return df
    df = df.copy()
    df['Fix_Fade_Short'] = 0
    df['Fix_Fade_Short_SL'] = np.nan
    df['Fix_Fade_Long'] = 0
    df['Fix_Fade_Long_SL'] = np.nan

    df['Day_Key'] = df.index.normalize()
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()

    day_open = df.groupby('Day_Key')['Open'].transform('first')
    df['Day_Trend_Vector'] = df['Close'] - day_open
    df['Day_Distance'] = df['Day_Trend_Vector'].abs()

    exhaustion_reached = df['Day_Distance'] > (0.8 * df['ATR_14D'])
    is_fix_time = (df['UA_Hour'] == 18)
    is_valid_day = (df.index.dayofweek != 4) & (df.index.month != 12)

    day_high = df['High'].rolling(18, min_periods=1).max()
    day_low = df['Low'].rolling(18, min_periods=1).min()

    short_cond = is_fix_time & is_valid_day & exhaustion_reached & (df['Day_Trend_Vector'] > 0)
    long_cond = is_fix_time & is_valid_day & exhaustion_reached & (df['Day_Trend_Vector'] < 0)

    df.loc[short_cond, 'Fix_Fade_Short'] = 1
    df.loc[short_cond, 'Fix_Fade_Short_SL'] = day_high

    df.loc[long_cond, 'Fix_Fade_Long'] = 1
    df.loc[long_cond, 'Fix_Fade_Long_SL'] = day_low

    df.drop(columns=['Day_Key', 'Daily_Range', 'ATR_14D', 'Day_Trend_Vector', 'Day_Distance'], inplace=True, errors='ignore')
    return df

def add_london_pdh_pdl_sweep_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'First_LDN_PDL_Long' in df.columns: return df
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    
    fractal_low = (df.get('Confirmed_Fractal_Low', pd.Series(0, index=df.index)) == 1)
    fractal_high = (df.get('Confirmed_Fractal_High', pd.Series(0, index=df.index)) == 1)
    
    if 'Confirmed_Fractal_Low_Price' in df.columns and 'PDL' in df.columns:
        sweep_pdl = df['Confirmed_Fractal_Low_Price'] < df['PDL']
        sweep_pdh = df['Confirmed_Fractal_High_Price'] > df['PDH']
    else:
        sweep_pdl = pd.Series(False, index=df.index)
        sweep_pdh = pd.Series(False, index=df.index)
        
    df['LDN_Sweep_PDL_Long'] = is_london_eval & fractal_low & sweep_pdl
    df['LDN_Sweep_PDH_Short'] = is_london_eval & fractal_high & sweep_pdh
    
    df['Date'] = df.index.date
    df['Sweep_Trigger_Count'] = (df['LDN_Sweep_PDL_Long'] | df['LDN_Sweep_PDH_Short']).groupby(df['Date']).cumsum()
    
    df['First_LDN_PDL_Long'] = (df['LDN_Sweep_PDL_Long'] & (df['Sweep_Trigger_Count'] == 1)).astype(int)
    df['First_LDN_PDH_Short'] = (df['LDN_Sweep_PDH_Short'] & (df['Sweep_Trigger_Count'] == 1)).astype(int)
    
    df.drop(columns=['Date', 'Sweep_Trigger_Count', 'LDN_Sweep_PDL_Long', 'LDN_Sweep_PDH_Short'], inplace=True)
    return df

def add_london_counter_fractal_context(df: pd.DataFrame,events: list = None) -> pd.DataFrame:
    if 'First_LDN_Counter_Low' in df.columns: return df
    is_london_eval = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 14)
    
    bull_trend = (df.get('1D_Swing_Bullish', pd.Series(0, index=df.index)) == 1)
    bear_trend = (df.get('1D_Swing_Bearish', pd.Series(0, index=df.index)) == 1)
    
    fractal_low = (df.get('Confirmed_Fractal_Low', pd.Series(0, index=df.index)) == 1)
    fractal_high = (df.get('Confirmed_Fractal_High', pd.Series(0, index=df.index)) == 1)

    df['LDN_Counter_Low_Trap'] = is_london_eval & bear_trend & fractal_low
    df['LDN_Counter_High_Trap'] = is_london_eval & bull_trend & fractal_high

    df['Date'] = df.index.date
    df['Trap_Trigger_Count'] = (df['LDN_Counter_Low_Trap'] | df['LDN_Counter_High_Trap']).groupby(df['Date']).cumsum()
    
    df['First_LDN_Counter_Low'] = (df['LDN_Counter_Low_Trap'] & (df['Trap_Trigger_Count'] == 1)).astype(int)
    df['First_LDN_Counter_High'] = (df['LDN_Counter_High_Trap'] & (df['Trap_Trigger_Count'] == 1)).astype(int)
    
    df.drop(columns=['Date', 'Trap_Trigger_Count', 'LDN_Counter_Low_Trap', 'LDN_Counter_High_Trap'], inplace=True)
    return df

def add_ny_expansion_context(df: pd.DataFrame,events: list = None) -> pd.DataFrame:
    if 'NY_Opened_In_Asia_Range' in df.columns: return df
    if 'Asia_High' not in df.columns:
        df['Date_Key_Tmp'] = df.index.date
        is_asia = (df['UA_Hour'] >= 0) & (df['UA_Hour'] <= 8)
        asia_stats = df[is_asia].groupby('Date_Key_Tmp').agg({'High': 'max', 'Low': 'min'})
        df['Asia_High'] = df['Date_Key_Tmp'].map(asia_stats['High'])
        df['Asia_Low'] = df['Date_Key_Tmp'].map(asia_stats['Low'])
        df.drop(columns=['Date_Key_Tmp'], inplace=True)

    # Replace with this bulletproof mapping:
    df['Date_Tmp'] = df.index.date
    ny_opens = df[df['UA_Hour'] == 15].groupby('Date_Tmp')['Open'].first()
    df['NY_Open_Price'] = df['Date_Tmp'].map(ny_opens)
    df.drop(columns=['Date_Tmp'], inplace=True)

    df['NY_Opened_In_Asia_Range'] = (df['NY_Open_Price'] < df['Asia_High']) & \
                                     (df['NY_Open_Price'] > df['Asia_Low'])
    
    is_ny_session = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 23)
    
    df['NY_Sweep_Asia_Low'] = is_ny_session & \
                            (df['Low'] < df['Asia_Low']) & \
                            (df['Low'].shift(1) >= df['Asia_Low'])

    df['NY_Sweep_Asia_High'] = is_ny_session & \
                            (df['High'] > df['Asia_High']) & \
                            (df['High'].shift(1) <= df['Asia_High'])

    df['NY_Opened_In_Asia_Range'] = df['NY_Opened_In_Asia_Range'].fillna(False).astype(int)
    df['NY_Sweep_Asia_High'] = df['NY_Sweep_Asia_High'].astype(int)
    df['NY_Sweep_Asia_Low'] = df['NY_Sweep_Asia_Low'].astype(int)
    
    return df

def add_ny_sr_touch_context(df: pd.DataFrame, events: list = None, tolerance_pips: int = 15) -> pd.DataFrame:
    if 'NY_First_Touch_Res' in df.columns: return df
    PIP = 0.0001
    tol = tolerance_pips * PIP
    
    df['Major_Resistance'] = df['High'].rolling(window=24*20).max().shift(1)
    df['Major_Support'] = df['Low'].rolling(window=24*20).min().shift(1)
    
    is_ny = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 22)
    
    df['NY_Touch_Res'] = is_ny & (df['High'] >= (df['Major_Resistance'] - tol))
    df['NY_Touch_Sup'] = is_ny & (df['Low'] <= (df['Major_Support'] + tol))
    
    df['Date'] = df.index.date
    df['Res_Daily_Touches'] = df.groupby('Date')['NY_Touch_Res'].cumsum()
    df['Sup_Daily_Touches'] = df.groupby('Date')['NY_Touch_Sup'].cumsum()
    
    df['NY_First_Touch_Res'] = df['NY_Touch_Res'] & (df['Res_Daily_Touches'] == 1)
    df['NY_First_Touch_Sup'] = df['NY_Touch_Sup'] & (df['Sup_Daily_Touches'] == 1)
    
    df.drop(columns=['Date', 'Res_Daily_Touches', 'Sup_Daily_Touches'], inplace=True)
    df['NY_First_Touch_Res'] = df['NY_First_Touch_Res'].astype(int)
    df['NY_First_Touch_Sup'] = df['NY_First_Touch_Sup'].astype(int)
    
    return df

def add_asian_sr_alignment_context(df: pd.DataFrame, max_dist_pips: int = 15) -> pd.DataFrame:
    if 'First_False_Break_Res' in df.columns: return df
    PIP = 0.0001
    dist = max_dist_pips * PIP

    # Безопасность на случай если еще не вызвано
    if 'Asia_High' not in df.columns or 'Major_Resistance' not in df.columns:
        df['First_False_Break_Res'] = 0
        df['First_False_Break_Sup'] = 0
        return df

    df['Asia_Res_Aligned'] = (abs(df['Asia_High'] - df['Major_Resistance']) <= dist)
    df['Asia_Sup_Aligned'] = (abs(df['Asia_Low'] - df['Major_Support']) <= dist)

    is_active_session = (df['UA_Hour'] >= 10) & (df['UA_Hour'] <= 21)

    df['Break_Aligned_Res'] = is_active_session & df['Asia_Res_Aligned'] & (df['High'] > df['Asia_High'])
    df['Break_Aligned_Sup'] = is_active_session & df['Asia_Sup_Aligned'] & (df['Low'] < df['Asia_Low'])

    df['Date'] = df.index.date
    df['Res_Break_Count'] = df.groupby('Date')['Break_Aligned_Res'].cumsum()
    df['Sup_Break_Count'] = df.groupby('Date')['Break_Aligned_Sup'].cumsum()

    df['First_False_Break_Res'] = df['Break_Aligned_Res'] & (df['Res_Break_Count'] == 1)
    df['First_False_Break_Sup'] = df['Break_Aligned_Sup'] & (df['Sup_Break_Count'] == 1)

    df.drop(columns=['Date', 'Res_Break_Count', 'Sup_Break_Count'], inplace=True)
    df['First_False_Break_Res'] = df['First_False_Break_Res'].fillna(False).astype(int)
    df['First_False_Break_Sup'] = df['First_False_Break_Sup'].fillna(False).astype(int)

    return df

def add_ny_continuation_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'NY_Cont_Long' in df.columns: return df
    df = df.copy()
    df['NY_Cont_Long'] = 0
    df['NY_Cont_Short'] = 0

    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()

    df['Day_Key'] = df.index.normalize()

    open_10 = df[df['UA_Hour'] == 10].groupby('Day_Key')['Open'].first()
    close_15 = df[df['UA_Hour'] == 15].groupby('Day_Key')['Close'].last()

    df['LDN_Open_10'] = df['Day_Key'].map(open_10)
    df['LDN_Close_15'] = df['Day_Key'].map(close_15)

    df['LDN_Vector'] = df['LDN_Close_15'] - df['LDN_Open_10']
    df['LDN_Distance'] = df['LDN_Vector'].abs()

    strong_trend = df['LDN_Distance'] > (0.5 * df['ATR_14D'])
    is_entry_time = (df['UA_Hour'] == 16)

    long_cond = is_entry_time & strong_trend & (df['LDN_Vector'] > 0)
    short_cond = is_entry_time & strong_trend & (df['LDN_Vector'] < 0)

    df.loc[long_cond, 'NY_Cont_Long'] = 1
    df.loc[short_cond, 'NY_Cont_Short'] = 1

    df.drop(columns=['Day_Key', 'LDN_Open_10', 'LDN_Close_15', 'LDN_Vector', 'LDN_Distance', 'Daily_Range', 'ATR_14D'], inplace=True, errors='ignore')
    return df

def add_ny_news_sweep_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'NY_Sweep_Short' in df.columns: return df
    df = df.copy()
    df['NY_Sweep_Short'] = 0
    df['NY_Sweep_Short_SL'] = np.nan
    df['NY_Sweep_Long'] = 0
    df['NY_Sweep_Long_SL'] = np.nan

    df['Day_Key'] = df.index.normalize()

    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    asia_stats = df[asia_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['Asia_High_Tmp'] = df['Day_Key'].map(asia_stats['High'])
    df['Asia_Low_Tmp'] = df['Day_Key'].map(asia_stats['Low'])

    ldn_mask = (df['UA_Hour'] >= 10) & (df['UA_Hour'] < 15)
    ldn_stats = df[ldn_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['LDN_High'] = df['Day_Key'].map(ldn_stats['High'])
    df['LDN_Low'] = df['Day_Key'].map(ldn_stats['Low'])

    df['LDN_Swept_Asia_High'] = df['LDN_High'] > df['Asia_High_Tmp']
    df['LDN_Swept_Asia_Low'] = df['LDN_Low'] < df['Asia_Low_Tmp']

    is_hunting_zone = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 17)
    is_not_friday = (df.index.dayofweek != 4)
    is_not_december = (df.index.month != 12)
    valid_time = is_hunting_zone & is_not_friday & is_not_december

    just_returned_high = (df['Close'] < df['LDN_High']) & (df['Close'].shift(1) >= df['LDN_High'])
    pinbar_high = (df['High'] > df['LDN_High']) & (df['Close'] < df['LDN_High'])
    sweep_high = (just_returned_high | pinbar_high) & (df['Close'] < df['Open'])

    just_returned_low = (df['Close'] > df['LDN_Low']) & (df['Close'].shift(1) <= df['LDN_Low'])
    pinbar_low = (df['Low'] < df['LDN_Low']) & (df['Close'] > df['LDN_Low'])
    sweep_low = (just_returned_low | pinbar_low) & (df['Close'] > df['Open'])

    valid_sweep_high = sweep_high & (~df['LDN_Swept_Asia_High'])
    valid_sweep_low = sweep_low & (~df['LDN_Swept_Asia_Low'])

    recent_highest = df['High'].rolling(3, min_periods=1).max()
    recent_lowest = df['Low'].rolling(3, min_periods=1).min()

    df.loc[valid_time & valid_sweep_high, 'NY_Sweep_Short'] = 1
    df.loc[valid_time & valid_sweep_high, 'NY_Sweep_Short_SL'] = recent_highest

    df.loc[valid_time & valid_sweep_low, 'NY_Sweep_Long'] = 1
    df.loc[valid_time & valid_sweep_low, 'NY_Sweep_Long_SL'] = recent_lowest

    df.drop(columns=['Day_Key', 'Asia_High_Tmp', 'Asia_Low_Tmp', 'LDN_High', 'LDN_Low', 'LDN_Swept_Asia_High', 'LDN_Swept_Asia_Low'], inplace=True, errors='ignore')
    return df

def add_friday_reversal_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Friday_Reversal_Short' in df.columns: return df
    df = df.copy()
    df['Friday_Reversal_Short'] = 0
    df['Friday_Reversal_Long'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    df['Week_Open'] = df['Open'].shift(112)
    
    is_friday_16 = (df.index.dayofweek == 4) & (df.index.hour == 16)
    
    weekly_rally = df['Close'] - df['Week_Open']
    weekly_dump = df['Week_Open'] - df['Close']
    
    short_condition = is_friday_16 & (weekly_rally > (2 * df['ATR_14D']))
    long_condition = is_friday_16 & (weekly_dump > (2 * df['ATR_14D']))
    
    df.loc[short_condition, 'Friday_Reversal_Short'] = 1
    df.loc[long_condition, 'Friday_Reversal_Long'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Week_Open'], inplace=True, errors='ignore')
    return df

def add_weekend_gap_context(df: pd.DataFrame,events: list = None) -> pd.DataFrame:
    if 'Gap_Up_Fade_Short' in df.columns: return df
    time_diff = df.index.to_series().diff()
    is_week_open = time_diff > pd.Timedelta(hours=24)
    gap_pips = (df['Open'] - df['Close'].shift(1)) * 10000
    
    df['Gap_Up_Fade_Short'] = ((is_week_open) & (gap_pips >= 40)).astype(int)
    df['Gap_Down_Fade_Long'] = ((is_week_open) & (gap_pips <= -40)).astype(int)
    
    return df

def add_monday_gap_reversion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Monday_Reversion_Short' in df.columns: return df
    df = df.copy()
    df['Monday_Reversion_Short'] = 0
    df['Monday_Reversion_Long'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    df['Weekly_Delta'] = df['Close'] - df['Open'].shift(120)
    is_monday_open = (df.index.dayofweek == 0) & (df.index.hour == 0)
    
    short_cond = is_monday_open & (df['Weekly_Delta'] > (2 * df['ATR_14D']))
    long_cond = is_monday_open & (df['Weekly_Delta'] < -(2 * df['ATR_14D']))
    
    df.loc[short_cond, 'Monday_Reversion_Short'] = 1
    df.loc[long_cond, 'Monday_Reversion_Long'] = 1
    
    df.drop(columns=['Daily_Range', 'ATR_14D', 'Weekly_Delta'], inplace=True, errors='ignore')
    return df

def add_turnaround_tuesday_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Tuesday_Resumption_Long' in df.columns: return df
    df = df.copy()
    df['Tuesday_Resumption_Long'] = 0
    df['Tuesday_Resumption_Short'] = 0
    
    df['Prev_Week_Open'] = df['Open'].shift(144)
    df['Prev_Week_Close'] = df['Close'].shift(24) 
    
    df['Monday_Open'] = df['Open'].shift(24)
    df['Monday_Close'] = df['Close'].shift(1)
    
    is_tuesday_london_open = (df.index.dayofweek == 1) & (df.index.hour == 8)
    
    long_cond = is_tuesday_london_open & (df['Prev_Week_Close'] > df['Prev_Week_Open']) & (df['Monday_Close'] < df['Monday_Open'])
    short_cond = is_tuesday_london_open & (df['Prev_Week_Close'] < df['Prev_Week_Open']) & (df['Monday_Close'] > df['Monday_Open'])
    
    df.loc[long_cond, 'Tuesday_Resumption_Long'] = 1
    df.loc[short_cond, 'Tuesday_Resumption_Short'] = 1
    
    df.drop(columns=['Prev_Week_Open', 'Prev_Week_Close', 'Monday_Open', 'Monday_Close'], inplace=True, errors='ignore')
    return df

def add_wednesday_fakeout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Wed_Fakeout_Long' in df.columns: return df
    df = df.copy()
    df['Wed_Fakeout_Long'] = 0
    df['Wed_Fakeout_Short'] = 0
    
    df['Date'] = df.index.date
    daily_hl = df.groupby('Date').agg({'High': 'max', 'Low': 'min'})
    prev_day_hl = daily_hl.shift(1)
    
    df['Prev_Day_High'] = df['Date'].map(prev_day_hl['High'])
    df['Prev_Day_Low'] = df['Date'].map(prev_day_hl['Low'])
    
    is_wednesday = df.index.dayofweek == 2
    
    sweep_high = (df['High'] > df['Prev_Day_High']) & (df['Close'] < df['Prev_Day_High']) & (df['Close'] < df['Open'])
    short_cond = is_wednesday & sweep_high
    
    sweep_low = (df['Low'] < df['Prev_Day_Low']) & (df['Close'] > df['Prev_Day_Low']) & (df['Close'] > df['Open'])
    long_cond = is_wednesday & sweep_low
    
    df.loc[short_cond, 'Wed_Fakeout_Short'] = 1
    df.loc[long_cond, 'Wed_Fakeout_Long'] = 1
    
    df.drop(columns=['Date', 'Prev_Day_High', 'Prev_Day_Low'], inplace=True, errors='ignore')
    return df

def add_thursday_expansion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Thursday_Trend_Long' in df.columns: return df
    df = df.copy()
    df['Thursday_Trend_Long'] = 0
    df['Thursday_Trend_Short'] = 0
    
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()
    
    df['Week_Open'] = df['Open'].shift(80)
    df['Wed_Close'] = df['Close'].shift(9) 
    
    is_thursday_london_open = (df.index.dayofweek == 3) & (df.index.hour == 8)
    
    trend_up = (df['Wed_Close'] - df['Week_Open']) > (0.5 * df['ATR_14D'])
    trend_down = (df['Week_Open'] - df['Wed_Close']) > (0.5 * df['ATR_14D'])
    
    long_cond = is_thursday_london_open & trend_up
    short_cond = is_thursday_london_open & trend_down
    
    df.loc[long_cond, 'Thursday_Trend_Long'] = 1
    df.loc[short_cond, 'Thursday_Trend_Short'] = 1
    
    df.drop(columns=['Week_Open', 'Wed_Close', 'Daily_Range', 'ATR_14D'], inplace=True, errors='ignore')
    return df

def add_pure_algo_vol_crush_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    if 'Algo_Vol_Crush_Short' in df.columns: return df
    df = df.copy()
    
    df['TR_Local'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    df['TR_Mean'] = df['TR_Local'].rolling(window=720).mean()
    df['TR_Std'] = df['TR_Local'].rolling(window=720).std()
    
    df['Is_3SD_Spike'] = df['TR_Local'] > (df['TR_Mean'] + 3 * df['TR_Std'])
    
    if 'Realized_Vol' not in df.columns:
        df['Realized_Vol'] = df['TR_Local'].rolling(window=4).mean() * 10000
        
    df['Algo_Vol_Crush_Short'] = df['Is_3SD_Spike'].shift(1).fillna(0).astype(int)

    df.drop(columns=['TR_Local', 'TR_Mean', 'TR_Std', 'Is_3SD_Spike'], inplace=True, errors='ignore')
    return df