import numpy as np
import pandas as pd

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

def add_london_fix_fade_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_050: The London Fix Fade (Разворот на фиксинге).
    В 18:00 UA проверяем, был ли день истощающим трендовым (>0.8 ATR).
    Если да — входим против тренда на закрытии позиций лондонскими банками.
    """
    df = df.copy()
    df['Fix_Fade_Short'] = 0
    df['Fix_Fade_Short_SL'] = np.nan
    df['Fix_Fade_Long'] = 0
    df['Fix_Fade_Long_SL'] = np.nan

    df['Day_Key'] = df.index.normalize()

    # 1. Локальный расчет 14-дневного ATR
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()

    # 2. Вычисляем вектор дня (Цена сейчас минус Открытие дня)
    # Используем transform('first'), чтобы взять самую первую цену открытия для каждой даты
    day_open = df.groupby('Day_Key')['Open'].transform('first')
    df['Day_Trend_Vector'] = df['Close'] - day_open
    df['Day_Distance'] = df['Day_Trend_Vector'].abs()

    # 3. Фильтр истощения (Тренд должен пройти больше 80% от среднего дневного движения)
    exhaustion_reached = df['Day_Distance'] > (0.8 * df['ATR_14D'])

    # 4. Время Лондонского Фиксинга (18:00 по Киеву / 16:00 UTC)
    is_fix_time = (df['UA_Hour'] == 18)
    
    # Исключаем Пятницу (фиксация может быть непредсказуемой) и Декабрь
    is_valid_day = (df.index.dayofweek != 4) & (df.index.month != 12)

    # 5. Экстремумы дня для Стоп-Лосса (Максимум/минимум за последние 18 часов)
    day_high = df['High'].rolling(18, min_periods=1).max()
    day_low = df['Low'].rolling(18, min_periods=1).min()

    # 6. ЛОГИКА СИГНАЛОВ
    # Шортим, если день был сильно бычьим (Вектор > 0)
    short_cond = is_fix_time & is_valid_day & exhaustion_reached & (df['Day_Trend_Vector'] > 0)
    
    # Лонгуем, если день был сильно медвежьим (Вектор < 0)
    long_cond = is_fix_time & is_valid_day & exhaustion_reached & (df['Day_Trend_Vector'] < 0)

    # 7. Запись
    df.loc[short_cond, 'Fix_Fade_Short'] = 1
    df.loc[short_cond, 'Fix_Fade_Short_SL'] = day_high

    df.loc[long_cond, 'Fix_Fade_Long'] = 1
    df.loc[long_cond, 'Fix_Fade_Long_SL'] = day_low

    # Очистка памяти
    df.drop(columns=['Day_Key', 'Daily_Range', 'ATR_14D', 'Day_Trend_Vector', 'Day_Distance'], inplace=True, errors='ignore')
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

def add_ny_continuation_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_048: NY Overlap Continuation (Pure Statistical Edge).
    Триггер в 16:00 UA, если Лондон показал сильный вектор (> 0.5 ATR).
    """
    df = df.copy()
    df['NY_Cont_Long'] = 0
    df['NY_Cont_Short'] = 0

    # 1. Локальный ATR для автономности
    df['Daily_Range'] = df['High'].rolling(24).max() - df['Low'].rolling(24).min()
    df['ATR_14D'] = df['Daily_Range'].rolling(336).mean()

    # 2. Безопасные даты
    df['Day_Key'] = df.index.normalize()

    # 3. Фиксируем цены: Открытие Лондона (10:00) и Открытие NY (15:00)
    open_10 = df[df['UA_Hour'] == 10].groupby('Day_Key')['Open'].first()
    close_15 = df[df['UA_Hour'] == 15].groupby('Day_Key')['Close'].last()

    df['LDN_Open_10'] = df['Day_Key'].map(open_10)
    df['LDN_Close_15'] = df['Day_Key'].map(close_15)

    # 4. Вектор Лондона (Направление и Дистанция)
    df['LDN_Vector'] = df['LDN_Close_15'] - df['LDN_Open_10']
    df['LDN_Distance'] = df['LDN_Vector'].abs()

    # ФИЛЬТР: Вектор должен быть больше 0.5 дневного ATR
    strong_trend = df['LDN_Distance'] > (0.5 * df['ATR_14D'])

    # 5. Время входа: ровно 16:00 UA
    is_entry_time = (df['UA_Hour'] == 16)

    # 6. Логика (только по направлению сильного вектора)
    long_cond = is_entry_time & strong_trend & (df['LDN_Vector'] > 0)
    short_cond = is_entry_time & strong_trend & (df['LDN_Vector'] < 0)

    # 7. Запись сигналов
    df.loc[long_cond, 'NY_Cont_Long'] = 1
    df.loc[short_cond, 'NY_Cont_Short'] = 1

    # Очистка мусора
    df.drop(columns=['Day_Key', 'LDN_Open_10', 'LDN_Close_15', 'LDN_Vector', 'LDN_Distance', 'Daily_Range', 'ATR_14D'], inplace=True, errors='ignore')
    return df

def add_ny_news_sweep_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    """
    H_049 v2: NY News Sweep (Protected Extremes Filter).
    Нью-Йорк НЕ пробивает те экстремумы Лондона, которые уже сняли ликвидность с Азии.
    """
    df = df.copy()
    df['NY_Sweep_Short'] = 0
    df['NY_Sweep_Short_SL'] = np.nan
    df['NY_Sweep_Long'] = 0
    df['NY_Sweep_Long_SL'] = np.nan

    df['Day_Key'] = df.index.normalize()

    # 1. КОРОБКА АЗИИ (00:00 - 06:59)
    asia_mask = (df.index.hour >= 0) & (df.index.hour < 7)
    asia_stats = df[asia_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['Asia_High'] = df['Day_Key'].map(asia_stats['High'])
    df['Asia_Low'] = df['Day_Key'].map(asia_stats['Low'])

    # 2. КОРОБКА ЛОНДОНА (10:00 - 14:59 UA)
    ldn_mask = (df['UA_Hour'] >= 10) & (df['UA_Hour'] < 15)
    ldn_stats = df[ldn_mask].groupby('Day_Key').agg({'High': 'max', 'Low': 'min'})
    df['LDN_High'] = df['Day_Key'].map(ldn_stats['High'])
    df['LDN_Low'] = df['Day_Key'].map(ldn_stats['Low'])

    # 3. ФИЛЬТР "ЗАЩИЩЕННЫХ ЭКСТРЕМУМОВ"
    # Если Лондон пробил Азию, этот экстремум защищен крупным капиталом.
    df['LDN_Swept_Asia_High'] = df['LDN_High'] > df['Asia_High']
    df['LDN_Swept_Asia_Low'] = df['LDN_Low'] < df['Asia_Low']

    # 4. ВРЕМЯ НЬЮ-ЙОРКА (15:00 - 17:00 UA)
    is_hunting_zone = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 17)
    is_not_friday = (df.index.dayofweek != 4)
    is_not_december = (df.index.month != 12)
    valid_time = is_hunting_zone & is_not_friday & is_not_december

    # 5. БАЗОВАЯ ЛОГИКА SWEEP (Охота за стопами Лондона)
    just_returned_high = (df['Close'] < df['LDN_High']) & (df['Close'].shift(1) >= df['LDN_High'])
    pinbar_high = (df['High'] > df['LDN_High']) & (df['Close'] < df['LDN_High'])
    sweep_high = (just_returned_high | pinbar_high) & (df['Close'] < df['Open'])

    just_returned_low = (df['Close'] > df['LDN_Low']) & (df['Close'].shift(1) <= df['LDN_Low'])
    pinbar_low = (df['Low'] < df['LDN_Low']) & (df['Close'] > df['LDN_Low'])
    sweep_low = (just_returned_low | pinbar_low) & (df['Close'] > df['Open'])

    # 6. СМАРТ-ФИЛЬТР (Исключаем защищенные уровни)
    # Шортим ложный пробой хая Лондона ТОЛЬКО если Лондон НЕ снимал хай Азии
    valid_sweep_high = sweep_high & (~df['LDN_Swept_Asia_High'])
    
    # Лонгуем ложный пробой лоу Лондона ТОЛЬКО если Лондон НЕ снимал лоу Азии
    valid_sweep_low = sweep_low & (~df['LDN_Swept_Asia_Low'])

    # 7. СТОП-ЛОССЫ
    recent_highest = df['High'].rolling(3, min_periods=1).max()
    recent_lowest = df['Low'].rolling(3, min_periods=1).min()

    # 8. ЗАПИСЬ СИГНАЛОВ
    df.loc[valid_time & valid_sweep_high, 'NY_Sweep_Short'] = 1
    df.loc[valid_time & valid_sweep_high, 'NY_Sweep_Short_SL'] = recent_highest

    df.loc[valid_time & valid_sweep_low, 'NY_Sweep_Long'] = 1
    df.loc[valid_time & valid_sweep_low, 'NY_Sweep_Long_SL'] = recent_lowest

    # Очистка
    df.drop(columns=['Day_Key', 'Asia_High', 'Asia_Low', 'LDN_High', 'LDN_Low', 'LDN_Swept_Asia_High', 'LDN_Swept_Asia_Low'], inplace=True, errors='ignore')
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