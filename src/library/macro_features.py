import pandas as pd
import numpy as np

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