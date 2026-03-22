import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# ХЕЛПЕР: УМНЫЙ ПОИСК ДАТ (Игнорируем минуты)
# ==========================================
def _get_event_dates(events, keywords: list) -> list:
    """Загружает JSON используя абсолютный путь, гарантируя чтение Категорий."""
    keywords_lower = [k.lower() for k in keywords]
    
    # Расширяем ключевые слова для надежности
    if any(k in keywords_lower for k in ['unemployment', 'jobless']):
        keywords_lower.append('unemp')
    if any(k in keywords_lower for k in ['gilt', 'bond', 'debt']):
        keywords_lower.extend(['shock', 'geopolitical', 'political', 'coup', 'war', 'sovereign'])
    if any(k in keywords_lower for k in ['rate', 'policy']):
        keywords_lower.extend(['fomc', 'boe', 'hike', 'cut', 'interest'])
    if 'retail' in keywords_lower:
        keywords_lower.extend(['sales', 'consumer'])

    events_list = []
    
    # 1. BULLETPROOF ABSOLUTE PATH
    # Computes exact path: src/library -> src -> hypothesis_lab -> data/macro_events.json
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    file_path = project_root / 'data' / 'macro_events.json'
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
            for cat, evt_list in raw_json.items():
                for e in evt_list:
                    if isinstance(e, dict):
                        e['Category'] = cat
                        events_list.append(e)
    else:
        # 2. BULLETPROOF DATAFRAME UNWRAP (If file mysteriously vanishes)
        print(f"⚠️ Warning: Could not find {file_path}. Using DataFrame fallback.")
        if isinstance(events, pd.DataFrame):
            for col in events.columns:
                for item in events[col].dropna():
                    if isinstance(item, dict):
                        item['Category'] = col
                        events_list.append(item)
        elif isinstance(events, dict):
            for cat, evt_list in events.items():
                for e in evt_list:
                    if isinstance(e, dict):
                        e['Category'] = cat
                        events_list.append(e)

    dates = set()
    for e in events_list:
        if not isinstance(e, dict): continue
        
        name_lower = str(e.get('name', e.get('Event', ''))).lower()
        category_lower = str(e.get('Category', '')).lower().replace('_', ' ')
        
        # Ищем совпадения в ИМЕНИ или КАТЕГОРИИ
        if any(k in name_lower or k in category_lower for k in keywords_lower):
            dt_raw = e.get('start_date', e.get('Date', ''))
            if dt_raw:
                date_part = str(dt_raw).split('T')[0].split(' ')[0]
                try:
                    dates.add(pd.to_datetime(date_part).date())
                except:
                    pass
                    
    # Отладочный принт (покажет в терминале, сколько дат нашел каждый макро-ивент)
    # print(f"   🔎 [MACRO] {keywords[0].upper()} -> Найдено {len(dates)} дат.")
    return list(dates)

# ==========================================
# 1. NON-FARM PAYROLLS (NFP)
# ==========================================
def add_nfp_divergence_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    for col in ['NFP_Fade_Long', 'NFP_Fade_Short', 'NFP_Resumption_Long', 'NFP_Resumption_Short']:
        df[col] = 0
        
    nfp_dates = _get_event_dates(events, ['nfp', 'non farm', 'payrolls', 'employment'])
    if not nfp_dates: return df

    date_strs = [str(d) for d in nfp_dates]
    is_nfp_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
    is_active = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 18)
    
    bullish_candle = df['Close'] > df['Open']
    bearish_candle = df['Close'] < df['Open']

    valid_mask = is_nfp_day & is_active
    
    df.loc[valid_mask & bearish_candle, 'NFP_Fade_Long'] = 1
    df.loc[valid_mask & bullish_candle, 'NFP_Fade_Short'] = 1
    df.loc[valid_mask & bullish_candle, 'NFP_Resumption_Long'] = 1
    df.loc[valid_mask & bearish_candle, 'NFP_Resumption_Short'] = 1
    
    return df

def add_nfp_revision_trap_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['NFP_Revision_Trap'] = 0
    
    nfp_dates = _get_event_dates(events, ['nfp', 'non farm'])
    if not nfp_dates: return df

    date_strs = [str(d) for d in nfp_dates]
    is_nfp_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
    is_active = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 18)
    
    df.loc[is_nfp_day & is_active, 'NFP_Revision_Trap'] = 1
    return df

# ==========================================
# 2. CPI / INFLATION
# ==========================================
def add_uk_cpi_momentum_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['CPI_Momentum_Long'] = 0
    df['CPI_Momentum_Short'] = 0
    
    cpi_dates = _get_event_dates(events, ['uk cpi', 'british cpi'])
    if not cpi_dates: return df

    date_strs = [str(d) for d in cpi_dates]
    is_cpi_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
    is_active = (df['UA_Hour'] >= 9) & (df['UA_Hour'] <= 12)
    
    df.loc[is_cpi_day & is_active & (df['Close'] > df['Open']), 'CPI_Momentum_Long'] = 1
    df.loc[is_cpi_day & is_active & (df['Close'] < df['Open']), 'CPI_Momentum_Short'] = 1
    
    return df

def add_uk_us_cpi_divergence_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Macro_CPI_Div_Long'] = 0
    df['Macro_CPI_Div_Short'] = 0
    cpi_dates = _get_event_dates(events, ['cpi'])
    
    if cpi_dates:
        date_strs = [str(d) for d in cpi_dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        df.loc[is_day & is_active & (df['Close'] > df['Open']), 'Macro_CPI_Div_Long'] = 1
        df.loc[is_day & is_active & (df['Close'] < df['Open']), 'Macro_CPI_Div_Short'] = 1
    return df

def add_cpi_match_mean_reversion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['CPI_Match_Reversion'] = 0
    cpi_dates = _get_event_dates(events, ['cpi'])
    if cpi_dates:
        date_strs = [str(d) for d in cpi_dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        df.loc[is_day & (df['UA_Hour'] == 16), 'CPI_Match_Reversion'] = 1
    return df

# ==========================================
# 3. CENTRAL BANKS (FOMC / BoE)
# ==========================================
def add_fomc_sell_the_news_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['FOMC_Sell_News_Long'] = 0
    df['FOMC_Sell_News_Short'] = 0
    
    fomc_dates = _get_event_dates(events, ['fomc', 'fed', 'interest rate'])
    if fomc_dates:
        date_strs = [str(d) for d in fomc_dates]
        is_fomc_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = (df['UA_Hour'] >= 21) & (df['UA_Hour'] <= 23)
        df.loc[is_fomc_day & is_active & (df['Close'] < df['Open']), 'FOMC_Sell_News_Long'] = 1
        df.loc[is_fomc_day & is_active & (df['Close'] > df['Open']), 'FOMC_Sell_News_Short'] = 1
    return df

def add_boe_hawkish_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['BoE_Hawkish'] = 0
    boe_dates = _get_event_dates(events, ['boe', 'bank of england'])
    if boe_dates:
        date_strs = [str(d) for d in boe_dates]
        is_boe_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        df.loc[is_boe_day & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 16), 'BoE_Hawkish'] = 1
    return df

def add_cb_divergence_state_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['CB_Divergence_Long'] = 0
    df['CB_Divergence_Short'] = 0
    dates = _get_event_dates(events, ['rate', 'policy'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = is_day & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        df.loc[is_active & (df['Close'] > df['Open']), 'CB_Divergence_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'CB_Divergence_Short'] = 1
    return df

def add_boe_tone_shift_proxy_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['BoE_Tone_Shift_Long'] = 0
    df['BoE_Tone_Shift_Short'] = 0
    dates = _get_event_dates(events, ['boe', 'bailey'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = is_day & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 17)
        df.loc[is_active & (df['Close'] > df['Open']), 'BoE_Tone_Shift_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'BoE_Tone_Shift_Short'] = 1
    return df

# ==========================================
# 4. OTHER MACRO (Retail Sales, Unemp)
# ==========================================
def add_retail_sales_divergence_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Retail_Div_Long'] = 0
    df['Retail_Div_Short'] = 0
    dates = _get_event_dates(events, ['retail'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = is_day & (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 18)
        df.loc[is_active & (df['Close'] > df['Open']), 'Retail_Div_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Retail_Div_Short'] = 1
    return df

def add_unemp_fakeout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Unemp_Fakeout_Long'] = 0
    df['Unemp_Fakeout_Short'] = 0
    dates = _get_event_dates(events, ['unemployment', 'jobless'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = is_day & (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 17)
        df.loc[is_active & (df['Close'] < df['Open']), 'Unemp_Fakeout_Long'] = 1
        df.loc[is_active & (df['Close'] > df['Open']), 'Unemp_Fakeout_Short'] = 1
    return df

# ==========================================
# 5. GEOPOLITICAL / SHOCKS
# ==========================================
def add_geopolitical_shock_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Geo_Shock_Long'] = 0
    df['Geo_Shock_Short'] = 0
    
    df['Z_Vol'] = (df['High'] - df['Low']).rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-5))
    is_shock = df['Z_Vol'] > 3.0
    
    df.loc[is_shock & (df['Close'] > df['Open']), 'Geo_Shock_Long'] = 1
    df.loc[is_shock & (df['Close'] < df['Open']), 'Geo_Shock_Short'] = 1
    df.drop(columns=['Z_Vol'], inplace=True)
    return df

def add_election_volatility_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Election_Vol_Crush_Long'] = 0
    df['Election_Vol_Crush_Short'] = 0
    dates = _get_event_dates(events, ['election', 'vote'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_active = df.index.strftime('%Y-%m-%d').isin(date_strs)
        df.loc[is_active & (df['Close'] > df['Open']), 'Election_Vol_Crush_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Election_Vol_Crush_Short'] = 1
    return df

def add_uk_political_shock_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['UK_Shock_Cont_Long'] = 0
    df['UK_Shock_Cont_Short'] = 0
    dates = _get_event_dates(events, ['parliament', 'pm', 'minister'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_active = df.index.strftime('%Y-%m-%d').isin(date_strs)
        df.loc[is_active & (df['Close'] > df['Open']), 'UK_Shock_Cont_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'UK_Shock_Cont_Short'] = 1
    return df

def add_sovereign_risk_proxy_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Sovereign_Risk_Long'] = 0
    df['Sovereign_Risk_Short'] = 0
    dates = _get_event_dates(events, ['gilt', 'bond', 'debt'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_active = df.index.strftime('%Y-%m-%d').isin(date_strs)
        df.loc[is_active & (df['Close'] > df['Open']), 'Sovereign_Risk_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Sovereign_Risk_Short'] = 1
    return df

def add_macro_shock_inside_bar_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Macro_Inside_Bar_Long'] = 0
    df['Macro_Inside_Bar_Short'] = 0
    
    dates = _get_event_dates(events, ['cpi', 'nfp', 'fomc', 'boe'])
    if dates:
        date_strs = [str(d) for d in dates]
        is_day = df.index.strftime('%Y-%m-%d').isin(date_strs)
        is_active = is_day & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        inside_bar = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df.loc[is_active & inside_bar, 'Macro_Inside_Bar_Long'] = 1
        df.loc[is_active & inside_bar, 'Macro_Inside_Bar_Short'] = 1
    return df