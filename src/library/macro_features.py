import pandas as pd
import numpy as np

# ==========================================
# ХЕЛПЕР: УМНЫЙ ПОИСК ДАТ (Игнорируем минуты)
# ==========================================
def _get_event_dates(events, keywords: list) -> list:
    """Универсальный парсер для словаря категорий macro_events.json."""
    if not events: return []
    all_events = []
    
    if isinstance(events, dict):
        for category, event_list in events.items():
            all_events.extend(event_list)
    else:
        all_events = events

    dates = set()
    for e in all_events:
        name = str(e.get('name', e.get('Event', ''))).lower()
        
        if any(k.lower() in name for k in keywords):
            dt_raw = e.get('start_date', e.get('Date', ''))
            if dt_raw:
                date_part = str(dt_raw).split('T')[0].split(' ')[0]
                try:
                    dates.add(pd.to_datetime(date_part).date())
                except:
                    pass
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

    is_nfp_day = np.isin(df.index.date, nfp_dates)
    is_active_window = (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 18)
    
    bullish_candle = df['Close'] > df['Open']
    bearish_candle = df['Close'] < df['Open']

    valid_mask = is_nfp_day & is_active_window
    
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

    is_nfp_day = np.isin(df.index.date, nfp_dates)
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

    is_cpi_day = np.isin(df.index.date, cpi_dates)
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
        is_day = np.isin(df.index.date, cpi_dates)
        is_active = (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        df.loc[is_day & is_active & (df['Close'] > df['Open']), 'Macro_CPI_Div_Long'] = 1
        df.loc[is_day & is_active & (df['Close'] < df['Open']), 'Macro_CPI_Div_Short'] = 1
    return df

def add_cpi_match_mean_reversion_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['CPI_Match_Reversion'] = 0
    cpi_dates = _get_event_dates(events, ['cpi'])
    if cpi_dates:
        is_day = np.isin(df.index.date, cpi_dates)
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
        is_fomc_day = np.isin(df.index.date, fomc_dates)
        is_active = (df['UA_Hour'] >= 21) & (df['UA_Hour'] <= 23)
        df.loc[is_fomc_day & is_active & (df['Close'] < df['Open']), 'FOMC_Sell_News_Long'] = 1
        df.loc[is_fomc_day & is_active & (df['Close'] > df['Open']), 'FOMC_Sell_News_Short'] = 1
    return df

def add_boe_hawkish_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['BoE_Hawkish'] = 0
    boe_dates = _get_event_dates(events, ['boe', 'bank of england'])
    if boe_dates:
        df.loc[np.isin(df.index.date, boe_dates) & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 16), 'BoE_Hawkish'] = 1
    return df

def add_cb_divergence_state_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['CB_Divergence_Long'] = 0
    df['CB_Divergence_Short'] = 0
    dates = _get_event_dates(events, ['rate', 'policy'])
    if dates:
        is_active = np.isin(df.index.date, dates) & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        df.loc[is_active & (df['Close'] > df['Open']), 'CB_Divergence_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'CB_Divergence_Short'] = 1
    return df

def add_boe_tone_shift_proxy_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['BoE_Tone_Shift_Long'] = 0
    df['BoE_Tone_Shift_Short'] = 0
    dates = _get_event_dates(events, ['boe', 'bailey'])
    if dates:
        is_active = np.isin(df.index.date, dates) & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 17)
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
        is_active = np.isin(df.index.date, dates) & (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 18)
        df.loc[is_active & (df['Close'] > df['Open']), 'Retail_Div_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Retail_Div_Short'] = 1
    return df

def add_unemp_fakeout_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Unemp_Fakeout_Long'] = 0
    df['Unemp_Fakeout_Short'] = 0
    dates = _get_event_dates(events, ['unemployment', 'jobless'])
    if dates:
        is_active = np.isin(df.index.date, dates) & (df['UA_Hour'] >= 15) & (df['UA_Hour'] <= 17)
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
        is_active = np.isin(df.index.date, dates)
        df.loc[is_active & (df['Close'] > df['Open']), 'Election_Vol_Crush_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Election_Vol_Crush_Short'] = 1
    return df

def add_uk_political_shock_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['UK_Shock_Cont_Long'] = 0
    df['UK_Shock_Cont_Short'] = 0
    dates = _get_event_dates(events, ['parliament', 'pm', 'minister'])
    if dates:
        is_active = np.isin(df.index.date, dates)
        df.loc[is_active & (df['Close'] > df['Open']), 'UK_Shock_Cont_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'UK_Shock_Cont_Short'] = 1
    return df

def add_sovereign_risk_proxy_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Sovereign_Risk_Long'] = 0
    df['Sovereign_Risk_Short'] = 0
    dates = _get_event_dates(events, ['gilt', 'bond', 'debt'])
    if dates:
        is_active = np.isin(df.index.date, dates)
        df.loc[is_active & (df['Close'] > df['Open']), 'Sovereign_Risk_Long'] = 1
        df.loc[is_active & (df['Close'] < df['Open']), 'Sovereign_Risk_Short'] = 1
    return df

def add_macro_shock_inside_bar_context(df: pd.DataFrame, events: list = None) -> pd.DataFrame:
    df = df.copy()
    df['Macro_Inside_Bar_Long'] = 0
    df['Macro_Inside_Bar_Short'] = 0
    
    dates = _get_event_dates(events, ['cpi', 'nfp', 'fomc', 'boe'])
    if dates:
        is_active = np.isin(df.index.date, dates) & (df['UA_Hour'] >= 14) & (df['UA_Hour'] <= 18)
        inside_bar = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        df.loc[is_active & inside_bar, 'Macro_Inside_Bar_Long'] = 1
        df.loc[is_active & inside_bar, 'Macro_Inside_Bar_Short'] = 1
    return df