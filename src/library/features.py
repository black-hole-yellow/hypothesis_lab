import pandas as pd
import numpy as np
import os
from hmmlearn.hmm import GaussianHMM
from src.utils.decorators import provides

@provides('Body_ZScore')
def add_volatility_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    body = abs(df['Close'] - df['Open'])
    rolling_mean = body.rolling(window=lookback).mean()
    rolling_std = body.rolling(window=lookback).std()
    
    df['Body_ZScore'] = (body - rolling_mean) / rolling_std
    return df

@provides('Volume_ZScore')
def add_volume_zscore(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    mean_vol = df['Volume'].rolling(window=lookback).mean()
    std_vol = df['Volume'].rolling(window=lookback).std()
    
    df['Volume_ZScore'] = (df['Volume'] - mean_vol) / std_vol
    return df

@provides('Price_ZScore')
def add_price_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    rolling_mean = df['Close'].rolling(window=lookback).mean()
    rolling_std = df['Close'].rolling(window=lookback).std()
    df['Price_ZScore'] = (df['Close'] - rolling_mean) / rolling_std
    return df

@provides('Fractal_High', 'Fractal_Low')
def add_williams_fractals(df: pd.DataFrame, timeframe: str, n: int = 2) -> pd.DataFrame:
    if timeframe not in ['1h', '4h', '1D', '1W']:
        df['Fractal_High'] = False
        df['Fractal_Low'] = False
        return df

    df['Fractal_High'] = df['High'] == df['High'].rolling(window=2*n + 1, center=True).max()
    df['Fractal_Low'] = df['Low'] == df['Low'].rolling(window=2*n + 1, center=True).min()
    return df

@provides('Norm_Slope')
def add_normalized_slope(df: pd.DataFrame, lookback: int = 20, atr_lookback: int = 14) -> pd.DataFrame:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_lookback).mean()

    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def calc_slope(y):
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).sum()
        return covariance / x_var

    raw_slope = df['Close'].rolling(window=lookback).apply(calc_slope, raw=True)
    df['Norm_Slope'] = raw_slope / atr
    return df

@provides('Log_Return')
def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

@provides('ATR')
def add_atr(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=lookback).mean()
    return df

@provides('Entropy')
def add_shannon_entropy(df: pd.DataFrame, lookback: int = 50, bins: int = 10) -> pd.DataFrame:
    def calc_entropy(x):
        x = x[~np.isnan(x)]
        if len(x) < 2: return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
        
    df['Entropy'] = df['Log_Return'].rolling(window=lookback).apply(calc_entropy, raw=True)
    return df

@provides('Hurst')
def add_hurst_exponent(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    def calculate_hurst(ts):
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    df['Hurst'] = df['Close'].rolling(window=lookback).apply(calculate_hurst, raw=True)
    return df

@provides('Markov_Regime')
def add_markov_regime(df: pd.DataFrame) -> pd.DataFrame:
    if 'ATR' not in df.columns or 'Norm_Slope' not in df.columns:
        raise ValueError("ATR and Norm_Slope must be generated before Markov Regime.")
    
    atr_baseline = df['ATR'].rolling(window=50).mean()
    vol_state = np.where(df['ATR'] > atr_baseline, 'High Vol', 'Low Vol')
    trend_state = np.where(df['Norm_Slope'] > 0, 'Bullish', 'Bearish')
    
    df['Markov_Regime'] = pd.Series(vol_state, index=df.index) + " / " + pd.Series(trend_state, index=df.index)
    
    df.loc[df['ATR'].isna() | df['Norm_Slope'].isna(), 'Markov_Regime'] = np.nan
    df['Markov_Regime'] = df['Markov_Regime'].ffill()
    return df

@provides('Vol_Ratio')
def add_volatility_ratio(df: pd.DataFrame, short_lookback: int = 14, long_lookback: int = 100) -> pd.DataFrame:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    short_atr = tr.rolling(window=short_lookback).mean()
    long_atr = tr.rolling(window=long_lookback).mean()
    
    df['Vol_Ratio'] = short_atr / long_atr
    return df

@provides('High_Vol_Prob')
def add_hmm_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    if 'Log_Return' not in df.columns:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
    train_data = df[['Log_Return']].dropna()
    X = train_data.values
    
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)
    
    hidden_states_probs = model.predict_proba(X)
    
    var_state_0 = model.covars_[0][0]
    var_state_1 = model.covars_[1][0]
    
    high_vol_state = 0 if var_state_0 > var_state_1 else 1
    high_vol_probs = hidden_states_probs[:, high_vol_state]
    
    df['High_Vol_Prob'] = np.nan
    df.loc[train_data.index, 'High_Vol_Prob'] = high_vol_probs
    df['High_Vol_Prob'] = df['High_Vol_Prob'].ffill().fillna(0)
    return df

@provides('HTF_Trend_Up', 'HTF_Trend_Down')
def add_htf_trend(df: pd.DataFrame, htf: str = 'D', ema_period: int = 20) -> pd.DataFrame:
    htf_df = df['Close'].resample(htf).last().to_frame(name='HTF_Close')
    htf_df['HTF_EMA'] = htf_df['HTF_Close'].ewm(span=ema_period, adjust=False).mean()
    htf_df['HTF_Trend_Up'] = htf_df['HTF_Close'] > htf_df['HTF_EMA']
    htf_df['HTF_Trend_Down'] = htf_df['HTF_Close'] < htf_df['HTF_EMA']
    
    htf_df['HTF_Trend_Up'] = htf_df['HTF_Trend_Up'].shift(1)
    htf_df['HTF_Trend_Down'] = htf_df['HTF_Trend_Down'].shift(1)
    
    df = df.join(htf_df[['HTF_Trend_Up', 'HTF_Trend_Down']])
    df['HTF_Trend_Up'] = df['HTF_Trend_Up'].ffill().fillna(False)
    df['HTF_Trend_Down'] = df['HTF_Trend_Down'].ffill().fillna(False)
    return df

@provides('Confirmed_Fractal_High', 'Confirmed_Fractal_Low', 'Confirmed_Fractal_High_Price', 'Confirmed_Fractal_Low_Price')
def add_confirmed_fractals(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    df['Confirmed_Fractal_High'] = df['Fractal_High'].shift(n).fillna(False).astype(int)
    df['Confirmed_Fractal_Low'] = df['Fractal_Low'].shift(n).fillna(False).astype(int)
    df['Confirmed_Fractal_High_Price'] = df['High'].shift(n)
    df['Confirmed_Fractal_Low_Price'] = df['Low'].shift(n)
    return df

@provides('In_Liquidity_Void', 'Dist_to_POC_Pips')
def add_volume_profile_features(df: pd.DataFrame, session_start: str = "00:00", session_end: str = "08:00", bin_size_pips: int = 5) -> pd.DataFrame:
    pip_value = 0.0001
    bin_width = bin_size_pips * pip_value
    
    df['is_session'] = (df.index.strftime('%H:%M') >= session_start) & (df.index.strftime('%H:%M') <= session_end)
    df['date_group'] = df.index.date
    
    df['In_Liquidity_Void'] = 0
    df['Dist_to_POC_Pips'] = 0.0
    
    for date, group in df.groupby('date_group'):
        session_data = group[group['is_session']]
        if session_data.empty: continue
            
        price_min = session_data['Low'].min()
        price_max = session_data['High'].max()
        bins = np.arange(price_min, price_max + bin_width, bin_width)
        
        vol_profile, _ = np.histogram(session_data['Close'], bins=bins, weights=session_data['Volume'])
        if len(vol_profile) == 0: continue
            
        poc_index = np.argmax(vol_profile)
        poc_price = bins[poc_index]
        
        avg_vol = np.mean(vol_profile)
        lvn_threshold = avg_vol * 0.15
        lvn_bins = bins[np.where(vol_profile < lvn_threshold)[0]]
        
        day_indices = group.index
        df.loc[day_indices, 'Dist_to_POC_Pips'] = (df.loc[day_indices, 'Close'] - poc_price) / pip_value
        
        for lvn_p in lvn_bins:
            is_in_void = (df.loc[day_indices, 'Close'] >= lvn_p) & (df.loc[day_indices, 'Close'] < lvn_p + bin_width)
            df.loc[day_indices[is_in_void], 'In_Liquidity_Void'] = 1

    df.drop(columns=['is_session', 'date_group'], inplace=True)
    return df