import pandas as pd
import numpy as np
import os

def add_williams_fractals(df: pd.DataFrame, timeframe: str, n: int = 2) -> pd.DataFrame:
    """
    Calculates N-period Williams Fractals.
    Gated strictly to 1h and 4h timeframes.
    """
    if timeframe not in ['1h', '4h']:
        df['Fractal_High'] = False
        df['Fractal_Low'] = False
        return df

    # center=True aligns the peak with the current candle
    df['Fractal_High'] = df['High'] == df['High'].rolling(window=2*n + 1, center=True).max()
    df['Fractal_Low'] = df['Low'] == df['Low'].rolling(window=2*n + 1, center=True).min()
    
    return df

def add_volatility_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Calculates the Z-Score of the candle body height.
    """
    body = abs(df['Close'] - df['Open'])
    rolling_mean = body.rolling(window=lookback).mean()
    rolling_std = body.rolling(window=lookback).std()
    
    df['Body_ZScore'] = (body - rolling_mean) / rolling_std
    return df

def add_normalized_slope(df: pd.DataFrame, lookback: int = 20, atr_lookback: int = 14) -> pd.DataFrame:
    """
    Calculates the Linear Regression Slope of the Close price normalized by ATR.
    """
    # 1. Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_lookback).mean()

    # 2. Vectorized Linear Regression Slope
    x = np.arange(lookback)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    def calc_slope(y):
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).sum()
        return covariance / x_var

    raw_slope = df['Close'].rolling(window=lookback).apply(calc_slope, raw=True)
    
    # 3. Normalize Slope
    df['Norm_Slope'] = raw_slope / atr
    return df