import pandas as pd
import numpy as np
import os
from hmmlearn.hmm import GaussianHMM

def add_williams_fractals(df: pd.DataFrame, timeframe: str, n: int = 2) -> pd.DataFrame:
    """
    Calculates N-period Williams Fractals.
    Gated strictly to 1h and 4h timeframes.
    """
    if timeframe not in ['1h', '4h', '1D', '1W']:
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

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logarithmic Returns for mathematical modeling.
    """
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def add_atr(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """
    Standalone Average True Range (ATR) for risk calibration.
    """
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=lookback).mean()
    return df

def add_price_zscore(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Z-Score of the Close Price. 
    Useful for finding inflection points (e.g., Z > 2 or Z < -2 implies extreme mean deviation).
    """
    rolling_mean = df['Close'].rolling(window=lookback).mean()
    rolling_std = df['Close'].rolling(window=lookback).std()
    df['Price_ZScore'] = (df['Close'] - rolling_mean) / rolling_std
    return df

def add_shannon_entropy(df: pd.DataFrame, lookback: int = 50, bins: int = 10) -> pd.DataFrame:
    """
    Measures the 'Chaos' of the market using Shannon Entropy of Log Returns.
    Higher value = More chaos/noise. Lower value = Organized trend/structure.
    """
    def calc_entropy(x):
        x = x[~np.isnan(x)]
        if len(x) < 2: return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0] # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
        
    df['Entropy'] = df['Log_Return'].rolling(window=lookback).apply(calc_entropy, raw=True)
    return df

def add_hurst_exponent(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Rolling Hurst Exponent estimation.
    H < 0.5: Mean Reverting | H ~ 0.5: Random Walk | H > 0.5: Trending
    Note: This is computationally heavy, adjusting 'lags' keeps it performant.
    """
    def calculate_hurst(ts):
        lags = range(2, 20)
        # Calculate the variance of the differences across different lags
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        # Linear fit on log-log scale
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    df['Hurst'] = df['Close'].rolling(window=lookback).apply(calculate_hurst, raw=True)
    return df

def add_markov_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies the current market into one of 4 discrete regimes.
    Requires 'ATR' and 'Norm_Slope' to be calculated first.
    """
    if 'ATR' not in df.columns or 'Norm_Slope' not in df.columns:
        raise ValueError("ATR and Norm_Slope must be generated before Markov Regime.")
    
    # 1. Volatility State (Compared to its own historical baseline)
    atr_baseline = df['ATR'].rolling(window=50).mean()
    vol_state = np.where(df['ATR'] > atr_baseline, 'High Vol', 'Low Vol')
    
    # 2. Directional State (Based on Linear Regression Slope)
    trend_state = np.where(df['Norm_Slope'] > 0, 'Bullish', 'Bearish')
    
    # 3. Combine into discrete Regime States (Fixing the index alignment!)
    df['Markov_Regime'] = pd.Series(vol_state, index=df.index) + " / " + pd.Series(trend_state, index=df.index)
    
    # Clean up early NaNs
    df.loc[df['ATR'].isna() | df['Norm_Slope'].isna(), 'Markov_Regime'] = np.nan
    df['Markov_Regime'] = df['Markov_Regime'].ffill()
    
    return df

def add_volatility_ratio(df: pd.DataFrame, short_lookback: int = 14, long_lookback: int = 100) -> pd.DataFrame:
    """
    Calculates the Volatility Ratio (Short-term Volatility / Long-term Volatility).
    Acts as a proxy for GARCH-style volatility clustering.
    > 1.0 : Volatility is expanding (potential breakout or shock).
    < 1.0 : Volatility is contracting (consolidation phase).
    """
    # Calculate raw True Range
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Short and Long ATR
    short_atr = tr.rolling(window=short_lookback).mean()
    long_atr = tr.rolling(window=long_lookback).mean()
    
    # Create the ratio
    df['Vol_Ratio'] = short_atr / long_atr
    
    return df

def add_hmm_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses a true Machine Learning Hidden Markov Model (2 states) to calculate 
    the probability of being in a High Volatility regime.
    """
    if 'Log_Return' not in df.columns:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
    # We need clean data to train the model
    train_data = df[['Log_Return']].dropna()
    
    # Reshape data for hmmlearn (requires 2D array)
    X = train_data.values
    
    # Fit the 2-state Gaussian HMM
    # n_iter=100 ensures the model converges to the best fit
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)
    
    # Predict the probability of being in each state
    hidden_states_probs = model.predict_proba(X)
    
    # The model doesn't know which state is "High Vol" vs "Low Vol", it just knows they are different.
    # We identify the High Vol state by finding which state has the higher variance in its covariance matrix.
    var_state_0 = model.covars_[0][0]
    var_state_1 = model.covars_[1][0]
    
    high_vol_state = 0 if var_state_0 > var_state_1 else 1
    
    # Extract just the probability of the High Volatility state
    high_vol_probs = hidden_states_probs[:, high_vol_state]
    
    # Safely map the probabilities back to the main DataFrame matching the exact dates
    df['High_Vol_Prob'] = np.nan
    df.loc[train_data.index, 'High_Vol_Prob'] = high_vol_probs
    
    # Forward fill any tiny gaps and fill initial NaNs with 0
    df['High_Vol_Prob'] = df['High_Vol_Prob'].ffill().fillna(0)
    
    return df