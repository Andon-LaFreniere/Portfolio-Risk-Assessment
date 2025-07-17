import pandas as pd
import numpy as np

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    Args:
        series (pd.Series): Price series.
        window (int): Lookback window.
    Returns:
        pd.Series: RSI values.
    """
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        print(f"Error in RSI calculation: {e}")
        return pd.Series(index=series.index, dtype=float)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD (Moving Average Convergence Divergence).
    Args:
        series (pd.Series): Price series.
        fast (int): Fast EMA period.
        slow (int): Slow EMA period.
        signal (int): Signal line EMA period.
    Returns:
        pd.DataFrame: DataFrame with columns 'MACD' and 'Signal'.
    """
    try:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})
    except Exception as e:
        print(f"Error in MACD calculation: {e}")
        return pd.DataFrame(index=series.index)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    Args:
        series (pd.Series): Price series.
        window (int): Rolling window size.
        num_std (float): Number of standard deviations.
    Returns:
        pd.DataFrame: DataFrame with columns 'Middle', 'Upper', 'Lower'.
    """
    try:
        middle = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return pd.DataFrame({'Middle': middle, 'Upper': upper, 'Lower': lower})
    except Exception as e:
        print(f"Error in Bollinger Bands calculation: {e}")
        return pd.DataFrame(index=series.index) 