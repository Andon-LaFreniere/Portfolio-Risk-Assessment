import pandas as pd
import numpy as np

def parkinson_volatility(df: pd.DataFrame, high_col: str = 'High', low_col: str = 'Low', window: int = 21) -> pd.Series:
    """
    Calculate the Parkinson realized volatility estimator.
    Args:
        df (pd.DataFrame): DataFrame with high and low price columns.
        high_col (str): Name of the high price column.
        low_col (str): Name of the low price column.
        window (int): Rolling window size (in days).
    Returns:
        pd.Series: Parkinson volatility estimate.
    """
    try:
        rs = (np.log(df[high_col] / df[low_col])) ** 2
        parkinson = rs.rolling(window).mean() * (1 / (4 * np.log(2)))
        return np.sqrt(parkinson)
    except Exception as e:
        print(f"Error in Parkinson volatility calculation: {e}")
        return pd.Series(index=df.index, dtype=float)

def garman_klass_volatility(df: pd.DataFrame, open_col: str = 'Open', high_col: str = 'High', low_col: str = 'Low', close_col: str = 'Close', window: int = 21) -> pd.Series:
    """
    Calculate the Garman-Klass realized volatility estimator.
    Args:
        df (pd.DataFrame): DataFrame with open, high, low, close price columns.
        open_col (str): Name of the open price column.
        high_col (str): Name of the high price column.
        low_col (str): Name of the low price column.
        close_col (str): Name of the close price column.
        window (int): Rolling window size (in days).
    Returns:
        pd.Series: Garman-Klass volatility estimate.
    """
    try:
        log_hl = np.log(df[high_col] / df[low_col])
        log_co = np.log(df[close_col] / df[open_col])
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        gk_rolling = gk.rolling(window).mean()
        return np.sqrt(gk_rolling)
    except Exception as e:
        print(f"Error in Garman-Klass volatility calculation: {e}")
        return pd.Series(index=df.index, dtype=float) 