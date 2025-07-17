import numpy as np
import pandas as pd
from typing import Optional

def fractal_dimension(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Estimate the fractal dimension using the box-counting method over a rolling window.
    Args:
        series (pd.Series): Price series.
        window (int): Rolling window size.
    Returns:
        pd.Series: Fractal dimension estimates.
    """
    try:
        def box_count(S):
            S = (S - S.min()) / (S.max() - S.min() + 1e-9)
            counts = []
            for k in range(2, 10):
                boxes = np.floor(S * k)
                counts.append(len(np.unique(boxes)))
            coeffs = np.polyfit(np.log(range(2, 10)), np.log(counts), 1)
            return -coeffs[0]
        return series.rolling(window).apply(box_count, raw=False)
    except Exception as e:
        print(f"Error in fractal dimension calculation: {e}")
        return pd.Series(index=series.index, dtype=float)

def hurst_exponent(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Estimate the Hurst exponent over a rolling window.
    Args:
        series (pd.Series): Price series.
        window (int): Rolling window size.
    Returns:
        pd.Series: Hurst exponent estimates.
    """
    try:
        def hurst(S):
            N = len(S)
            if N < 20:
                return np.nan
            Y = np.cumsum(S - np.mean(S))
            R = np.max(Y) - np.min(Y)
            S_ = np.std(S)
            if S_ == 0:
                return np.nan
            return np.log(R / S_) / np.log(N) if R > 0 and S_ > 0 else np.nan
        return series.rolling(window).apply(hurst, raw=True)
    except Exception as e:
        print(f"Error in Hurst exponent calculation: {e}")
        return pd.Series(index=series.index, dtype=float) 