import pandas as pd
from typing import Optional

def rolling_correlation(df: pd.DataFrame, col1: str, col2: str, window: int = 21) -> pd.Series:
    """
    Calculate rolling correlation between two columns.
    Args:
        df (pd.DataFrame): DataFrame with columns col1 and col2.
        col1 (str): First column name.
        col2 (str): Second column name.
        window (int): Rolling window size.
    Returns:
        pd.Series: Rolling correlation.
    """
    try:
        return df[col1].rolling(window).corr(df[col2])
    except Exception as e:
        print(f"Error in rolling correlation: {e}")
        return pd.Series(index=df.index, dtype=float) 