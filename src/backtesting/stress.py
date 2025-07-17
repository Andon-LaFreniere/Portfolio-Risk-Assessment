import pandas as pd
import numpy as np
from typing import Callable

def stress_test(portfolio: pd.DataFrame, returns_col: str = 'strategy_returns', shock: float = -0.05) -> pd.DataFrame:
    """
    Apply a shock to returns and recalculate portfolio value.
    Args:
        portfolio (pd.DataFrame): Backtest results with returns column.
        returns_col (str): Name of the returns column.
        shock (float): Shock to apply (e.g., -0.05 for -5%).
    Returns:
        pd.DataFrame: Stressed portfolio value.
    """
    try:
        stressed = portfolio.copy()
        stressed[returns_col] += shock
        stressed['stressed_value'] = (1 + stressed[returns_col]).cumprod() * stressed['portfolio_value'].iloc[0]
        return stressed
    except Exception as e:
        print(f"Error in stress test: {e}")
        return pd.DataFrame() 