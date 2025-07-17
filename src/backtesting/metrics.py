import numpy as np
import pandas as pd

def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio.
    Args:
        returns (pd.Series): Strategy returns.
        risk_free_rate (float): Risk-free rate (annualized).
    Returns:
        float: Sortino ratio.
    """
    try:
        downside = returns[returns < 0]
        downside_std = downside.std() + 1e-9
        excess = returns.mean() - risk_free_rate / 252
        return np.sqrt(252) * excess / downside_std
    except Exception as e:
        print(f"Error in Sortino ratio: {e}")
        return np.nan

def calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """
    Calculate the Calmar ratio.
    Args:
        returns (pd.Series): Strategy returns.
        max_drawdown (float): Maximum drawdown (negative value).
    Returns:
        float: Calmar ratio.
    """
    try:
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown + 1e-9)
    except Exception as e:
        print(f"Error in Calmar ratio: {e}")
        return np.nan 