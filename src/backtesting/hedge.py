import numpy as np
from scipy.stats import norm

def dynamic_hedge_ratio(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Calculate Black-Scholes delta (hedge ratio) for a European option.
    Args:
        S (float): Spot price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
    Returns:
        float: Hedge ratio (delta).
    """
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-9)
        if option_type == 'call':
            return norm.cdf(d1)
        elif option_type == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    except Exception as e:
        print(f"Error in dynamic hedge ratio: {e}")
        return np.nan 