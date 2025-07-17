from typing import Any, Optional
import numpy as np
from scipy.stats import genpareto
# from arch.univariate import ConstantMean, GARCH, Normal  # Uncomment for full implementation

class TailRiskModel:
    """
    Extreme Value Theory for VaR/CVaR calculation.
    """
    def __init__(self, threshold_quantile: float = 0.95):
        self.threshold_quantile = threshold_quantile
        self.gpd_params = None

    def fit(self, returns: np.ndarray):
        """Fit GPD to the tail of the returns distribution."""
        try:
            threshold = np.quantile(returns, self.threshold_quantile)
            excess = returns[returns > threshold] - threshold
            self.gpd_params = genpareto.fit(excess)
        except Exception as e:
            print(f"Error fitting GPD: {e}")
            raise

    def var(self, alpha: float = 0.99) -> float:
        """Calculate Value at Risk (VaR) at given alpha."""
        if self.gpd_params is None:
            raise ValueError("Model not fitted.")
        c, loc, scale = self.gpd_params
        return genpareto.ppf(alpha, c, loc=loc, scale=scale)

    def cvar(self, alpha: float = 0.99) -> float:
        """Calculate Conditional Value at Risk (CVaR) at given alpha."""
        if self.gpd_params is None:
            raise ValueError("Model not fitted.")
        c, loc, scale = self.gpd_params
        var = self.var(alpha)
        # CVaR for GPD: E[X | X > VaR]
        return (var + scale / (1 - c)) if c < 1 else np.nan 