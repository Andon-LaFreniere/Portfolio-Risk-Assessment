import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any

class BacktestEngine:
    """
    Simple backtesting engine for portfolio strategies.
    """
    def __init__(self, initial_cash: float = 1_000_000, transaction_cost: float = 0.001):
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.results = None

    def run(self, prices: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Run a backtest given price data and trading signals.
        Args:
            prices (pd.DataFrame): Price data (index: date, columns: assets).
            signals (pd.Series): Position signals (index: date, values: -1, 0, 1).
        Returns:
            pd.DataFrame: Backtest results (portfolio value, returns, drawdown).
        """
        try:
            portfolio = pd.DataFrame(index=prices.index)
            portfolio['signal'] = signals
            portfolio['price'] = prices.iloc[:, 0]  # Assume single asset for simplicity
            portfolio['returns'] = portfolio['price'].pct_change().fillna(0)
            portfolio['strategy_returns'] = portfolio['returns'] * portfolio['signal'].shift(1).fillna(0)
            portfolio['strategy_returns'] -= self.transaction_cost * (portfolio['signal'].diff().abs().fillna(0))
            portfolio['portfolio_value'] = self.initial_cash * (1 + portfolio['strategy_returns']).cumprod()
            portfolio['drawdown'] = (portfolio['portfolio_value'] / portfolio['portfolio_value'].cummax()) - 1
            self.results = portfolio
            return portfolio
        except Exception as e:
            print(f"Error in backtest: {e}")
            return pd.DataFrame()

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if self.results is None:
            raise ValueError("Run the backtest first.")
        excess = self.results['strategy_returns'] - risk_free_rate / 252
        return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)

    def max_drawdown(self) -> float:
        if self.results is None:
            raise ValueError("Run the backtest first.")
        return self.results['drawdown'].min() 