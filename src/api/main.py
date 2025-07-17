from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd
from src.backtesting.engine import BacktestEngine

app = FastAPI(title="Portfolio Risk Assessment API")

class RiskRequest(BaseModel):
    prices: list[float]
    signals: list[int]


def _to_df_and_series(prices: list[float], signals: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.RangeIndex(len(prices))
    prices_df = pd.DataFrame({'A': prices}, index=idx)
    signals_ser = pd.Series(signals, index=idx)
    return prices_df, signals_ser

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/risk")
def risk_assessment(request: RiskRequest) -> Dict[str, Any]:
    """Compute risk score using max drawdown from backtest."""
    try:
        prices, signals = _to_df_and_series(request.prices, request.signals)
        engine = BacktestEngine()
        results = engine.run(prices, signals)
        mdd = engine.max_drawdown()
        sharpe = engine.sharpe_ratio()
        # Simple risk score: higher drawdown = higher risk
        risk_score = min(1.0, max(0.0, -mdd * 10))
        return {
            "risk_score": risk_score,
            "max_drawdown": float(mdd),
            "sharpe_ratio": float(sharpe),
            "portfolio_value": results['portfolio_value'].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
def run_backtest(request: RiskRequest) -> Dict[str, Any]:
    """Run backtest and return key metrics and portfolio value series."""
    try:
        prices, signals = _to_df_and_series(request.prices, request.signals)
        engine = BacktestEngine()
        results = engine.run(prices, signals)
        mdd = engine.max_drawdown()
        sharpe = engine.sharpe_ratio()
        return {
            "max_drawdown": float(mdd),
            "sharpe_ratio": float(sharpe),
            "portfolio_value": results['portfolio_value'].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 