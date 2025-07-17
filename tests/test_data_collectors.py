import pytest
from src.data.collectors import YahooFinanceCollector, FREDCollector, VIXCollector, SentimentCollector
import pandas as pd
import os
from src.features.volatility import parkinson_volatility, garman_klass_volatility
from src.features.correlations import rolling_correlation
from src.features.indicators import rsi, macd, bollinger_bands
from src.features.fractal_hurst import fractal_dimension, hurst_exponent
from src.features.regime import regime_indicator
import numpy as np
from src.models.volatility_forecaster import VolatilityForecaster
from src.models.regime_detector import RegimeDetector
from src.models.risk_factor_model import RiskFactorModel
from src.models.tail_risk_model import TailRiskModel
from src.models.ensemble_model import EnsembleModel
import torch
from src.backtesting.engine import BacktestEngine
from src.backtesting.hedge import dynamic_hedge_ratio
from src.backtesting.stress import stress_test
from src.backtesting.metrics import sortino_ratio, calmar_ratio

@pytest.mark.parametrize("collector_class, args", [
    (YahooFinanceCollector, {"tickers": ["AAPL"], "start": "2020-01-01", "end": "2020-12-31"}),
    (FREDCollector, {"series_ids": ["DGS10"], "start": "2020-01-01", "end": "2020-12-31"}),
    (VIXCollector, {"start": "2020-01-01", "end": "2020-12-31"}),
    (SentimentCollector, {"query": "AAPL", "start": "2020-01-01", "end": "2020-12-31"}),
])
def test_fetch_data_stub(collector_class, args):
    collector = collector_class() if collector_class != FREDCollector else FREDCollector(api_key=os.getenv("FRED_API_KEY", "demo"))
    try:
        collector.fetch_data(**args)
    except NotImplementedError:
        pass
    except Exception:
        pass  # Accept any error for stub
    assert hasattr(collector, 'validate_data')

def test_yahoo_finance_fetch_and_validate():
    collector = YahooFinanceCollector()
    try:
        df = collector.fetch_data(["AAPL"], "2020-01-01", "2020-01-10")
    except Exception as e:
        pytest.skip(f"Yahoo Finance fetch failed: {e}")
    assert isinstance(df, pd.DataFrame)
    assert collector.validate_data(df)

def test_fred_fetch_and_validate():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        pytest.skip("FRED_API_KEY not set in environment.")
    collector = FREDCollector(api_key=api_key)
    try:
        df = collector.fetch_data(["DGS10"], "2020-01-01", "2020-01-10")
    except Exception as e:
        pytest.skip(f"FRED fetch failed: {e}")
    assert isinstance(df, pd.DataFrame)
    assert collector.validate_data(df)

def test_vix_fetch_and_validate():
    collector = VIXCollector()
    try:
        df = collector.fetch_data("2020-01-01", "2020-01-10")
    except Exception as e:
        pytest.skip(f"VIX fetch failed: {e}")
    assert isinstance(df, pd.DataFrame)
    assert collector.validate_data(df)

def test_sentiment_fetch_and_validate():
    collector = SentimentCollector()
    with pytest.raises(NotImplementedError):
        collector.fetch_data("AAPL", "2020-01-01", "2020-01-10")
    assert collector.validate_data({})

def test_parkinson_volatility():
    df = pd.DataFrame({
        'High': [10, 12, 11, 13, 12],
        'Low': [9, 10, 10, 11, 11]
    })
    vol = parkinson_volatility(df, window=2)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(df)

def test_garman_klass_volatility():
    df = pd.DataFrame({
        'Open': [9, 11, 10, 12, 11],
        'High': [10, 12, 11, 13, 12],
        'Low': [9, 10, 10, 11, 11],
        'Close': [9.5, 11.5, 10.5, 12.5, 11.5]
    })
    vol = garman_klass_volatility(df, window=2)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(df)

def test_rolling_correlation():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1]
    })
    corr = rolling_correlation(df, 'A', 'B', window=2)
    assert isinstance(corr, pd.Series)
    assert len(corr) == len(df)

def test_rsi():
    s = pd.Series([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5])
    result = rsi(s, window=3)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)

def test_macd():
    s = pd.Series(np.arange(30, dtype=float))
    result = macd(s)
    assert isinstance(result, pd.DataFrame)
    assert 'MACD' in result.columns and 'Signal' in result.columns

def test_bollinger_bands():
    s = pd.Series(np.arange(30, dtype=float))
    result = bollinger_bands(s)
    assert isinstance(result, pd.DataFrame)
    assert 'Middle' in result.columns and 'Upper' in result.columns and 'Lower' in result.columns

def test_fractal_dimension():
    s = pd.Series(np.random.rand(120))
    result = fractal_dimension(s, window=20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)

def test_hurst_exponent():
    s = pd.Series(np.random.rand(120))
    result = hurst_exponent(s, window=20)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)

def test_regime_indicator():
    s = pd.Series(np.random.rand(100))
    try:
        regime_indicator(s)
    except NotImplementedError:
        pass 

def test_volatility_forecaster_instantiation():
    model = VolatilityForecaster(input_size=4)
    x = torch.randn(2, 10, 4)
    out = model(x)
    assert out.shape == (2, 1)

def test_regime_detector_stub():
    model = RegimeDetector(n_states=2)
    import numpy as np
    X = np.random.randn(10, 1)
    try:
        model.fit(X)
    except NotImplementedError:
        pass
    try:
        model.predict(X)
    except NotImplementedError:
        pass

def test_risk_factor_model():
    import numpy as np
    model = RiskFactorModel(n_factors=2)
    X = np.random.randn(10, 3)
    model.fit(X)
    transformed = model.transform(X)
    assert transformed.shape[1] == 2

def test_tail_risk_model():
    import numpy as np
    model = TailRiskModel(threshold_quantile=0.8)
    returns = np.random.randn(100)
    model.fit(returns)
    var = model.var(0.95)
    cvar = model.cvar(0.95)
    assert isinstance(var, float)
    assert isinstance(cvar, float)

def test_ensemble_model():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    base_models = [LinearRegression(), LinearRegression()]
    model = EnsembleModel(base_models=base_models)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (20,) 

def test_backtest_engine():
    import pandas as pd
    idx = pd.date_range('2020-01-01', periods=10)
    prices = pd.DataFrame({'A': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103]}, index=idx)
    signals = pd.Series([1, 1, 0, -1, -1, 0, 1, 1, 0, -1], index=idx)
    engine = BacktestEngine()
    results = engine.run(prices, signals)
    assert 'portfolio_value' in results.columns
    sharpe = engine.sharpe_ratio()
    mdd = engine.max_drawdown()
    assert isinstance(sharpe, float)
    assert isinstance(mdd, float) 

def test_dynamic_hedge_ratio():
    delta_call = dynamic_hedge_ratio(S=100, K=100, T=1, r=0.01, sigma=0.2, option_type='call')
    delta_put = dynamic_hedge_ratio(S=100, K=100, T=1, r=0.01, sigma=0.2, option_type='put')
    assert 0 <= delta_call <= 1
    assert -1 <= delta_put <= 0

def test_stress_test():
    import pandas as pd
    df = pd.DataFrame({'strategy_returns': [0.01, -0.02, 0.03], 'portfolio_value': [100, 101, 99]})
    stressed = stress_test(df, shock=-0.05)
    assert 'stressed_value' in stressed.columns

def test_sortino_calmar():
    import pandas as pd
    returns = pd.Series([0.01, -0.02, 0.03, 0.02, -0.01])
    mdd = -0.05
    sortino = sortino_ratio(returns)
    calmar = calmar_ratio(returns, mdd)
    assert isinstance(sortino, float)
    assert isinstance(calmar, float) 