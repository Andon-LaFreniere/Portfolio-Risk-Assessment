import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def sample_data():
    return {
        "prices": [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        "signals": [1, 1, 0, -1, -1, 0, 1, 1, 0, -1]
    }

def test_risk_assessment_api(sample_data):
    resp = client.post("/risk", json=sample_data)
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data
    assert "max_drawdown" in data
    assert "sharpe_ratio" in data
    assert "portfolio_value" in data
    assert isinstance(data["risk_score"], float)
    assert isinstance(data["max_drawdown"], float)
    assert isinstance(data["sharpe_ratio"], float)
    assert isinstance(data["portfolio_value"], list)
    assert len(data["portfolio_value"]) == len(sample_data["prices"])

def test_backtest_api(sample_data):
    resp = client.post("/backtest", json=sample_data)
    assert resp.status_code == 200
    data = resp.json()
    assert "max_drawdown" in data
    assert "sharpe_ratio" in data
    assert "portfolio_value" in data
    assert isinstance(data["max_drawdown"], float)
    assert isinstance(data["sharpe_ratio"], float)
    assert isinstance(data["portfolio_value"], list)
    assert len(data["portfolio_value"]) == len(sample_data["prices"]) 