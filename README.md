# Historical Portfolio Risk Assessment Tool

## What is this project?
This is a production-ready, end-to-end platform for advanced portfolio risk assessment and dynamic hedging recommendations. It combines modern machine learning, quantitative finance, and robust software engineering to help investors, risk managers, and researchers:
- **Quantify and monitor portfolio risk** in real time
- **Forecast volatility and market regimes** using state-of-the-art models
- **Backtest and stress-test strategies** with realistic constraints
- **Receive dynamic hedge recommendations** based on market conditions
- **Visualize and interpret risk metrics** through an interactive dashboard

**Who is it for?**
- Quantitative analysts and researchers
- Portfolio managers and risk officers
- Fintech developers and students
- Anyone interested in advanced financial risk analytics

---

## What risk is being assessed?
This system assesses **portfolio risk**—the potential for financial loss or underperformance in a portfolio of assets (such as stocks, ETFs, or other securities). Specifically, it quantifies and analyzes:

- **Market Risk**: Losses due to movements in market prices (e.g., stock prices, interest rates, volatility)
- **Volatility Risk**: The magnitude and unpredictability of price changes
- **Tail Risk**: The risk of rare but severe losses (extreme events), measured using Value at Risk (VaR) and Conditional VaR (CVaR)
- **Drawdown Risk**: The risk of a significant decline from a portfolio’s peak value (maximum drawdown)
- **Regime Risk**: The risk of being exposed to adverse market regimes (e.g., bear markets, high-volatility periods)
- **Systematic Risk**: The risk from factors that affect the entire market, not just individual assets

The system uses advanced models and metrics (e.g., Sharpe ratio, Sortino ratio, Calmar ratio, volatility estimators, regime detection) to provide a comprehensive view of your portfolio’s risk profile.

---

## What is backtesting?
**Backtesting** is the process of simulating a trading or investment strategy using historical data to evaluate how it would have performed in the past. It helps you answer questions like:

- Would my strategy have made money over the last 5 years?
- How risky was it? What was the worst drawdown?
- How did it perform in different market conditions (bull, bear, sideways)?
- How sensitive is it to transaction costs, slippage, or stress scenarios?

**How it works in this system:**
- You provide historical price data and your trading signals (when to buy, sell, or hold)
- The backtesting engine simulates trades, applies transaction costs, and tracks portfolio value over time
- It calculates key performance metrics (returns, Sharpe ratio, drawdown, etc.) and visualizes the results
- You can stress-test your strategy by simulating shocks or adverse scenarios

**Why is backtesting important?**
- It helps you validate and refine your strategies before risking real money
- It reveals hidden risks and weaknesses in your approach
- It provides evidence and confidence for deploying strategies in live markets

---

## What are the user inputs?

### For Risk Assessment (API or Dashboard)
- **prices**: A list of historical prices for your asset or portfolio (e.g., daily closing prices)
- **signals**: A list of trading signals for each date (e.g., 1 for long/buy, -1 for short/sell, 0 for hold/cash)
  - The length of `prices` and `signals` must be the same
- **How to input**: Upload a CSV file with columns `price` and `signal`, or enter comma-separated values manually in the dashboard

### For Backtesting (API or Dashboard)
- **prices**: Same as above
- **signals**: Same as above
- **How to input**: Same as above

### For Advanced Use (Notebooks, Custom Scripts)
- You can use the data pipeline to fetch and clean data from Yahoo Finance, FRED, VIX, and sentiment APIs
- Feature engineering modules let you compute volatility, technical indicators, and more
- ML models can be trained and evaluated on your own data
- The backtesting engine and risk metrics can be used programmatically for research or integration

---

## How should a user use it?
1. **Data Ingestion**: Collect historical and real-time data from multiple sources (Yahoo Finance, FRED, VIX, sentiment APIs) using the built-in data pipeline.
2. **Feature Engineering**: Automatically compute advanced features (volatility estimators, technical indicators, regime indicators) for your assets.
3. **Modeling**: Use or extend the provided ML models (LSTM/GRU, GARCH, HMM, PCA, EVT, ensemble) to forecast volatility, detect regimes, and decompose risk.
4. **Backtesting & Stress Testing**: Simulate your strategies with realistic transaction costs, slippage, and stress scenarios. Evaluate performance with Sharpe, Sortino, Calmar ratios, and more.
5. **Risk Assessment & Hedging**: Use the API or dashboard to assess portfolio risk, receive real-time alerts, and get dynamic hedge recommendations based on Black-Scholes Greeks and model outputs.
6. **Visualization & Reporting**: Interact with the Streamlit dashboard to upload your portfolio, run analyses, and visualize results. Export reports and metrics for further analysis.

## Typical Workflow
1. **Start the API and dashboard** (locally or with Docker)
2. **Upload your portfolio data** (CSV or manual entry) in the dashboard
3. **Run risk assessment** to get risk scores, drawdown, and volatility forecasts
4. **Run backtests** to evaluate your strategy’s historical performance
5. **Review hedge recommendations** and stress test your portfolio
6. **Download results or integrate with your own tools via the API**

## Example Use Cases
- **Institutional risk monitoring**: Integrate with your data feeds and use the API for daily risk reporting
- **Strategy research**: Backtest new hedging or allocation strategies with advanced ML and risk metrics
- **Education**: Use the notebooks and dashboard to learn about modern risk management and quantitative finance
- **Fintech apps**: Build on top of the API for custom dashboards, alerts, or trading tools

---

## Overview
A machine learning system for advanced portfolio risk assessment and dynamic hedging recommendations, leveraging state-of-the-art quantitative finance and ML techniques.

## Features
- Multi-source data ingestion (Yahoo Finance, FRED, VIX, sentiment APIs)
- Advanced feature engineering (volatility estimators, technical indicators, regime detection)
- ML models: LSTM/GRU, GARCH, HMM, PCA, EVT, ensemble learning
- Backtesting with realistic constraints and performance metrics
- FastAPI endpoints and Streamlit dashboard
- Dockerized deployment

## Project Structure
```
./
├── src/
│   ├── data/
│   ├── models/
│   ├── features/
│   ├── backtesting/
│   └── api/
├── notebooks/
├── tests/
├── config/
├── requirements.txt
├── README.md
└── docker/
```

## Quickstart
1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run API**
   ```sh
   python -m uvicorn src.api.main:app --reload
   # Visit http://localhost:8000/docs
   ```
3. **Run Dashboard**
   ```sh
   python -m streamlit run src/api/dashboard.py
   # Visit http://localhost:8501
   ```
4. **Run tests**
   ```sh
   pytest
   ```

## Docker Usage
See [docker/README-docker.md](docker/README-docker.md) for full instructions.

- Build and run both API and dashboard:
  ```sh
  docker-compose -f docker/docker-compose.yml up --build
  ```
- API: [http://localhost:8000](http://localhost:8000)
- Dashboard: [http://localhost:8501](http://localhost:8501)

## API Usage Example
- **Risk Assessment**
  ```sh
  curl -X POST "http://localhost:8000/risk" -H "Content-Type: application/json" -d '{"prices": [100,101,102], "signals": [1,1,0]}'
  ```
- **Backtesting**
  ```sh
  curl -X POST "http://localhost:8000/backtest" -H "Content-Type: application/json" -d '{"prices": [100,101,102], "signals": [1,1,0]}'
  ```
- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Dashboard
- Upload CSV or enter data manually for risk assessment and backtesting
- Visualize portfolio metrics and risk scores

## Testing
- Run all tests:
  ```sh
  pytest
  ```

## License
MIT 
