import streamlit as st
import pandas as pd
import requests
import numpy as np

st.set_page_config(page_title="Portfolio Risk Dashboard", layout="wide")
st.title("Portfolio Risk Assessment Dashboard")

API_URL = "http://api:8000"

def get_data(key=None):
    st.write("### Upload CSV or Enter Data Manually")
    uploaded = st.file_uploader("Upload CSV with columns 'price' and 'signal'", type=["csv"], key=key)
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'price' in df.columns and 'signal' in df.columns:
            return df['price'].tolist(), df['signal'].tolist()
        else:
            st.error("CSV must have 'price' and 'signal' columns.")
            return None, None
    else:
        prices = st.text_area("Enter prices (comma-separated)", key=f"prices_{key}")
        signals = st.text_area("Enter signals (comma-separated, e.g. 1,0,-1)", key=f"signals_{key}")
        if prices and signals:
            try:
                price_list = [float(x) for x in prices.split(",")]
                signal_list = [int(x) for x in signals.split(",")]
                if len(price_list) != len(signal_list):
                    st.error("Prices and signals must have the same length.")
                    return None, None
                return price_list, signal_list
            except Exception as e:
                st.error(f"Error parsing input: {e}")
                return None, None
        return None, None

tabs = st.tabs(["Risk Assessment", "Backtesting"])

with tabs[0]:
    st.header("Risk Assessment")
    prices, signals = get_data(key="risk")
    if prices and signals:
        if st.button("Run Risk Assessment"):
            with st.spinner("Assessing risk..."):
                try:
                    resp = requests.post(f"{API_URL}/risk", json={"prices": prices, "signals": signals})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Risk Score: {data['risk_score']:.2f}")
                        st.write(f"Max Drawdown: {data['max_drawdown']:.2%}")
                        st.write(f"Sharpe Ratio: {data['sharpe_ratio']:.2f}")
                        st.line_chart(data['portfolio_value'], use_container_width=True)
                    else:
                        st.error(f"API error: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    else:
        st.info("Upload a CSV or enter data to assess risk.")

with tabs[1]:
    st.header("Backtesting")
    prices, signals = get_data(key="backtest")
    if prices and signals:
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    resp = requests.post(f"{API_URL}/backtest", json={"prices": prices, "signals": signals})
                    if resp.status_code == 200:
                        data = resp.json()
                        st.write(f"Max Drawdown: {data['max_drawdown']:.2%}")
                        st.write(f"Sharpe Ratio: {data['sharpe_ratio']:.2f}")
                        st.line_chart(data['portfolio_value'], use_container_width=True)
                    else:
                        st.error(f"API error: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")
    else:
        st.info("Upload a CSV or enter data to run a backtest.") 