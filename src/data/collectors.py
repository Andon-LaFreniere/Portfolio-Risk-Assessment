from typing import Any, Dict
import yfinance as yf
import pandas as pd
from fredapi import Fred
import os
import datetime

class YahooFinanceCollector:
    """Collects historical price and volume data from Yahoo Finance."""
    def fetch_data(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch data for given tickers and date range from Yahoo Finance.
        Args:
            tickers (list[str]): List of ticker symbols.
            start (str): Start date (YYYY-MM-DD).
            end (str): End date (YYYY-MM-DD).
        Returns:
            pd.DataFrame: Multi-index DataFrame with ticker and date.
        """
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: stack to long format
                data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index()
            else:
                # Single ticker: add Ticker column
                data = data.reset_index()
                data['Ticker'] = tickers[0] if tickers else None
            return data
        except Exception as e:
            print(f"Error fetching Yahoo Finance data: {e}")
            raise

    def validate_data(self, data: Any) -> bool:
        """
        Validate the integrity and completeness of the Yahoo Finance data.
        Args:
            data (Any): Data to validate (should be pd.DataFrame).
        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(data, pd.DataFrame):
            print("Data is not a DataFrame.")
            return False
        required_cols = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(set(data.columns)):
            print(f"Missing required columns: {required_cols - set(data.columns)}")
            return False
        if data.isnull().sum().sum() > 0:
            print("Data contains missing values.")
            return False
        return True

class FREDCollector:
    """Collects economic data from FRED."""
    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key (str, optional): FRED API key. If None, will use FRED_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED API key must be provided or set as FRED_API_KEY environment variable.")
        self.fred = Fred(api_key=self.api_key)

    def fetch_data(self, series_ids: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch data for given FRED series IDs and date range.
        Args:
            series_ids (list[str]): List of FRED series IDs.
            start (str): Start date (YYYY-MM-DD).
            end (str): End date (YYYY-MM-DD).
        Returns:
            pd.DataFrame: DataFrame with columns for each series.
        """
        try:
            data = {}
            for series_id in series_ids:
                s = self.fred.get_series(series_id, observation_start=start, observation_end=end)
                data[series_id] = s
            df = pd.DataFrame(data)
            df.index.name = "Date"
            df = df.reset_index()
            return df
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            raise

    def validate_data(self, data: Any) -> bool:
        """
        Validate the integrity and completeness of the FRED data.
        Args:
            data (Any): Data to validate (should be pd.DataFrame).
        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(data, pd.DataFrame):
            print("FRED data is not a DataFrame.")
            return False
        if data.shape[1] < 2:  # At least Date + 1 series
            print("FRED data missing expected columns.")
            return False
        # Allow some missing values, but not all missing for any series
        for col in data.columns:
            if col == "Date":
                continue
            if data[col].isnull().all():
                print(f"All values missing for series: {col}")
                return False
        return True

class VIXCollector:
    """Collects VIX and volatility term structure data."""
    def fetch_data(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch VIX index data from Yahoo Finance for the given date range.
        Args:
            start (str): Start date (YYYY-MM-DD).
            end (str): End date (YYYY-MM-DD).
        Returns:
            pd.DataFrame: DataFrame with VIX data.
        """
        try:
            data = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                data.columns = [col[0] for col in data.columns]
            data = data.reset_index()
            data['Symbol'] = 'VIX'
            return data
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            raise

    def validate_data(self, data: Any) -> bool:
        """
        Validate the integrity and completeness of the VIX data.
        Args:
            data (Any): Data to validate (should be pd.DataFrame).
        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(data, pd.DataFrame):
            print("VIX data is not a DataFrame.")
            return False
        if data.empty:
            print("VIX data is empty.")
            return False
        required_cols = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
        if data['Close'].isnull().all():
            print("All VIX close values are missing.")
            return False
        return True

class SentimentCollector:
    """Collects sentiment data from Twitter and Reddit APIs."""
    def __init__(self, twitter_bearer_token: str = None, reddit_client_id: str = None, reddit_client_secret: str = None, reddit_user_agent: str = None):
        """
        Args:
            twitter_bearer_token (str, optional): Twitter API Bearer Token.
            reddit_client_id (str, optional): Reddit API client ID.
            reddit_client_secret (str, optional): Reddit API client secret.
            reddit_user_agent (str, optional): Reddit API user agent.
        """
        self.twitter_bearer_token = twitter_bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        self.reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = reddit_user_agent or os.getenv("REDDIT_USER_AGENT")
        # TODO: Initialize API clients if credentials are provided

    def fetch_data(self, query: str, start: str, end: str) -> Any:
        """
        Fetch sentiment data from Twitter and Reddit for a given query and date range.
        Args:
            query (str): Search query (e.g., ticker or keyword).
            start (str): Start date (YYYY-MM-DD).
            end (str): End date (YYYY-MM-DD).
        Returns:
            Any: Placeholder for sentiment data (to be implemented).
        """
        try:
            # TODO: Implement Twitter API fetch (e.g., using tweepy or requests)
            # TODO: Implement Reddit API fetch (e.g., using praw)
            # TODO: Aggregate and return sentiment scores
            raise NotImplementedError("Sentiment data collection not yet implemented.")
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            raise

    def validate_data(self, data: Any) -> bool:
        """
        Validate the integrity and completeness of the sentiment data.
        Args:
            data (Any): Data to validate.
        Returns:
            bool: True if valid, False otherwise.
        """
        # TODO: Implement validation logic for sentiment data
        return True 