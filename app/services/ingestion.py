"""
Bestand: app/services/ingestion.py
Relatief pad: ./app/services/ingestion.py
Functie: Haalt OHLCV-marktdata en nieuwsdata op met minimale validatie.
"""

from typing import Any

import pandas as pd
import requests
import yfinance as yf


def fetch_market_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 40:
        raise ValueError(f"Onvoldoende marktdata voor ticker {ticker}.")
    return df.reset_index()


def fetch_news_articles(news_query: str, news_api_key: str | None) -> list[dict[str, Any]]:
    if not news_api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": news_query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 25,
        "apiKey": news_api_key,
    }
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        return []
    return resp.json().get("articles", [])
