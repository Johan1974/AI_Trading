import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

load_dotenv()

app = FastAPI(title="AI Trading Bot", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

STATE: dict[str, Any] = {
    "last_prediction": None,
    "events": [],
    "started_at": datetime.utcnow().isoformat(),
}


class PredictionResponse(BaseModel):
    ticker: str
    predicted_next_close: float
    latest_close: float
    expected_return_pct: float
    signal: str
    news_sentiment: float
    generated_at: str


def _fetch_market_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", auto_adjust=True, progress=False)
    if df.empty or len(df) < 40:
        raise ValueError(f"Onvoldoende marktdata voor ticker {ticker}.")
    df = df.reset_index()
    return df


def _simple_news_sentiment_score(news_query: str, news_api_key: str | None) -> float:
    if not news_api_key:
        return 0.0

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
        return 0.0

    data = resp.json()
    articles = data.get("articles", [])
    if not articles:
        return 0.0

    positive_words = {
        "surge",
        "growth",
        "profit",
        "strong",
        "bull",
        "rally",
        "beat",
        "optimism",
        "recovery",
        "upside",
    }
    negative_words = {
        "crash",
        "loss",
        "weak",
        "bear",
        "fear",
        "recession",
        "inflation",
        "war",
        "downturn",
        "risk",
    }

    score = 0
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        pos_hits = sum(1 for w in positive_words if w in text)
        neg_hits = sum(1 for w in negative_words if w in text)
        score += pos_hits - neg_hits

    normalized = max(-1.0, min(1.0, score / max(1, len(articles) * 2)))
    return float(normalized)


def generate_prediction(ticker: str, lookback_days: int) -> PredictionResponse:
    df = _fetch_market_data(ticker, lookback_days)
    close_prices = df["Close"].values.astype(float)

    # Simple ML baseline: linear regression on recent closing trend.
    window = 30
    recent = close_prices[-window:]
    x = np.arange(len(recent)).reshape(-1, 1)
    y = recent.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    next_index = np.array([[len(recent)]])
    base_prediction = float(model.predict(next_index).flatten()[0])

    news_api_key = os.getenv("NEWS_API_KEY")
    news_query = os.getenv("NEWS_QUERY", "stock market OR economy OR inflation")
    sentiment = _simple_news_sentiment_score(news_query, news_api_key)

    # Blend model with sentiment impact (+/- 1.5% max).
    sentiment_multiplier = 1.0 + (sentiment * 0.015)
    predicted_next_close = base_prediction * sentiment_multiplier
    latest_close = float(close_prices[-1])
    expected_return_pct = ((predicted_next_close - latest_close) / latest_close) * 100.0

    signal = "HOLD"
    if expected_return_pct > 1.0:
        signal = "BUY"
    elif expected_return_pct < -1.0:
        signal = "SELL"

    prediction = PredictionResponse(
        ticker=ticker.upper(),
        predicted_next_close=round(predicted_next_close, 2),
        latest_close=round(latest_close, 2),
        expected_return_pct=round(expected_return_pct, 2),
        signal=signal,
        news_sentiment=round(sentiment, 3),
        generated_at=datetime.utcnow().isoformat(),
    )
    return prediction


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "AAPL")),
    lookback_days: int = Query(default=int(os.getenv("LOOKBACK_DAYS", "400")), ge=60, le=3000),
) -> PredictionResponse:
    try:
        prediction = generate_prediction(ticker, lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    STATE["last_prediction"] = prediction.model_dump()
    STATE["events"].insert(
        0,
        {
            "ts": datetime.utcnow().isoformat(),
            "type": "prediction",
            "ticker": prediction.ticker,
            "signal": prediction.signal,
            "expected_return_pct": prediction.expected_return_pct,
        },
    )
    STATE["events"] = STATE["events"][:100]
    return prediction


@app.get("/activity")
def activity() -> dict[str, Any]:
    return {
        "started_at": STATE["started_at"],
        "last_prediction": STATE["last_prediction"],
        "events": STATE["events"],
    }


@app.get("/", response_class=HTMLResponse)
def portal(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_ticker": os.getenv("DEFAULT_TICKER", "AAPL"),
        },
    )
