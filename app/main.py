"""
Bestand: app/main.py
Relatief pad: ./app/main.py
Functie: FastAPI-app voor marktdata- en nieuwsgebaseerde voorspellingen plus activiteitstracking voor de portal.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.schemas.prediction import PredictionResponse
from app.services.execution import build_paper_order
from app.services.features import build_trend_window, compute_simple_news_sentiment
from app.services.ingestion import fetch_market_data, fetch_news_articles
from app.services.model import predict_next_close_from_trend
from app.services.risk import compute_risk_controls, signal_from_expected_return
from app.services.state import STATE, append_event

load_dotenv()

app = FastAPI(title="AI Trading Bot", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def generate_prediction(ticker: str, lookback_days: int) -> PredictionResponse:
    df = fetch_market_data(ticker, lookback_days)
    close_prices = df["Close"].values.astype(float)

    x, y = build_trend_window(close_prices, window=30)
    base_prediction = predict_next_close_from_trend(x, y)

    news_api_key = os.getenv("NEWS_API_KEY")
    news_query = os.getenv("NEWS_QUERY", "stock market OR economy OR inflation")
    articles = fetch_news_articles(news_query, news_api_key)
    sentiment = compute_simple_news_sentiment(articles)

    # Blend model with sentiment impact (+/- 1.5% max).
    sentiment_multiplier = 1.0 + (sentiment * 0.015)
    predicted_next_close = base_prediction * sentiment_multiplier
    latest_close = float(close_prices[-1])
    expected_return_pct = ((predicted_next_close - latest_close) / latest_close) * 100.0

    signal = signal_from_expected_return(expected_return_pct)

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
    risk_controls = compute_risk_controls(prediction.latest_close)
    paper_order = build_paper_order(
        signal=prediction.signal,
        ticker=prediction.ticker,
        price=prediction.latest_close,
        size_fraction=risk_controls["position_size_fraction"],
    )
    STATE["last_order"] = {"risk_controls": risk_controls, "order": paper_order}

    append_event(
        {
            "ts": datetime.utcnow().isoformat(),
            "type": "prediction",
            "ticker": prediction.ticker,
            "signal": prediction.signal,
            "expected_return_pct": prediction.expected_return_pct,
        }
    )
    return prediction


@app.get("/activity")
def activity() -> dict[str, Any]:
    return {
        "started_at": STATE["started_at"],
        "last_prediction": STATE["last_prediction"],
        "last_order": STATE["last_order"],
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
