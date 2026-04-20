"""
Bestand: app/schemas/prediction.py
Relatief pad: ./app/schemas/prediction.py
Functie: Definieert het API-response model voor prijsvoorspellingen en signalen.
"""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    ticker: str
    predicted_next_close: float
    latest_close: float
    expected_return_pct: float
    signal: str
    news_sentiment: float
    generated_at: str
