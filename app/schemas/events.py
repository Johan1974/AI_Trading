"""
Bestand: app/schemas/events.py
Relatief pad: ./app/schemas/events.py
Functie: Definieert event- en activity-schema's voor logging en monitoring.
"""

from pydantic import BaseModel


class PredictionEvent(BaseModel):
    ts: str
    type: str
    ticker: str
    signal: str
    expected_return_pct: float
