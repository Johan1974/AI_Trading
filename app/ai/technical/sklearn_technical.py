"""
Bestand: app/ai/technical/sklearn_technical.py
Relatief pad: ./app/ai/technical/sklearn_technical.py
Functie: Technical analyzer op basis van Scikit-learn trendprojectie.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from app.ai.base import TechnicalAnalyzer
from app.ai.types import TechnicalResult


class SklearnTechnicalAnalyzer(TechnicalAnalyzer):
    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.model_name = "sklearn-linear-regression"

    def score(self, close_prices: np.ndarray) -> TechnicalResult:
        prices = np.asarray(close_prices, dtype=float).reshape(-1)
        if prices.size < max(10, self.window):
            raise ValueError("Onvoldoende close-prijzen voor technical analyse.")

        recent = prices[-self.window :]
        x = np.arange(len(recent)).reshape(-1, 1)
        y = recent.reshape(-1, 1)

        model = LinearRegression()
        model.fit(x, y)
        next_price = float(model.predict(np.array([[len(recent)]])).flatten()[0])
        latest_price = float(recent[-1])
        predicted_return_pct = ((next_price - latest_price) / latest_price) * 100.0

        # Normaliseer voor judge-input naar [-1, 1].
        score = max(-1.0, min(1.0, predicted_return_pct / 5.0))
        return TechnicalResult(
            score=round(score, 4),
            predicted_return_pct=round(predicted_return_pct, 4),
            model_name=self.model_name,
        )
