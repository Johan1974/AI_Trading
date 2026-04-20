"""
Bestand: app/services/model.py
Relatief pad: ./app/services/model.py
Functie: Bevat de baseline voorspellingslogica voor de volgende slotprijs.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def predict_next_close_from_trend(x: np.ndarray, y: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(x, y)
    next_index = np.array([[len(x)]])
    return float(model.predict(next_index).flatten()[0])
