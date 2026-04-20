"""
Bestand: app/rl/events.py
Relatief pad: ./app/rl/events.py
Functie: Definieert grote crypto nieuws-events voor feature-annotatie in RL-training.
"""

from datetime import datetime


MAJOR_CRYPTO_EVENTS_UTC = [
    {
        "name": "us_spot_btc_etf_approval",
        "timestamp": datetime.fromisoformat("2024-01-10T21:00:00+00:00"),
        "impact": 1.0,
    },
    {
        "name": "bitcoin_halving_2024",
        "timestamp": datetime.fromisoformat("2024-04-20T00:09:00+00:00"),
        "impact": 1.4,
    },
    {
        "name": "fed_rate_cut_cycle_signal_2024",
        "timestamp": datetime.fromisoformat("2024-09-18T18:00:00+00:00"),
        "impact": 0.7,
    },
    {
        "name": "us_election_2024",
        "timestamp": datetime.fromisoformat("2024-11-05T00:00:00+00:00"),
        "impact": 0.8,
    },
    {
        "name": "macro_volatility_2025_q1",
        "timestamp": datetime.fromisoformat("2025-03-15T00:00:00+00:00"),
        "impact": 0.6,
    },
]
