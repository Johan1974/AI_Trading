"""
Bestand: app/services/state.py
Relatief pad: ./app/services/state.py
Functie: Beheert de in-memory applicatiestatus en activity-events voor de portal.
"""

from datetime import datetime
from typing import Any

from app.services.paper import initialize_portfolio


STATE: dict[str, Any] = {
    "bot_status": "running",
    "selected_market": "BTC-EUR",
    "active_markets": [],
    "last_prediction": None,
    "last_scores": None,
    "last_order": None,
    "news_insights": [],
    "news_lag_history": [],
    "macro_context": {},
    "fear_greed": {},
    "whale_watch": {},
    "cmc_metrics": {},
    "rl_last_state": {},
    "signal_markers": [],
    "paper_portfolio": initialize_portfolio(),
    "events": [],
    "started_at": datetime.utcnow().isoformat(),
}


def append_event(event: dict[str, Any], max_events: int = 100) -> None:
    STATE["events"].insert(0, event)
    STATE["events"] = STATE["events"][:max_events]
