"""
Bestand: app/services/state.py
Relatief pad: ./app/services/state.py
Functie: Beheert de in-memory applicatiestatus en activity-events voor de portal.
"""

from datetime import datetime
from typing import Any


STATE: dict[str, Any] = {
    "last_prediction": None,
    "last_order": None,
    "events": [],
    "started_at": datetime.utcnow().isoformat(),
}


def append_event(event: dict[str, Any], max_events: int = 100) -> None:
    STATE["events"].insert(0, event)
    STATE["events"] = STATE["events"][:max_events]
