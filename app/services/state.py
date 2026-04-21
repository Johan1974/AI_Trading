"""
Bestand: app/services/state.py
Relatief pad: ./app/services/state.py
Functie: Beheert de in-memory applicatiestatus en activity-events voor de portal.
"""

from contextvars import ContextVar
from datetime import datetime
from typing import Any

from app.services.paper import initialize_portfolio

_BASE_STATE: dict[str, Any] = {
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
_CURRENT_TENANT: ContextVar[str] = ContextVar("current_tenant_id", default="default")
_TENANT_STATES: dict[str, dict[str, Any]] = {"default": dict(_BASE_STATE)}


def current_tenant_id() -> str:
    raw = str(_CURRENT_TENANT.get() or "default").strip().lower()
    return raw or "default"


def set_current_tenant(tenant_id: str) -> None:
    raw = str(tenant_id or "default").strip().lower()
    _CURRENT_TENANT.set(raw or "default")


def get_tenant_state(tenant_id: str | None = None) -> dict[str, Any]:
    tid = str(tenant_id or current_tenant_id() or "default").strip().lower() or "default"
    if tid not in _TENANT_STATES:
        _TENANT_STATES[tid] = dict(_BASE_STATE)
        _TENANT_STATES[tid]["paper_portfolio"] = initialize_portfolio()
        _TENANT_STATES[tid]["events"] = []
        _TENANT_STATES[tid]["started_at"] = datetime.utcnow().isoformat()
    return _TENANT_STATES[tid]


class _TenantAwareState(dict):
    def __getitem__(self, key: str) -> Any:
        return get_tenant_state().get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        get_tenant_state()[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return get_tenant_state().get(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        get_tenant_state().update(*args, **kwargs)

    def setdefault(self, key: str, default: Any = None) -> Any:
        return get_tenant_state().setdefault(key, default)


STATE: _TenantAwareState = _TenantAwareState()


def append_event(event: dict[str, Any], max_events: int = 100) -> None:
    s = get_tenant_state()
    s["events"].insert(0, event)
    s["events"] = s["events"][:max_events]
