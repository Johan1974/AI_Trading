"""
Auto-Repair Bridge: schrijft repair_request.json bij onmogelijke portfolio-/equity-data
zodat Cursor (of een externe validator) unit_conversion / database_mapping kan voorstellen.

Zet uit met AUTO_REPAIR_BRIDGE=0. Drempel „impossible” equity-afwijking: AUTO_REPAIR_IMPOSSIBLE_REL_PCT (default 1000).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from app.datetime_util import UTC

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]


def repair_request_default_path() -> Path:
    if Path("/.dockerenv").exists():
        return Path("/app/logs") / "repair_request.json"
    return Path.cwd() / "_logs_hub" / "repair_request.json"


def _repair_bridge_enabled() -> bool:
    return str(os.getenv("AUTO_REPAIR_BRIDGE", "1")).strip().lower() in ("1", "true", "yes", "on")


def _impossible_rel_pct_threshold() -> float:
    try:
        return max(100.0, float(os.getenv("AUTO_REPAIR_IMPOSSIBLE_REL_PCT", "1000") or 1000.0))
    except (TypeError, ValueError):
        return 1000.0


def _wallet_public_summary(wallet: dict[str, Any] | None, *, max_markets: int = 24) -> dict[str, Any]:
    if not isinstance(wallet, dict):
        return {"error": "wallet_not_a_dict"}
    out: dict[str, Any] = {}
    for key in ("cash", "equity", "position_symbol", "last_price"):
        if key in wallet:
            try:
                out[key] = float(wallet[key]) if key != "position_symbol" else str(wallet.get(key) or "")
            except (TypeError, ValueError):
                out[key] = wallet.get(key)
    lp = wallet.get("last_prices_by_market")
    out["last_prices_by_market_count"] = len(lp) if isinstance(lp, dict) else 0
    obm = wallet.get("open_lots_by_market")
    if isinstance(obm, dict):
        mkts = list(obm.keys())[:max_markets]
        out["open_lots_by_market_sample"] = {}
        for m in mkts:
            lots = obm.get(m) or []
            if not isinstance(lots, list):
                continue
            slim: list[dict[str, Any]] = []
            for lot in lots[:5]:
                if not isinstance(lot, dict):
                    continue
                slim.append(
                    {
                        "qty": lot.get("qty"),
                        "entry_price": lot.get("entry_price"),
                    }
                )
            out["open_lots_by_market_sample"][str(m)] = slim
        out["open_lots_market_count"] = len(obm)
    pbm = wallet.get("position_by_market")
    if isinstance(pbm, dict):
        items = [(str(k), float(v)) for k, v in list(pbm.items())[:max_markets] if v is not None]
        try:
            out["position_by_market_sample"] = {k: v for k, v in items}
        except (TypeError, ValueError):
            out["position_by_market_sample"] = str(pbm)[:2000]
    return out


def _collect_bot_state(phase: str, extra: dict[str, Any] | None) -> dict[str, Any]:
    env_keys = (
        "PORTFOLIO_EQUITY_MISMATCH_PCT",
        "PORTFOLIO_MATH_FATAL_EXIT",
        "PAPER_EQUITY_HARD_CAP_MULT",
        "AUTO_REPAIR_BRIDGE",
        "AUTO_REPAIR_IMPOSSIBLE_REL_PCT",
        "LIVE_MODE",
        "DRY_RUN",
    )
    env_subset = {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None}
    st: dict[str, Any] = {
        "phase": phase,
        "pid": os.getpid(),
        "argv": sys.argv[:48],
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "env_subset": env_subset,
    }
    if extra:
        st["extra"] = extra
    return st


def _snippet_around_def(rel_path: str, def_name: str, *, before: int = 4, after: int = 72) -> dict[str, Any]:
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return {"path": rel_path, "note": "file_not_found", "excerpt": ""}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"path": rel_path, "note": str(exc), "excerpt": ""}
    needle = f"def {def_name}"
    idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith(needle)), -1)
    if idx < 0:
        return {"path": rel_path, "note": f"def {def_name} not found", "excerpt": ""}
    start = max(0, idx - before)
    end = min(len(lines), idx + after)
    excerpt = "\n".join(f"{j + 1:5d} | {lines[j]}" for j in range(start, end))
    return {
        "path": rel_path,
        "line_start": start + 1,
        "line_end": end,
        "anchor": def_name,
        "excerpt": excerpt,
    }


def _default_code_context() -> list[dict[str, Any]]:
    return [
        _snippet_around_def("app/services/reporting.py", "refresh_portfolio_equity_integrity"),
        _snippet_around_def("app/services/reporting.py", "reporting_coerce_btc_qty_to_btc_base"),
        _snippet_around_def("app/services/reporting.py", "sanitize_wallet_corrupt_positions_vs_cash"),
        _snippet_around_def("app/services/portfolio_qty.py", "normalize_paper_base_qty"),
        _snippet_around_def("app/services/portfolio_qty.py", "implied_equity_eur_from_wallet"),
        _snippet_around_def("core/risk_management.py", "allocation_snapshot"),
    ]


def _validator_hints(trigger: str, impossible_equity: bool) -> dict[str, Any]:
    return {
        "suggested_focus": ["unit_conversion", "database_mapping", "wallet_schema"],
        "cursor_prompt_nl": (
            "Open repair_request.json. Controleer wallet_snapshot vs reported_equity_eur / implied_equity_eur. "
            "Als qty extreem is t.o.v. equity: pas reporting_coerce_btc_qty_to_btc_base of "
            "portfolio_qty.normalize_paper_base_qty aan, of fix de bron (SQLite/paper sync). "
            "Bij invalid_weight: allocation_snapshot vs position_by_market / open_lots_by_market."
        ),
        "trigger": trigger,
        "flagged_impossible_equity_deviation": impossible_equity,
    }


def write_repair_request_if_enabled(
    *,
    detail: str,
    trigger: str,
    phase: str = "refresh_portfolio_equity_integrity",
    wallet: dict[str, Any] | None = None,
    reported_equity_eur: float | None = None,
    implied_equity_eur: float | None = None,
    rel_deviation_pct: float | None = None,
    threshold_pct: float | None = None,
    allocation_snapshot: dict[str, Any] | None = None,
    exception: BaseException | None = None,
    exception_traceback: str | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path | None:
    """
    Schrijft één JSON-bestand (atomisch) voor Cursor/validator. Geen secrets uit wallet.
    """
    if not _repair_bridge_enabled():
        return None

    imp_th = _impossible_rel_pct_threshold()
    impossible_equity = (
        rel_deviation_pct is not None and math.isfinite(rel_deviation_pct) and rel_deviation_pct >= imp_th
    )

    tb = exception_traceback
    if tb is None and exception is not None:
        tb = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    if tb is None:
        tb = traceback.format_stack()[-12:]
        tb = "".join(tb) if tb else ""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "trigger": trigger,
        "phase": phase,
        "summary": f"Portfolio/equity integrity failure: {trigger}",
        "detail": str(detail)[:8000],
        "flags": {
            "impossible_equity_deviation": bool(impossible_equity),
            "impossible_rel_pct_threshold": imp_th,
        },
        "numbers": {
            "reported_equity_eur": reported_equity_eur,
            "implied_equity_eur": implied_equity_eur,
            "rel_deviation_pct": rel_deviation_pct,
            "integrity_threshold_pct": threshold_pct,
        },
        "traceback": (tb or "")[:24000],
        "bot_state": _collect_bot_state(phase, extra_state),
        "wallet_snapshot": _wallet_public_summary(wallet),
        "allocation_snapshot": allocation_snapshot,
        "code_context": _default_code_context(),
        "validator": _validator_hints(trigger, impossible_equity),
        "feedback_loop": {
            "en": (
                "Attach this file in Cursor and ask the agent to propose a minimal patch "
                "(unit conversion or DB mapping). Accept or edit the diff in one pass."
            ),
            "nl": (
                "Voeg repair_request.json toe in Cursor-chat en vraag om een minimale patch; "
                "accepteer de voorgestelde wijziging indien correct."
            ),
        },
    }

    path = repair_request_default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        data = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        _log.warning("repair_bridge: could not write %s: %s", path, exc)
        return None
    _log.error(
        "AUTO_REPAIR_BRIDGE: wrote %s (trigger=%s impossible_equity_flag=%s)",
        path,
        trigger,
        impossible_equity,
    )
    return path
