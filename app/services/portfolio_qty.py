"""
Normalisatie van paper base-qty voor rapportage en allocatie.

Bitvavo/paper-lagen kunnen soms satoshi's (of andere schaal) als 'qty' laten staan;
zonder correctie lijkt bv. BTC-EUR op tientallen BTC bij een kleine equity.
"""

from __future__ import annotations

import logging
import math
from typing import Any

_log = logging.getLogger(__name__)

# Zeer grote float: vrijwel altijd satoshi-integer als base-eenheid (niet 64.5 "BTC").
_BTC_SAT_FLOAT_HEURISTIC_MIN = 100_000.0
_ETH_WEI_SUSPECT_ABOVE = 1e12


def _base_symbol(market: str) -> str:
    m = str(market or "").strip().upper().replace("/", "-")
    if "-" in m:
        return m.split("-", 1)[0].strip()
    return m


def normalize_paper_base_qty(
    market: str,
    qty_raw: float,
    *,
    entry_price: float | None = None,
    equity_eur: float | None = None,
) -> float:
    """
    Zet qty om naar waarschijnlijke **base units** (BTC, ETH, …) voor weergave en notional.

    Heuristieken:
    - BTC (en vergelijkbaar): zeer grote integer-achtige waarden → satoshi's (/1e8).
    - ETH: extreem grote waarden → wei (/1e18).
    - Als ``qty * entry`` de equity veel overschrijdt, één schaal-correctie proberen.
    """
    try:
        q = float(qty_raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(q) or q <= 0.0:
        return 0.0

    base = _base_symbol(market)
    eq = float(equity_eur) if equity_eur is not None and math.isfinite(float(equity_eur)) else None
    ep = float(entry_price) if entry_price is not None and math.isfinite(float(entry_price)) else None

    def _notional(qq: float) -> float | None:
        if ep is None or ep <= 0 or qq <= 0:
            return None
        n = qq * ep
        return n if math.isfinite(n) else None

    q_work = q

    # Satoshi-schaal voor BTC-achtige markten (alleen extreem grote waarden)
    if base in ("BTC", "TBTC") and q_work >= _BTC_SAT_FLOAT_HEURISTIC_MIN:
        q_try = q_work / 1e8
        n_old = _notional(q_work)
        n_new = _notional(q_try)
        if eq is not None and eq > 0 and n_old is not None and n_old > eq * 5 and n_new is not None and n_new <= eq * 3:
            _log.warning(
                "portfolio_qty: BTC qty schaal gecorrigeerd (satoshi?) market=%s raw=%s -> %s entry=%s eq=%s",
                market,
                q_work,
                q_try,
                ep,
                eq,
            )
            q_work = q_try

    # Wei-schaal voor ETH
    if base == "ETH" and q_work >= _ETH_WEI_SUSPECT_ABOVE:
        q_try = q_work / 1e18
        n_old = _notional(q_work)
        n_new = _notional(q_try)
        if eq is not None and eq > 0 and n_old is not None and n_old > eq * 5 and n_new is not None and n_new <= eq * 3:
            _log.warning(
                "portfolio_qty: ETH qty schaal gecorrigeerd (wei?) market=%s raw=%s -> %s",
                market,
                q_work,
                q_try,
            )
            q_work = q_try

    # Generiek: notional >> equity → één keer /1e8 (veel APIs gebruiken sats-integer als float)
    n = _notional(q_work)
    if eq is not None and eq > 100 and n is not None and n > eq * 50 and q_work >= 10_000:
        q_try = q_work / 1e8
        n2 = _notional(q_try)
        if n2 is not None and n2 <= eq * 5:
            _log.warning(
                "portfolio_qty: qty vermoedelijk verkeerde schaal market=%s raw=%s -> %s (notional %s -> %s eq=%s)",
                market,
                q_work,
                q_try,
                n,
                n2,
                eq,
            )
            q_work = q_try

    # BTC: qty die als "BTC" gelezen extreem veel notional geeft t.o.v. equity (bv. 64.534 i.p.v. 0.064534)
    if base in ("BTC", "TBTC") and 1.0 < q_work < _BTC_SAT_FLOAT_HEURISTIC_MIN and ep and eq and eq > 100:
        n = _notional(q_work)
        if n is not None and n > eq * 15:
            q_try = q_work / 1e8
            n2 = _notional(q_try)
            if n2 is not None and eq * 0.01 <= n2 <= eq * 5:
                _log.warning(
                    "portfolio_qty: BTC qty notional te groot — /1e8 (satoshi) market=%s raw=%s -> %s",
                    market,
                    q_work,
                    q_try,
                )
                q_work = q_try
            elif 10.0 <= q_work <= 500.0:
                q_alt = q_work / 1000.0
                n3 = _notional(q_alt)
                if n3 is not None and eq * 0.01 <= n3 <= eq * 5:
                    _log.warning(
                        "portfolio_qty: BTC qty notional te groot — /1000 (decimaal-shift) market=%s raw=%s -> %s",
                        market,
                        q_work,
                        q_alt,
                    )
                    q_work = q_alt

    return q_work


def implied_equity_eur_from_wallet(wallet: dict[str, Any]) -> float:
    """Cash + som(qty_genormaliseerd * last_price) over open posities uit lots en position_by_market."""
    if not isinstance(wallet, dict):
        return 0.0
    try:
        cash = float(wallet.get("cash", 0.0) or 0.0)
    except (TypeError, ValueError):
        cash = 0.0
    if not math.isfinite(cash):
        cash = 0.0

    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
    eq_hint = None
    try:
        eq_hint = float(wallet.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        eq_hint = None
    try:
        anchor = float(wallet.get("paper_anchor_equity_eur") or wallet.get("starting_balance_eur") or 1000.0)
    except (TypeError, ValueError):
        anchor = 1000.0
    anchor = max(100.0, anchor)
    # Bij corrupt/inflate ``equity`` toch realistische schaal voor qty-normalisatie
    if eq_hint is None or not math.isfinite(eq_hint) or eq_hint < 1.0:
        eq_hint = anchor
    elif eq_hint > anchor * 5:
        eq_hint = anchor

    crypto = 0.0
    for mkt, lots in obm.items():
        if not isinstance(lots, list):
            continue
        mku = str(mkt).strip().upper().replace("/", "-")
        px = 0.0
        try:
            px = float(lp.get(mku, 0.0) or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        if px <= 0 and isinstance(lp, dict):
            for lk, lv in lp.items():
                if str(lk).strip().upper().replace("/", "-") == mku:
                    try:
                        px = float(lv or 0.0)
                    except (TypeError, ValueError):
                        px = 0.0
                    break
        if px <= 0 and mku == str(wallet.get("position_symbol") or "").strip().upper().replace("/", "-"):
            try:
                px = float(wallet.get("last_price") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
        for lot in lots:
            if not isinstance(lot, dict):
                continue
            try:
                rq = float(lot.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                ep = float(lot.get("entry_price") or 0.0) or None
            except (TypeError, ValueError):
                ep = None
            qn = normalize_paper_base_qty(mku, rq, entry_price=ep, equity_eur=eq_hint)
            use_px = px if px > 0 else (ep if ep and ep > 0 else 0.0)
            if use_px > 0 and qn > 0:
                crypto += qn * use_px

    if crypto <= 0 and isinstance(wallet.get("position_by_market"), dict):
        pbm = wallet["position_by_market"]
        for mkt, qv in pbm.items():
            try:
                qq = float(qv or 0.0)
            except (TypeError, ValueError):
                continue
            mku = str(mkt).strip().upper().replace("/", "-")
            px = 0.0
            try:
                px = float(lp.get(mku, 0.0) or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px <= 0 and isinstance(lp, dict):
                for lk, lv in lp.items():
                    if str(lk).strip().upper().replace("/", "-") == mku:
                        try:
                            px = float(lv or 0.0)
                        except (TypeError, ValueError):
                            px = 0.0
                        break
            if px <= 0 and mku == str(wallet.get("position_symbol") or "").strip().upper().replace("/", "-"):
                try:
                    px = float(wallet.get("last_price") or 0.0)
                except (TypeError, ValueError):
                    px = 0.0
            qn = normalize_paper_base_qty(mku, qq, entry_price=None, equity_eur=eq_hint)
            if px > 0 and qn > 0:
                crypto += qn * px

    out = cash + max(0.0, crypto)
    return out if math.isfinite(out) else cash
