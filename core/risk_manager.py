"""
Bestand: core/risk_manager.py
Relatief pad: ./core/risk_manager.py
Functie: Omgevingsgestuurde position sizing, max-position guard, en harde SL/TP (veiligheidsnet buiten de RL-agent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from core.trading_constraints_redis import read_trading_constraints

Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class RiskEngineConfig:
    sizing_mode: str
    base_trade_eur: float
    max_trade_equity_pct: float
    max_position_equity_pct: float
    stop_loss_pct: float
    take_profit_pct: float


def _risk_config_from_env() -> RiskEngineConfig:
    mode = os.getenv("RISK_SIZING_MODE", "fixed_eur").strip().lower()
    if mode not in ("fixed_eur", "equity_pct"):
        mode = "fixed_eur"
    return RiskEngineConfig(
        sizing_mode=mode,
        base_trade_eur=float(os.getenv("RISK_BASE_TRADE_EUR", "100")),
        max_trade_equity_pct=float(os.getenv("RISK_MAX_TRADE_EQUITY_PCT", "10")),
        max_position_equity_pct=float(os.getenv("RISK_MAX_POSITION_EQUITY_PCT", "40")),
        stop_loss_pct=float(os.getenv("RISK_STOP_LOSS_PCT", "2.5")),
        take_profit_pct=float(os.getenv("RISK_TAKE_PROFIT_PCT", "5.0")),
    )


def load_risk_engine_config() -> RiskEngineConfig:
    """Env-baseline + optionele overrides uit Redis ``trading:constraints``."""
    base = _risk_config_from_env()
    d = read_trading_constraints()
    if not d:
        return base
    mode = str(d.get("sizing_mode", base.sizing_mode)).strip().lower()
    if mode not in ("fixed_eur", "equity_pct"):
        mode = base.sizing_mode
    be_raw, mx_raw = d.get("base_trade_eur"), d.get("max_trade_equity_pct")
    try:
        base_trade = float(be_raw) if be_raw is not None else base.base_trade_eur
    except (TypeError, ValueError):
        base_trade = base.base_trade_eur
    try:
        max_trade = float(mx_raw) if mx_raw is not None else base.max_trade_equity_pct
    except (TypeError, ValueError):
        max_trade = base.max_trade_equity_pct
    return RiskEngineConfig(
        sizing_mode=mode,
        base_trade_eur=max(1.0, base_trade),
        max_trade_equity_pct=max(0.05, min(100.0, max_trade)),
        max_position_equity_pct=base.max_position_equity_pct,
        stop_loss_pct=base.stop_loss_pct,
        take_profit_pct=base.take_profit_pct,
    )


def risk_profile_dict(cfg: RiskEngineConfig | RiskManager | None = None) -> dict[str, float | str]:
    if cfg is None or isinstance(cfg, RiskManager):
        c = load_risk_engine_config()
    else:
        c = cfg
    return {
        "sizing_mode": c.sizing_mode,
        "base_trade_eur": c.base_trade_eur,
        "max_risk_pct": c.max_trade_equity_pct,
        "max_position_pct": c.max_position_equity_pct,
        "stop_loss_pct": c.stop_loss_pct,
        "take_profit_pct": c.take_profit_pct,
    }


def risk_controls_for_close(latest_close: float) -> dict[str, float]:
    """Zelfde keys als app.services.risk.compute_risk_controls (SL/TP-prijzen + fractie)."""
    c = load_risk_engine_config()
    px = float(latest_close)
    frac = min(1.0, max(0.0, c.max_trade_equity_pct / 100.0))
    return {
        "stop_loss_price": round(px * (1.0 - c.stop_loss_pct / 100.0), 2),
        "take_profit_price": round(px * (1.0 + c.take_profit_pct / 100.0), 2),
        "position_size_fraction": frac,
    }


def weighted_avg_entry(wallet: dict[str, Any], market: str | None = None) -> float:
    lots: list[Any] = []
    mku = str(market or "").strip().upper()
    if mku:
        bym = wallet.get("open_lots_by_market") or {}
        if isinstance(bym, dict):
            raw = bym.get(mku)
            if isinstance(raw, list):
                lots = [x for x in raw if isinstance(x, dict)]
        if not lots and str(wallet.get("position_symbol") or "").upper() == mku:
            flat = wallet.get("open_lots") or []
            if isinstance(flat, list):
                lots = [x for x in flat if isinstance(x, dict)]
    else:
        flat = wallet.get("open_lots") or []
        if isinstance(flat, list):
            lots = [x for x in flat if isinstance(x, dict)]
    if not isinstance(lots, list) or not lots:
        return float(wallet.get("avg_entry_price") or 0.0)
    total_qty = 0.0
    cost = 0.0
    for lot in lots:
        if not isinstance(lot, dict):
            continue
        q = max(0.0, float(lot.get("qty", 0.0) or 0.0))
        ep = float(lot.get("entry_price", 0.0) or 0.0)
        total_qty += q
        cost += q * ep
    if total_qty <= 0:
        return float(wallet.get("avg_entry_price") or 0.0)
    return cost / total_qty


def position_value_eur(wallet: dict[str, Any], price: float, market: str) -> float:
    mku = (market or "").strip().upper().replace("/", "-")
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    qty = float(pbm.get(mku, 0.0) or 0.0) if pbm else 0.0
    if qty <= 1e-12 and pbm:
        for k, q in pbm.items():
            if str(k).strip().upper().replace("/", "-") == mku:
                qty = float(q or 0.0)
                break
    if qty <= 1e-12:
        qty = float(wallet.get("position_qty") or 0.0)
        sym = str(wallet.get("position_symbol") or "").strip().upper().replace("/", "-")
        if qty <= 0 or sym != mku:
            return 0.0
    px = float(price)
    return max(0.0, qty * px)


def iron_strict_allocation_enabled() -> bool:
    return str(os.getenv("IRON_STRICT_ALLOCATION", "1")).strip().lower() in ("1", "true", "yes", "on")


def paper_base_from_market(m: str) -> str:
    mku = str(m or "").strip().upper().replace("/", "-")
    return mku.split("-", 1)[0].strip() if "-" in mku else mku


def _bases_with_open_qty(wallet: dict[str, Any]) -> set[str]:
    bases: set[str] = set()
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    for mk, qv in pbm.items():
        if float(qv or 0.0) > 1e-12:
            bases.add(paper_base_from_market(str(mk)))
    obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
    for mk, lots in obm.items():
        if not isinstance(lots, list):
            continue
        tq = sum(float(x.get("qty") or 0) for x in lots if isinstance(x, dict))
        if tq > 1e-12:
            bases.add(paper_base_from_market(str(mk)))
    sym = str(wallet.get("position_symbol") or "").strip().upper().replace("/", "-")
    if sym and float(wallet.get("position_qty") or 0.0) > 1e-12:
        bases.add(paper_base_from_market(sym))
    return bases


def open_distinct_coin_count(wallet: dict[str, Any]) -> int:
    return len(_bases_with_open_qty(wallet))


def total_open_market_value_eur(wallet: dict[str, Any]) -> float:
    """Som marktwaarde open posities (qty × prijs; last_prices_by_market, anders gewogen entry)."""
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    total = 0.0
    for mkt, qv in pbm.items():
        q = float(qv or 0.0)
        if q <= 1e-12:
            continue
        mku = str(mkt).strip().upper().replace("/", "-")
        pxv = float(lp.get(mku, 0.0) or 0.0)
        if pxv <= 0:
            pxv = weighted_avg_entry(wallet, mku)
        total += q * max(0.0, pxv)
    return max(0.0, total)


def paper_iron_lockdown_buy_ok(
    wallet: dict[str, Any],
    market: str,
    price: float,
    proposed_quote_eur: float,
    fee_rate: float = 0.0015,
) -> tuple[bool, str]:
    """
    Harde allocatie-/exposure-limieten voor paper BUY (risk_manager + PaperTradeManager).
    """
    if not iron_strict_allocation_enabled():
        return True, "ok"
    mku = str(market or "").strip().upper().replace("/", "-")
    base = paper_base_from_market(mku)
    max_trade = float(os.getenv("IRON_MAX_TRADE_EUR", "100"))
    if float(proposed_quote_eur) > max_trade + 1e-6:
        return False, "iron_max_trade_eur_exceeded"
    max_coins = int(float(os.getenv("IRON_MAX_OPEN_COINS", "8")))
    max_expo = float(os.getenv("IRON_MAX_TOTAL_EXPOSURE_EUR", "800"))
    bases = _bases_with_open_qty(wallet)
    cur_mv = total_open_market_value_eur(wallet)
    if cur_mv > max_expo + 1e-6:
        return False, "iron_total_exposure_cap_exceeded"
    if base not in bases and len(bases) >= max_coins:
        return False, "iron_max_open_coins_exceeded"
    projected = cur_mv + max(0.0, float(proposed_quote_eur))
    if projected > max_expo + 1e-6:
        return False, "iron_projected_exposure_cap_exceeded"
    return True, "ok"


class RiskManager:
    """Position sizing en veiligheids-SL/TP; RL levert alleen BUY/SELL/HOLD."""

    def __init__(self, cfg: RiskEngineConfig | None = None) -> None:
        self.cfg = cfg or load_risk_engine_config()

    def _sync_cfg(self) -> None:
        self.cfg = load_risk_engine_config()

    def hard_exit_for_sl_tp(
        self,
        *,
        market: str,
        price: float,
        wallet: dict[str, Any],
        current_volatility_pct: float | None = None,
    ) -> tuple[bool, str | None]:
        self._sync_cfg()
        mku = (market or "").strip().upper()
        qty = float((wallet.get("position_by_market") or {}).get(mku, 0.0) or 0.0) if isinstance(
            wallet.get("position_by_market"), dict
        ) else 0.0
        if qty <= 1e-12:
            if "position_peak_prices" in wallet:
                wallet["position_peak_prices"].pop(mku, None)
            qty = float(wallet.get("position_qty") or 0.0)
            sym = str(wallet.get("position_symbol") or "")
            if qty <= 0 or sym.upper() != mku:
                return False, None
        entry = weighted_avg_entry(wallet, market=mku)
        px = float(price)
        if entry <= 0 or px <= 0:
            return False, None
            
        # Trailing Stop: Bewaar de hoogste koers sinds instap
        peaks = wallet.setdefault("position_peak_prices", {})
        peak = max(float(peaks.get(mku, entry)), px)
        peaks[mku] = peak

        # ATR-gebaseerde Trailing Stop: Gebruik de hoogste van (vaste SL) of (2.5x ATR)
        base_sl_pct = self.cfg.stop_loss_pct
        dyn_sl_pct = max(base_sl_pct, current_volatility_pct * 2.5) if current_volatility_pct else base_sl_pct
        
        sl_mult = 1.0 - (dyn_sl_pct / 100.0)
        tp_mult = 1.0 + (self.cfg.take_profit_pct / 100.0)
        if px <= peak * sl_mult:
            return True, "hard_trailing_stop"
        if px >= entry * tp_mult:
            return True, "hard_take_profit"
        return False, None

    def full_exit_size_fraction(self, *, equity: float, wallet: dict[str, Any], price: float, market: str) -> float:
        eq = max(1e-9, float(equity))
        pv = position_value_eur(wallet, price, market)
        if pv <= 0:
            return 0.0
        return min(1.0, pv / eq)

    def check_safety(
        self,
        *,
        signal: str,
        market: str,
        equity: float,
        cash: float,
        price: float,
        wallet: dict[str, Any],
        proposed_quote_eur: float,
        fee_rate: float = 0.0015,
    ) -> tuple[bool, str]:
        self._sync_cfg()
        if str(signal or "").upper() != "BUY":
            return True, "ok"
        from core.trading_engine import (
            has_active_paper_position_for_base_currency,
            has_active_paper_position_for_ticker,
        )

        if has_active_paper_position_for_ticker(wallet, market):
            return False, "duplicate_open_pair_guard"
        if str(os.getenv("PAPER_ENFORCE_ONE_OPEN_TRADE_PER_BASE", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) and has_active_paper_position_for_base_currency(wallet, market):
            return False, "duplicate_open_base_currency_guard"
        ok_iron, why_iron = paper_iron_lockdown_buy_ok(
            wallet, market, float(price), proposed_quote_eur, fee_rate
        )
        if not ok_iron:
            return False, why_iron
        px = float(price)
        anchor = float(
            wallet.get("paper_anchor_equity_eur")
            or wallet.get("starting_balance_eur")
            or equity
        )
        anchor = max(1e-9, anchor)
        cap = anchor * (self.cfg.max_position_equity_pct / 100.0)
        current = position_value_eur(wallet, price, market)
        projected = current + max(0.0, proposed_quote_eur)
        if projected > cap + 1e-6:
            return False, "max_position_size_exceeded"
        fee_est = max(0.0, proposed_quote_eur) * max(0.0, float(fee_rate))
        if float(cash) < proposed_quote_eur + fee_est - 1e-9:
            return False, "insufficient_cash_for_buy"
        return True, "ok"

    def calculate_trade_size(
        self,
        *,
        signal: str,
        equity: float,
        cash: float,
        price: float,
        wallet: dict[str, Any],
        market: str,
    ) -> tuple[float, float, str]:
        """
        Geeft (size_fraction, amount_quote_eur, note).
        BUY: 10% van beschikbare cash; vloer €10 (Bitvavo-minimum), anders overgeslagen.
        SELL: size_fraction = quote / live equity (afsluitfractie).
        """
        self._sync_cfg()
        eq = max(1e-9, float(equity))
        c = max(0.0, float(cash))
        sig = str(signal or "").upper()
        if sig not in {"BUY", "SELL"}:
            return 0.0, 0.0, "no_signal"
        px = float(price)
        if px <= 0:
            return 0.0, 0.0, "invalid_price"
        sizing_eq = max(1e-9, c) if sig == "BUY" else eq
        max_trade_eur = sizing_eq * (self.cfg.max_trade_equity_pct / 100.0)
        if self.cfg.sizing_mode == "equity_pct":
            quote = max_trade_eur
        else:
            quote = min(self.cfg.base_trade_eur, max_trade_eur)
        quote = max(0.0, min(quote, max_trade_eur))
        if sig == "BUY" and iron_strict_allocation_enabled():
            cap_iron = float(os.getenv("IRON_MAX_TRADE_EUR", "100"))
            quote = min(quote, max(0.0, cap_iron))
        if sig == "SELL":
            quote = min(quote, c * 0.995)
            pos_val = position_value_eur(wallet, px, market)
            if pos_val <= 0:
                return 0.0, 0.0, "no_position_for_sell"
            quote = min(quote, pos_val)
        if quote <= 0:
            return 0.0, 0.0, "zero_notional"
        if sig == "BUY" and quote < 10.0:
            return 0.0, 0.0, "below_bitvavo_min_trade"
        denom = max(1e-9, c) if sig == "BUY" else eq
        frac = min(1.0, quote / denom)
        return frac, quote, f"mode={self.cfg.sizing_mode}"


def sentiment_buy_threshold_offset(agg_sentiment: float) -> float:
    """Verlaag buy_confidence_threshold met 0.05 bij sterk positief sentiment (> 0.6)."""
    try:
        s = float(agg_sentiment)
    except (TypeError, ValueError):
        return 0.0
    return -0.05 if s > 0.6 else 0.0
