"""
Bestand: core/risk_manager.py
Relatief pad: ./core/risk_manager.py
Functie: Omgevingsgestuurde position sizing, max-position guard, en harde SL/TP (veiligheidsnet buiten de RL-agent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class RiskEngineConfig:
    sizing_mode: str
    base_trade_eur: float
    max_trade_equity_pct: float
    max_position_equity_pct: float
    stop_loss_pct: float
    take_profit_pct: float


def load_risk_engine_config() -> RiskEngineConfig:
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


def risk_profile_dict(cfg: RiskEngineConfig | None = None) -> dict[str, float | str]:
    c = cfg or load_risk_engine_config()
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
    mku = (market or "").strip().upper()
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    qty = float(pbm.get(mku, 0.0) or 0.0) if pbm else 0.0
    if qty <= 1e-12:
        qty = float(wallet.get("position_qty") or 0.0)
        sym = str(wallet.get("position_symbol") or "")
        if qty <= 0 or sym.upper() != mku:
            return 0.0
    px = float(price)
    return max(0.0, qty * px)


class RiskManager:
    """Position sizing en veiligheids-SL/TP; RL levert alleen BUY/SELL/HOLD."""

    def __init__(self, cfg: RiskEngineConfig | None = None) -> None:
        self.cfg = cfg or load_risk_engine_config()

    def hard_exit_for_sl_tp(
        self,
        *,
        market: str,
        price: float,
        wallet: dict[str, Any],
        current_volatility_pct: float | None = None,
    ) -> tuple[bool, str | None]:
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
        if str(signal or "").upper() != "BUY":
            return True, "ok"
        px = float(price)
        cap = max(0.0, float(equity)) * (self.cfg.max_position_equity_pct / 100.0)
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
        size_fraction = notional / equity (compatible met PaperTradeManager).
        """
        eq = max(1e-9, float(equity))
        sig = str(signal or "").upper()
        if sig not in {"BUY", "SELL"}:
            return 0.0, 0.0, "no_signal"
        px = float(price)
        if px <= 0:
            return 0.0, 0.0, "invalid_price"
        max_trade_eur = eq * (self.cfg.max_trade_equity_pct / 100.0)
        if self.cfg.sizing_mode == "equity_pct":
            quote = max_trade_eur
        else:
            quote = min(self.cfg.base_trade_eur, max_trade_eur)
        quote = max(0.0, min(quote, max_trade_eur, float(cash) * 0.995))
        if sig == "SELL":
            pos_val = position_value_eur(wallet, px, market)
            if pos_val <= 0:
                return 0.0, 0.0, "no_position_for_sell"
            quote = min(quote, pos_val)
        if quote <= 0:
            return 0.0, 0.0, "zero_notional"
        frac = min(1.0, quote / eq)
        return frac, quote, f"mode={self.cfg.sizing_mode}"
