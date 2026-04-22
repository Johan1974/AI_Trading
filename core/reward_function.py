"""
Bestand: core/reward_function.py
Relatief pad: ./core/reward_function.py
Functie: Gecentraliseerde RL-beloning: PnL-% t.o.v. vorige stap, exponentiële drawdown-straf (>2% onder HWM),
         transactiewrijving, consistency-bonus (HOLD + whales + winstpositie), en optionele stop-loss-shock.
"""

from __future__ import annotations

import math
import os


def _f(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


# Schaal: staprendement (fraction) → reward (ordergelijk aan eerdere EUR/alpha-mix).
REWARD_PNL_SCALE = _f("RL_REWARD_PNL_SCALE", "400.0")
# Drawdown t.o.v. piek (0..1); pas straf toe boven deze drempel (default 2%).
REWARD_DD_THRESHOLD = _f("RL_REWARD_DD_THRESHOLD", "0.02")
REWARD_DD_EXP_K = _f("RL_REWARD_DD_EXP_K", "6.0")
REWARD_DD_EXP_SCALE = _f("RL_REWARD_DD_EXP_SCALE", "1.25")
# Vaste straf per uitgevoerde trade (overtrading remmen).
TRANSACTION_FRICTION = _f("RL_TRANSACTION_FRICTION", "0.06")
# Bonus: HOLD vasthouden terwijl whales sterk bullish zijn en positie in the green.
CONSISTENCY_BONUS = _f("RL_CONSISTENCY_BONUS", "0.05")
CONSISTENCY_WHALE_MIN = _f("RL_CONSISTENCY_WHALE_MIN", "0.5")
# Harde straf wanneer gesimuleerde risk-stop wordt geraakt (align met RISK_STOP_LOSS_PCT in de env).
STOP_LOSS_SHOCK = _f("RL_STOP_LOSS_SHOCK", "12.0")
# Straf per stap als een positie te lang open staat (stagnatie).
STAGNATION_MAX_HOURS = _f("RL_STAGNATION_MAX_HOURS", "72.0")
STAGNATION_PENALTY = _f("RL_STAGNATION_PENALTY", "0.02")


def compute_trading_step_reward(
    *,
    equity: float,
    last_equity: float,
    equity_peak: float,
    initial_balance_eur: float,
    action: int,
    executed_trade: bool,
    whale_pressure: float,
    position_btc: float,
    entry_price: float,
    current_price: float,
    forced_stop_loss: bool,
    position_hours: float = 0.0,
) -> float:
    """
    Beloning voor één omgevingsstap na uitvoering van de actie (en eventuele risk-forced exit).

    - PnL: procentuele verandering equity t.o.v. vorige stap.
    - Drawdown: exponentiële penalty zodra portfolio > drempel onder high-water mark zakt.
    - Friction: kleine vaste aftrek per trade.
    - Consistency: bonus bij HOLD, whale_pressure > drempel, en positieve unrealized PnL.
    - Stop-loss shock: grote negatieve spike als risk-stop in de simulatie wordt getriggerd.
    - Stagnation: lichte straf per stap als de trade langer open staat dan STAGNATION_MAX_HOURS.
    """
    _ = initial_balance_eur  # gereserveerd voor toekomstige normalisatie / logging

    le = max(float(last_equity), 1.0)
    step_return_frac = (float(equity) - float(last_equity)) / le
    pnl_reward = step_return_frac * REWARD_PNL_SCALE

    peak = max(float(equity_peak), 1e-9)
    dd_ratio = max(0.0, (peak - float(equity)) / peak)
    excess_dd = max(0.0, dd_ratio - REWARD_DD_THRESHOLD)
    drawdown_penalty = -((math.exp(REWARD_DD_EXP_K * excess_dd) - 1.0) * REWARD_DD_EXP_SCALE)

    friction = -TRANSACTION_FRICTION if executed_trade else 0.0

    unrealized_eur = 0.0
    if position_btc > 1e-12 and entry_price > 0 and current_price > 0:
        unrealized_eur = float(position_btc) * (float(current_price) - float(entry_price))
    consistency = 0.0
    if (
        int(action) == 0
        and float(whale_pressure) > CONSISTENCY_WHALE_MIN
        and position_btc > 1e-12
        and unrealized_eur > 0.0
    ):
        consistency = CONSISTENCY_BONUS

    sl_shock = -STOP_LOSS_SHOCK if forced_stop_loss else 0.0

    stagnation = 0.0
    if position_btc > 1e-12 and position_hours > STAGNATION_MAX_HOURS:
        stagnation = -STAGNATION_PENALTY
        
        # Tijdelijke debug: print de straf naar de terminal (zonder de logs te overspoelen)
        # print(f"[REWARD DEBUG] Stagnatie straf! uren={position_hours:.1f}, penalty={stagnation:.2f}, totaal_reward={(pnl_reward + drawdown_penalty + friction + consistency + sl_shock + stagnation):.4f}")

    return float(pnl_reward + drawdown_penalty + friction + consistency + sl_shock + stagnation)
