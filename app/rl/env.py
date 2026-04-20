"""
Bestand: app/rl/env.py
Relatief pad: ./app/rl/env.py
Functie: Gymnasium trading omgeving; beloning via core.reward_function (PnL-%, drawdown, friction, consistency, SL-shock).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from core.reward_function import compute_trading_step_reward
from core.risk_manager import load_risk_engine_config


class BitvavoTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance_eur: float = 10000.0,
        fee_bps: float = 8.0,
        max_trade_notional_eur: float = 500.0,
        max_trades: int = 10000,
        drawdown_penalty_factor: float = 0.25,
    ) -> None:
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.initial_balance_eur = initial_balance_eur
        self.fee_bps = fee_bps
        self.max_trade_notional_eur = max_trade_notional_eur
        self.max_trades = max_trades
        # Legacy: beloning zit in core.reward_function; parameter blijft voor backwards compatibility.
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self._risk_cfg = load_risk_engine_config()

        self.feature_cols = [
            "price_action",
            "volatility_24",
            "volume_change",
            "sentiment_score",
            "news_confidence",
            "social_volume",
            "fear_greed_score",
            "btc_dominance_pct",
            "whale_pressure",
            "macro_volatility_window",
            "bollinger_width",
            "bollinger_position",
            "orderbook_imbalance",
            "macd",
            "rsi_14",
            "ema_gap_pct",
        ]
        for col in self.feature_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        # 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        obs_size = len(self.feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.step_idx = 0
        self.balance_eur = self.initial_balance_eur
        self.position_btc = 0.0
        self.entry_price = 0.0
        self.trade_count = 0
        self.equity_peak = self.initial_balance_eur
        self.last_equity = self.initial_balance_eur

    def _current_price(self) -> float:
        return float(self.df.loc[self.step_idx, "close"])

    def _equity(self) -> float:
        return float(self.balance_eur + self.position_btc * self._current_price())

    def _build_observation(self) -> np.ndarray:
        row = self.df.loc[self.step_idx, self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        equity = self._equity()
        account = np.array(
            [
                self.balance_eur / self.initial_balance_eur,
                self.position_btc,
                (equity - self.initial_balance_eur) / self.initial_balance_eur,
                self.trade_count / max(1, self.max_trades),
            ],
            dtype=np.float32,
        )
        return np.concatenate([row, account], axis=0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._build_observation()
        info = {"equity_eur": self._equity(), "trade_count": self.trade_count}
        return obs, info

    def step(self, action: int):
        price = self._current_price()
        fee_rate = self.fee_bps / 10000.0
        executed_trade = False
        forced_stop_loss = False

        # Execution rules (RL-actie eerst; daarna harde stop-loss zoals core.risk_manager)
        if action == 1 and self.balance_eur > 10:
            notional = min(self.max_trade_notional_eur, self.balance_eur)
            fee = notional * fee_rate
            qty = max(0.0, (notional - fee) / price)
            self.balance_eur -= notional
            self.position_btc += qty
            self.entry_price = price
            self.trade_count += 1
            executed_trade = True
        elif action == 2 and self.position_btc > 0:
            qty = self.position_btc
            notional = qty * price
            fee = notional * fee_rate
            self.balance_eur += max(0.0, notional - fee)
            self.position_btc = 0.0
            self.entry_price = 0.0
            self.trade_count += 1
            executed_trade = True

        if self.position_btc > 1e-12 and self.entry_price > 1e-12:
            sl_mult = 1.0 - (self._risk_cfg.stop_loss_pct / 100.0)
            if price <= self.entry_price * sl_mult:
                qty = self.position_btc
                notional = qty * price
                fee = notional * fee_rate
                self.balance_eur += max(0.0, notional - fee)
                self.position_btc = 0.0
                self.entry_price = 0.0
                self.trade_count += 1
                executed_trade = True
                forced_stop_loss = True

        equity = self._equity()
        peak_for_dd = max(self.equity_peak, equity)
        drawdown = 0.0 if peak_for_dd <= 0 else (peak_for_dd - equity) / peak_for_dd
        whale_pressure = float(self.df.loc[self.step_idx, "whale_pressure"]) if self.step_idx < len(self.df) else 0.0

        reward = compute_trading_step_reward(
            equity=equity,
            last_equity=self.last_equity,
            equity_peak=peak_for_dd,
            initial_balance_eur=self.initial_balance_eur,
            action=int(action),
            executed_trade=executed_trade,
            whale_pressure=whale_pressure,
            position_btc=self.position_btc,
            entry_price=self.entry_price,
            current_price=price,
            forced_stop_loss=forced_stop_loss,
        )

        self.last_equity = equity
        self.equity_peak = max(self.equity_peak, equity)

        self.step_idx += 1
        terminated = self.step_idx >= (len(self.df) - 1)
        truncated = self.trade_count >= self.max_trades

        obs = self._build_observation() if not (terminated or truncated) else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = {
            "equity_eur": round(equity, 2),
            "balance_eur": round(self.balance_eur, 2),
            "position_btc": self.position_btc,
            "drawdown": round(drawdown, 6),
            "trade_count": self.trade_count,
            "pnl_eur_total": round(equity - self.initial_balance_eur, 2),
            "forced_stop_loss": forced_stop_loss,
        }
        return obs, reward, terminated, truncated, info
