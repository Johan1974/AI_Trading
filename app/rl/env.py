"""
Bestand: app/rl/env.py
Relatief pad: ./app/rl/env.py
Functie: Gymnasium trading omgeving; beloning via core.reward_function (PnL-%, drawdown, friction, consistency, SL-shock).
"""

from __future__ import annotations

from datetime import datetime
from app.datetime_util import UTC
import logging
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from core.preprocessor import attention_gate_weights
from core.reward_function import compute_trading_step_reward
from core.risk_manager import load_risk_engine_config

_log = logging.getLogger(__name__)


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
        interval_hours: float = 1.0,
    ) -> None:
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.initial_balance_eur = initial_balance_eur
        self.fee_bps = fee_bps
        self.max_trade_notional_eur = max_trade_notional_eur
        self.max_trades = max_trades
        # Legacy: beloning zit in core.reward_function; parameter blijft voor backwards compatibility.
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self.interval_hours = interval_hours
        self._risk_cfg = load_risk_engine_config()
        self._attention_enabled = True

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
            "bid_ask_spread",
            "macd",
            "rsi_14",
            "ema_gap_pct",
        ]
        for col in self.feature_cols:
            if col not in self.df.columns:
                print(
                    f"WARNING: RL observation column '{col}' ontbreekt in training frame; "
                    "gevuld met 0.0 (mogelijke dead channel)."
                )
                self.df[col] = 0.0

        # 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        self._reset_state()
        sample_obs = self._build_observation()
        obs_shape = sample_obs.shape[0]
        
        # ALIGNMENT CHECK: 17 marktsignalen + 5 account variabelen = 22.
        assert obs_shape == 22, f"Observation space mismatch! Verwacht 22 features, kreeg {obs_shape}."

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

    def _safe_row_idx(self) -> int:
        """Voorkomt ``df.loc[step_idx]`` OOB als de policy langer doorloopt dan er rijen zijn."""
        n = len(self.df)
        if n <= 0:
            return 0
        return int(max(0, min(self.step_idx, n - 1)))

    def _reset_state(self) -> None:
        self.step_idx = 0
        self.balance_eur = self.initial_balance_eur
        self.position_btc = 0.0
        self.entry_price = 0.0
        self.position_peak_price = 0.0
        self.position_hours = 0.0
        self.trade_count = 0
        self.equity_peak = self.initial_balance_eur
        self.last_equity = self.initial_balance_eur

    def _current_price(self) -> float:
        return float(self.df.loc[self._safe_row_idx(), "close"])

    def _equity(self) -> float:
        return float(self.balance_eur + self.position_btc * self._current_price())

    def _close_position(self, price: float, fee_rate: float) -> None:
        """Sluit de huidige positie en verwerkt de balans update en metrics."""
        qty = self.position_btc
        notional = qty * price
        fee = notional * fee_rate
        self.balance_eur += max(0.0, notional - fee)
        self.position_btc = 0.0
        self.entry_price = 0.0
        self.position_peak_price = 0.0
        self.position_hours = 0.0
        self.trade_count += 1

    def _build_observation(self) -> np.ndarray:
        row = self.df.loc[self._safe_row_idx(), self.feature_cols].astype(float).to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(row)):
            print(
                f"WARNING: Observation vector bevat NaN/Inf op step_idx={self.step_idx}; "
                "waarden worden gesaneerd (nan_to_num)."
            )
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if self._attention_enabled and row.size:
            # Lightweight feature-attention gating before policy input.
            gate = attention_gate_weights(row, temperature=0.7)
            row = (row * gate).astype(np.float32)
        equity = self._equity()
        account = np.array(
            [
                self.balance_eur / self.initial_balance_eur,
                self.position_btc,
                (equity - self.initial_balance_eur) / self.initial_balance_eur,
                self.trade_count / max(1, self.max_trades),
                self.position_hours,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([row, account], axis=0).astype(np.float32)
        _log.debug(
            "%s [AI-ENGINE][DEBUG] _build_observation completed. Obs features: %s (Expected: 22)",
            datetime.now().astimezone().isoformat(),
            obs.shape[0],
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._build_observation()
        info = {"equity_eur": self._equity(), "trade_count": self.trade_count}
        return obs, info

    def step(self, action: int):
        if len(self.df) <= 0:
            z = np.zeros(self.observation_space.shape, dtype=np.float32)
            return z, 0.0, True, False, {"equity_eur": 0.0, "trade_count": self.trade_count, "empty_frame": True}
        if self.step_idx >= len(self.df):
            z = np.zeros(self.observation_space.shape, dtype=np.float32)
            return z, 0.0, True, False, {"equity_eur": self._equity(), "trade_count": self.trade_count, "oob_step": True}

        price = self._current_price()
        fee_rate = self.fee_bps / 10000.0
        executed_trade = False
        forced_stop_loss = False
        # Rendement t.o.v. entry op het moment van sluiten (alleen gezet vóór _close_position wist entry_price).
        closed_trade_return_frac: float | None = None

        # Execution rules (RL-actie eerst; daarna harde stop-loss zoals core.risk_manager)
        futile_sell = action == 2 and self.position_btc <= 1e-12
        if action == 1 and self.balance_eur > 10:
            notional = min(self.max_trade_notional_eur, self.balance_eur)
            fee = notional * fee_rate
            qty = max(0.0, (notional - fee) / price)
            self.balance_eur -= notional
            self.position_btc += qty
            self.entry_price = price
            self.position_peak_price = price
            self.position_hours = 0.0
            self.trade_count += 1
            executed_trade = True
        elif action == 2 and self.position_btc > 0:
            ep = float(self.entry_price)
            if ep > 1e-12:
                closed_trade_return_frac = (float(price) - ep) / ep
            self._close_position(price, fee_rate)
            executed_trade = True

        if self.position_btc > 1e-12:
            self.position_peak_price = max(self.position_peak_price, price)

        if self.position_btc > 1e-12 and self.position_peak_price > 1e-12:
            # Dynamische Trailing Stop via 'range_pct' (True Range / ATR proxy)
            ri = self._safe_row_idx()
            raw_range_pct = float(self.df.loc[ri, "range_pct"]) * 100.0 if "range_pct" in self.df.columns else 0.0
            dynamic_sl_pct = max(self._risk_cfg.stop_loss_pct, raw_range_pct * 2.5)
            
            sl_mult = 1.0 - (dynamic_sl_pct / 100.0)
            if price <= self.position_peak_price * sl_mult:
                ep = float(self.entry_price)
                if ep > 1e-12:
                    closed_trade_return_frac = (float(price) - ep) / ep
                self._close_position(price, fee_rate)
                executed_trade = True
                forced_stop_loss = True

        if self.position_btc > 1e-12:
            self.position_hours += self.interval_hours

        equity = self._equity()
        peak_for_dd = max(self.equity_peak, equity)
        drawdown = 0.0 if peak_for_dd <= 0 else (peak_for_dd - equity) / peak_for_dd
        ri = self._safe_row_idx()
        whale_pressure = float(self.df.loc[ri, "whale_pressure"]) if "whale_pressure" in self.df.columns else 0.0
        sentiment_score = float(self.df.loc[ri, "sentiment_score"]) if "sentiment_score" in self.df.columns else 0.0
        news_conf = float(self.df.loc[ri, "news_confidence"]) if "news_confidence" in self.df.columns else 0.0

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
            position_hours=self.position_hours,
            closed_trade_return_frac=closed_trade_return_frac,
        )
        # Kleine straf voor SELL zonder positie: model leert dat dit zinloos is
        if futile_sell:
            reward -= 0.01
        # Internal shaping: reward alignment with strong news/whale impulses.
        aligned_signal = False
        strong_bull = (whale_pressure >= 0.6) or (sentiment_score >= 0.6 and news_conf >= 0.2)
        strong_bear = (whale_pressure <= -0.6) or (sentiment_score <= -0.6 and news_conf >= 0.2)
        if executed_trade and int(action) == 1 and strong_bull:
            reward += 0.02
            aligned_signal = True
        elif executed_trade and int(action) == 2 and strong_bear:
            reward += 0.02
            aligned_signal = True
        # Clip per-step reward to prevent value function divergence over 50k steps.
        reward = float(np.clip(reward, -2.0, 2.0))

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
            "position_hours": round(self.position_hours, 2),
            "drawdown": round(drawdown, 6),
            "trade_count": self.trade_count,
            "pnl_eur_total": round(equity - self.initial_balance_eur, 2),
            "forced_stop_loss": forced_stop_loss,
            "signal_alignment_reward": aligned_signal,
        }
        return obs, reward, terminated, truncated, info
