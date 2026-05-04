"""
Bestand: app/rl/env.py
Relatief pad: ./app/rl/env.py
Functie: Gymnasium trading omgeving; beloning via core.reward_function (PnL-%, drawdown, friction, consistency, SL-shock).
"""

from __future__ import annotations

from datetime import datetime
from app.datetime_util import UTC
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from core.preprocessor import attention_gate_weights
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
        
        # ALIGNMENT CHECK: We verwachten exact 16 marktsignalen + 5 account variabelen = 21.
        assert obs_shape == 21, f"Observation space mismatch! Verwacht 21 features, kreeg {obs_shape}."

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32,
        )

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
        return float(self.df.loc[self.step_idx, "close"])

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
        row = self.df.loc[self.step_idx, self.feature_cols].astype(float).to_numpy(dtype=np.float32)
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
        print(f"{datetime.now().astimezone().isoformat()} [AI-ENGINE][INFO] _build_observation completed. Obs features: {obs.shape[0]} (Expected: 21)")
        return obs

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
            self.position_peak_price = price
            self.position_hours = 0.0
            self.trade_count += 1
            executed_trade = True
        elif action == 2 and self.position_btc > 0:
            self._close_position(price, fee_rate)
            executed_trade = True

        if self.position_btc > 1e-12:
            self.position_peak_price = max(self.position_peak_price, price)

        if self.position_btc > 1e-12 and self.position_peak_price > 1e-12:
            # Dynamische Trailing Stop via 'range_pct' (True Range / ATR proxy)
            raw_range_pct = float(self.df.loc[self.step_idx, "range_pct"]) * 100.0 if "range_pct" in self.df.columns else 0.0
            dynamic_sl_pct = max(self._risk_cfg.stop_loss_pct, raw_range_pct * 2.5)
            
            sl_mult = 1.0 - (dynamic_sl_pct / 100.0)
            if price <= self.position_peak_price * sl_mult:
                self._close_position(price, fee_rate)
                executed_trade = True
                forced_stop_loss = True

        if self.position_btc > 1e-12:
            self.position_hours += self.interval_hours

        equity = self._equity()
        peak_for_dd = max(self.equity_peak, equity)
        drawdown = 0.0 if peak_for_dd <= 0 else (peak_for_dd - equity) / peak_for_dd
        whale_pressure = float(self.df.loc[self.step_idx, "whale_pressure"]) if self.step_idx < len(self.df) else 0.0
        sentiment_score = float(self.df.loc[self.step_idx, "sentiment_score"]) if self.step_idx < len(self.df) else 0.0
        news_conf = float(self.df.loc[self.step_idx, "news_confidence"]) if self.step_idx < len(self.df) else 0.0

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
        )
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
