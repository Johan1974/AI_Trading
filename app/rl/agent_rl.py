"""
Bestand: app/rl/agent_rl.py
Relatief pad: ./app/rl/agent_rl.py
Functie: PPO-agent service met pre-train, realtime beslissing, reasoning en trainingsmonitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.datetime_util import UTC
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import os

from core.preprocessor import attention_gate_weights
from core.social_engine import apply_whale_attention_blend
from app.rl.data import build_rl_training_frame, fetch_bitvavo_historical_candles
from app.rl.env import BitvavoTradingEnv
from app.services import rl_metrics_store

MIN_LEARNING_RATE = 1.0e-5
MIN_EXPLORATION_EPS = 0.05


class RewardLossCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.cumulative_rewards: list[float] = []
        self.loss_series: list[float] = []
        self.episode_lengths: list[float] = []
        self.policy_entropy: list[float] = []
        self.approx_kl: list[float] = []
        self.value_loss: list[float] = []
        self.global_steps: list[int] = []
        self._cum = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None and len(rewards):
            self._cum += float(np.mean(rewards))
            self.cumulative_rewards.append(round(self._cum, 6))
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if bool(done) and idx < len(infos):
                    trade_count = infos[idx].get("trade_count")
                    if trade_count is not None:
                        self.episode_lengths.append(float(trade_count))
        try:
            logs = self.model.logger.name_to_value
            loss = logs.get("train/loss")
            if loss is not None:
                self.loss_series.append(float(loss))
            entropy_loss = logs.get("train/entropy_loss")
            if entropy_loss is not None:
                # SB3 logs entropy as negative loss component.
                self.policy_entropy.append(float(max(0.0, -float(entropy_loss))))
            kl = logs.get("train/approx_kl")
            if kl is not None:
                self.approx_kl.append(float(kl))
            val_loss = logs.get("train/value_loss")
            if val_loss is not None:
                self.value_loss.append(float(val_loss))
        except Exception:
            pass
        self.global_steps.append(int(self.num_timesteps))
        return True


@dataclass
class RLDecision:
    action: int
    action_name: str
    confidence: float
    expected_reward_pct: float
    feature_weights: dict[str, float]
    reasoning: str


class RLAgentService:
    def __init__(self, model_dir: str = "artifacts/rl") -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: PPO | None = None
        self.active_pair: str | None = None
        self.last_training_progress: dict[str, list[float]] = {"reward": [], "loss": []}
        self.last_training_stats: dict[str, float | int] = {
            "learning_rate": MIN_LEARNING_RATE,
            "global_step_count": 0,
            "exploration_rate_pct": 100.0,
            "exploration_final_eps": max(
                MIN_EXPLORATION_EPS,
                float(os.getenv("RL_EXPLORATION_FINAL_EPS", "0.05") or 0.05),
            ),
        }
        self._paper_sl_hits: int = 0
        self.last_network_logs: dict[str, list[float]] = {"approx_kl": [], "value_loss": []}
        self.last_benchmark: dict[str, float] = {"rl_pnl_pct": 0.0, "buy_hold_pnl_pct": 0.0, "alpha_pct": 0.0}
        self.last_correlation: dict[str, float] = {
            "sentiment_price_correlation": 0.0,
            "news_weight": 0.0,
            "price_weight": 0.0,
        }
        self.last_decision: RLDecision | None = None
        self.feature_names = [
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
        self.model_top_k = 5

    def _chart_series_cap(self) -> int:
        return max(200, min(int(os.getenv("RL_TRAINING_CHART_MAX_POINTS", "8000") or 8000), 50000))

    def _exploration_decay_per_1k_pct(self) -> float:
        return max(0.0, float(os.getenv("RL_EPS_DECAY_PCT_PER_1K_STEPS", "0.5") or 0.5))

    def _exploration_live_from_steps(self, global_steps: int) -> float:
        floor = max(MIN_EXPLORATION_EPS, float(os.getenv("RL_EXPLORATION_MIN_EPS", "0.05") or 0.05))
        steps = max(0, int(global_steps))
        decay_pct = self._exploration_decay_per_1k_pct() * (float(steps) / 1000.0)
        eps_pct = max(floor * 100.0, 100.0 - decay_pct)
        return max(floor, min(1.0, eps_pct / 100.0))

    def _persist_training_from_callback(self, pair: str, cb: RewardLossCallback) -> None:
        try:
            gs = int(cb.global_steps[-1]) if cb.global_steps else int(self.last_training_stats.get("global_step_count", 0))
            rl_metrics_store.append_training_chunk(
                pair=pair,
                rewards=list(cb.cumulative_rewards),
                loss=list(cb.loss_series),
                value_loss=list(cb.value_loss),
                policy_entropy=list(cb.policy_entropy),
                episode_length=list(cb.episode_lengths),
                global_step=gs,
            )
        except Exception:
            pass

    def save_hourly_checkpoint(self, pair: str) -> None:
        """Schrijft model-weights weg (PPO .zip) zonder de reward-ranglijst te verstoren."""
        if self.model is None:
            return
        pair = pair.upper()
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self._model_path(pair, timestamp=f"hourly_{ts}")
        try:
            self.model.save(str(out))
            print(f"[RL-CHECKPOINT] Hourly weights saved: {out}.zip")
        except Exception as exc:
            print(f"[RL-CHECKPOINT] Save failed: {exc}")

    def online_update(
        self,
        pair: str,
        lookback_days: int = 30,
        total_timesteps: int = 3000,
        cmc_metrics: dict[str, Any] | None = None,
    ) -> None:
        pair = pair.upper()
        if self.model is None or self.active_pair != pair:
            self.ensure_pretrained(pair=pair, lookback_days=max(90, lookback_days))
        if self.model is None:
            return
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=max(7, lookback_days))
        candles = fetch_bitvavo_historical_candles(
            market=pair,
            interval="1h",
            start_dt=start_dt,
            end_dt=end_dt,
        )
        frame = build_rl_training_frame(
            candles_df=candles,
            market=pair,
            news_query="crypto",
            news_api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cmc_metrics=cmc_metrics or {"btc_dominance_pct": float(os.getenv("CMC_BTC_DOMINANCE_FALLBACK", "0") or 0.0)},
        )
        vec_env = DummyVecEnv([lambda: BitvavoTradingEnv(data=frame, max_trades=10000)])
        self.model.set_env(vec_env)
        cb = RewardLossCallback()
        self.model.learn(
            total_timesteps=int(max(512, total_timesteps)),
            callback=cb,
            progress_bar=False,
            reset_num_timesteps=False,
        )
        cap = self._chart_series_cap()
        self.last_training_progress = {
            "reward": (self.last_training_progress.get("reward", []) + cb.cumulative_rewards)[-cap:],
            "loss": (self.last_training_progress.get("loss", []) + cb.loss_series)[-cap:],
            "episode_length": (self.last_training_progress.get("episode_length", []) + cb.episode_lengths)[-cap:],
            "policy_entropy": (self.last_training_progress.get("policy_entropy", []) + cb.policy_entropy)[-cap:],
        }
        self.last_network_logs = {
            "approx_kl": (self.last_network_logs.get("approx_kl", []) + cb.approx_kl)[-cap:],
            "value_loss": (self.last_network_logs.get("value_loss", []) + cb.value_loss)[-cap:],
        }
        self._persist_training_from_callback(pair, cb)
        self.last_training_stats["global_step_count"] = int(
            cb.global_steps[-1] if cb.global_steps else self.last_training_stats.get("global_step_count", 0)
        )
        self.last_benchmark = self._benchmark_vs_buy_hold(model=self.model, frame=frame)
        if self.model is not None:
            self.last_training_stats["learning_rate"] = round(
                max(MIN_LEARNING_RATE, float(self.model.lr_schedule(1.0))),
                8,
            )
        gs_now = int(self.last_training_stats.get("global_step_count", 0) or 0)
        eps_now = self._exploration_live_from_steps(gs_now)
        os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{eps_now:.4f}"
        self.last_training_stats["exploration_final_eps"] = round(
            eps_now, 4
        )

    def ingest_paper_stop_loss(self) -> None:
        """Harde stop-loss vanuit risk-engine (paper): één 'geheugen'-puls voor volgende decide()."""
        self._paper_sl_hits = min(20, self._paper_sl_hits + 1)

    def _model_prefix(self, pair: str) -> str:
        safe = pair.replace("/", "-").replace(":", "-")
        return f"ppo_{safe}"

    def _model_path(self, pair: str, timestamp: str | None = None) -> Path:
        prefix = self._model_prefix(pair)
        if timestamp:
            return self.model_dir / f"{prefix}_{timestamp}"
        return self.model_dir / prefix

    def _metadata_path(self, pair: str) -> Path:
        return self.model_dir / f"{self._model_prefix(pair)}_models.json"

    def _load_model_registry(self, pair: str) -> list[dict[str, Any]]:
        path = self._metadata_path(pair)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [x for x in payload if isinstance(x, dict)]

    def _write_model_registry(self, pair: str, rows: list[dict[str, Any]]) -> None:
        self._metadata_path(pair).write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    def _register_model_version(self, pair: str, timestamp: str, reward_score: float) -> None:
        rows = self._load_model_registry(pair)
        rows.append(
            {
                "timestamp": timestamp,
                "reward_score": float(reward_score),
                "model_path": f"{self._model_path(pair, timestamp=timestamp)}.zip",
            }
        )
        rows = sorted(rows, key=lambda x: float(x.get("reward_score", -1e18)), reverse=True)
        keep = rows[: self.model_top_k]
        drop = rows[self.model_top_k :]
        for item in drop:
            model_path = Path(str(item.get("model_path", "")))
            if model_path.exists():
                model_path.unlink()
        self._write_model_registry(pair, keep)

    def _load_if_available(self, pair: str) -> bool:
        rows = self._load_model_registry(pair)
        if rows:
            best = sorted(rows, key=lambda x: float(x.get("reward_score", -1e18)), reverse=True)[0]
            best_model_path = Path(str(best.get("model_path", "")))
            if best_model_path.exists():
                self.model = PPO.load(str(best_model_path.with_suffix("")))
                self.active_pair = pair
                return True
        path = self._model_path(pair)
        zip_path = Path(f"{path}.zip")
        if zip_path.exists():
            self.model = PPO.load(str(path))
            self.active_pair = pair
            return True
        return False

    def ensure_pretrained(
        self,
        pair: str,
        lookback_days: int = 400,
        total_timesteps: int = 5000,
    ) -> None:
        pair = pair.upper()
        if self.active_pair == pair and self.model is not None:
            return
        if self._load_if_available(pair):
            return

        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=max(90, lookback_days))
        candles = fetch_bitvavo_historical_candles(
            market=pair,
            interval="1h",
            start_dt=start_dt,
            end_dt=end_dt,
        )
        frame = build_rl_training_frame(
            candles_df=candles,
            market=pair,
            news_query="crypto",
            news_api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cmc_metrics={"btc_dominance_pct": float(os.getenv("CMC_BTC_DOMINANCE_FALLBACK", "0") or 0.0)},
        )

        def _make_env():
            return BitvavoTradingEnv(data=frame, max_trades=10000)

        vec_env = DummyVecEnv([_make_env])
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
        cb = RewardLossCallback()
        model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self._model_path(pair, timestamp=timestamp)
        model.save(str(out))
        self.model = model
        self.active_pair = pair
        cap = self._chart_series_cap()
        self.last_training_progress = {
            "reward": cb.cumulative_rewards[-cap:],
            "loss": cb.loss_series[-cap:],
            "episode_length": cb.episode_lengths[-cap:],
            "policy_entropy": cb.policy_entropy[-cap:],
        }
        self.last_network_logs = {
            "approx_kl": cb.approx_kl[-cap:],
            "value_loss": cb.value_loss[-cap:],
        }
        self._persist_training_from_callback(pair, cb)
        current_lr = max(MIN_LEARNING_RATE, float(model.lr_schedule(1.0)))
        latest_entropy = float(cb.policy_entropy[-1]) if cb.policy_entropy else 0.0
        max_entropy = float(np.log(3.0))
        exploration = MIN_EXPLORATION_EPS if max_entropy <= 1e-9 else min(
            1.0,
            max(MIN_EXPLORATION_EPS, latest_entropy / max_entropy),
        )
        gs_now = int(cb.global_steps[-1] if cb.global_steps else total_timesteps)
        eps_floor = self._exploration_live_from_steps(gs_now)
        os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{eps_floor:.4f}"
        self.last_training_stats = {
            "learning_rate": round(current_lr, 8),
            "global_step_count": gs_now,
            "exploration_rate_pct": round(eps_floor * 100.0, 2),
            "exploration_final_eps": round(eps_floor, 4),
        }
        self.last_benchmark = self._benchmark_vs_buy_hold(model=model, frame=frame)
        self.last_correlation = self._sentiment_price_correlation(frame=frame)
        reward_score = float(cb.cumulative_rewards[-1]) if cb.cumulative_rewards else 0.0
        self._register_model_version(pair=pair, timestamp=timestamp, reward_score=reward_score)

    def _benchmark_vs_buy_hold(self, model: PPO, frame) -> dict[str, float]:
        env = BitvavoTradingEnv(data=frame, max_trades=10000)
        obs, _ = env.reset()
        done = False
        truncated = False
        last_info = {"equity_eur": 10000.0}
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(int(action))
            last_info = info
        rl_final_equity = float(last_info.get("equity_eur") or 10000.0)
        rl_pnl_pct = ((rl_final_equity - 10000.0) / 10000.0) * 100.0

        close0 = float(frame["close"].iloc[0])
        closeN = float(frame["close"].iloc[-1])
        buy_hold_pnl_pct = 0.0 if close0 <= 0 else ((closeN - close0) / close0) * 100.0
        return {
            "rl_pnl_pct": round(rl_pnl_pct, 4),
            "buy_hold_pnl_pct": round(buy_hold_pnl_pct, 4),
            "alpha_pct": round(rl_pnl_pct - buy_hold_pnl_pct, 4),
        }

    def _sentiment_price_correlation(self, frame) -> dict[str, float]:
        corr = 0.0
        if "sentiment_score" in frame.columns and "price_action" in frame.columns:
            valid = frame[["sentiment_score", "price_action"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 3:
                sent = valid["sentiment_score"].astype(float).to_numpy()
                price = valid["price_action"].astype(float).to_numpy()
                sent_std = float(np.std(sent))
                price_std = float(np.std(price))
                # Guard against near-constant series to avoid divide-by-zero warnings.
                if sent_std > 1e-12 and price_std > 1e-12:
                    sent_centered = sent - float(np.mean(sent))
                    price_centered = price - float(np.mean(price))
                    denom = sent_std * price_std
                    if denom > 1e-12:
                        corr = float(np.mean(sent_centered * price_centered) / denom)
                        if not np.isfinite(corr):
                            corr = 0.0
        news_weight = 0.0
        price_weight = 0.0
        if self.last_decision and isinstance(self.last_decision.feature_weights, dict):
            fw = self.last_decision.feature_weights
            news_weight = (
                float(fw.get("sentiment_score", 0.0))
                + float(fw.get("news_confidence", 0.0))
                + float(fw.get("social_volume", 0.0))
                + float(fw.get("fear_greed_score", 0.0))
                + float(fw.get("btc_dominance_pct", 0.0))
                + float(fw.get("whale_pressure", 0.0))
                + float(fw.get("macro_volatility_window", 0.0))
            )
            price_weight = (
                float(fw.get("price_action", 0.0))
                + float(fw.get("volatility_24", 0.0))
                + float(fw.get("volume_change", 0.0))
                + float(fw.get("bollinger_width", 0.0))
                + float(fw.get("bollinger_position", 0.0))
                + float(fw.get("orderbook_imbalance", 0.0))
                + float(fw.get("macd", 0.0))
                + float(fw.get("rsi_14", 0.0))
                + float(fw.get("ema_gap_pct", 0.0))
            )
        return {
            "sentiment_price_correlation": round(corr, 4),
            "news_weight": round(news_weight, 4),
            "price_weight": round(price_weight, 4),
        }

    def _action_to_name(self, action: int) -> str:
        return {0: "HOLD", 1: "BUY", 2: "SELL"}.get(int(action), "HOLD")

    def _extract_feature_weighting(self) -> np.ndarray:
        if self.model is None:
            return np.ones(len(self.feature_names), dtype=float)
        with torch.no_grad():
            first_layer = self.model.policy.mlp_extractor.policy_net[0]
            w = first_layer.weight.detach().cpu().numpy()
        # Observation includes 4 account features at the end; keep first N state features.
        base = np.mean(np.abs(w[:, : len(self.feature_names)]), axis=0)
        if np.sum(base) <= 1e-12:
            return np.ones(len(self.feature_names), dtype=float)
        return base / np.sum(base)

    def decide(
        self,
        latest_row: dict[str, float],
        account: dict[str, float] | None = None,
        trade_context: dict[str, Any] | None = None,
    ) -> RLDecision:
        if self.model is None:
            raise RuntimeError("RL model not initialized.")
        tc = trade_context if isinstance(trade_context, dict) else {}
        acct = account or {
            "balance_ratio": 1.0,
            "position": 0.0,
            "pnl_ratio": 0.0,
            "trade_ratio": 0.0,
        }
        obs_features = np.array([float(latest_row.get(k, 0.0)) for k in self.feature_names], dtype=np.float32)
        if not np.all(np.isfinite(obs_features)):
            print("WARNING: RL decide() state-features bevatten NaN/Inf; invoer wordt gesaneerd.")
            obs_features = np.nan_to_num(obs_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        gate = attention_gate_weights(obs_features, temperature=0.7)
        gate = apply_whale_attention_blend(gate, list(self.feature_names))
        obs_features = (obs_features * gate).astype(np.float32)
        obs = np.concatenate(
            [
                obs_features,
                np.array(
                    [
                        float(acct.get("balance_ratio", 1.0)),
                        float(acct.get("position", 0.0)),
                        float(acct.get("pnl_ratio", 0.0)),
                        float(acct.get("trade_ratio", 0.0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        gs_live = int(self.last_training_stats.get("global_step_count", 0) or 0)
        eps_live = self._exploration_live_from_steps(gs_live)
        eps_live = max(MIN_EXPLORATION_EPS, min(1.0, eps_live))
        explore_roll = float(np.random.random())
        explored = explore_roll < eps_live
        greedy = int(np.argmax(probs))
        if explored:
            action = int(np.random.randint(0, 3))
        else:
            action = greedy
        confidence = float(probs[int(action)])

        whale_bias = str(tc.get("whale_bias") or "neutral")
        whale_st = float(tc.get("whale_strength", 0.0) or 0.0)
        price_push = float(latest_row.get("price_action", 0.0) or 0.0)
        conflict_mult = float(os.getenv("WHALE_TECH_CONFLICT_CONF_MULT", "0.82") or 0.82)
        whale_damp_note = ""
        if whale_bias == "inflow" and whale_st >= float(os.getenv("WHALE_CONFLICT_MIN_STRENGTH", "0.28") or 0.28):
            if int(action) == 1 and price_push > float(os.getenv("WHALE_TECH_PRICE_ACTION_THRESH", "0.12") or 0.12):
                confidence *= conflict_mult
                whale_damp_note = " Whale inflow vs bullish tape → confidence reduced."
        if whale_bias == "outflow" and whale_st >= float(os.getenv("WHALE_OUTFLOW_CONFIRM_MIN_STRENGTH", "0.28") or 0.28):
            if int(action) == 1:
                confidence = min(1.0, confidence * float(os.getenv("WHALE_OUTFLOW_BUY_CONF_BOOST", "1.05") or 1.05))

        min_trade_conf = float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0") or 0.0)
        min_trade_conf = max(0.0, min(1.0, min_trade_conf))
        gated_note = ""
        if min_trade_conf > 0 and int(action) in (1, 2) and confidence < min_trade_conf:
            gated_note = (
                f"Confidence {confidence:.2f} < actiedrempel {min_trade_conf:.2f} → HOLD i.p.v. "
                f"{self._action_to_name(int(action))}. "
            )
            action = 0
            confidence = float(probs[int(action)])

        risk_prefix = ""
        if self._paper_sl_hits > 0:
            risk_prefix = (
                "Na harde stop-loss (risk-engine, zelfde drempel als RL-simulatie): dit pad heeft een zware "
                "negatieve beloning in training — positie en risico voorzichtig afwegen. "
            )
            self._paper_sl_hits -= 1

        base_w = self._extract_feature_weighting()
        contrib = np.abs(obs_features) * base_w * gate
        denom = np.sum(contrib) if float(np.sum(contrib)) > 1e-12 else 1.0
        norm = contrib / denom
        feature_weights = {
            name: round(float(weight), 4) for name, weight in zip(self.feature_names, norm.tolist())
        }
        top = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:2]
        expected_reward_pct = float((probs[1] - probs[2]) * 1.5)
        explore_note = ""
        if explored and eps_live > 0:
            explore_note = f"(Exploratie ε={eps_live:g}: random actie i.p.v. greedy {self._action_to_name(greedy)}.) "
        reasoning = (
            f"{risk_prefix}"
            f"{gated_note}"
            f"Besluit: {self._action_to_name(action)}. Reden: {top[0][0]} ({top[0][1]:.2f}) "
            f"en {top[1][0]} ({top[1][1]:.2f}) sturen de policy. "
            f"{explore_note}"
            f"{whale_damp_note}"
            f"Verwachte beloning: {expected_reward_pct:+.2f}%."
        )
        decision = RLDecision(
            action=action,
            action_name=self._action_to_name(action),
            confidence=round(confidence, 4),
            expected_reward_pct=round(expected_reward_pct, 4),
            feature_weights=feature_weights,
            reasoning=reasoning,
        )
        self.last_decision = decision
        # Keep dynamic weighting visible in AI Brain.
        self.last_correlation["news_weight"] = round(
            float(feature_weights.get("sentiment_score", 0.0))
            + float(feature_weights.get("news_confidence", 0.0))
            + float(feature_weights.get("social_volume", 0.0))
            + float(feature_weights.get("fear_greed_score", 0.0))
            + float(feature_weights.get("btc_dominance_pct", 0.0))
            + float(feature_weights.get("whale_pressure", 0.0))
            + float(feature_weights.get("macro_volatility_window", 0.0)),
            4,
        )
        self.last_correlation["price_weight"] = round(
            float(feature_weights.get("price_action", 0.0))
            + float(feature_weights.get("volatility_24", 0.0))
            + float(feature_weights.get("volume_change", 0.0))
            + float(feature_weights.get("bollinger_width", 0.0))
            + float(feature_weights.get("bollinger_position", 0.0))
            + float(feature_weights.get("orderbook_imbalance", 0.0))
            + float(feature_weights.get("macd", 0.0))
            + float(feature_weights.get("rsi_14", 0.0))
            + float(feature_weights.get("ema_gap_pct", 0.0)),
            4,
        )
        return decision

    @staticmethod
    def _normalize_reward_series_for_ui(values: list[Any]) -> list[float]:
        """Cumulatieve training-reward schalen naar [-1, 1] voor stabiele charts (UI + WebSocket)."""
        if not values:
            return []
        arr = np.asarray([float(v) for v in values], dtype=np.float64)
        m = float(np.max(np.abs(arr)))
        if not np.isfinite(m) or m < 1e-12:
            return [0.0] * len(values)
        clipped = np.clip(arr / m, -1.0, 1.0)
        return [float(x) for x in clipped.tolist()]

    def training_monitor(self) -> dict[str, Any]:
        raw_r = self.last_training_progress.get("reward", [])
        r_list = [float(x) for x in raw_r] if isinstance(raw_r, list) else []
        return {
            **self.last_training_progress,
            "reward_normalized": self._normalize_reward_series_for_ui(r_list),
            "stats": self.last_training_stats,
            "network_logs": self.last_network_logs,
            "benchmark": self.last_benchmark,
            "correlation": self.last_correlation,
        }
