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

import logging
import zlib
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
import torch

# MEMORY FIX: Beperk PyTorch CPU-threads om RAM reserveringen te minimaliseren
torch.set_num_threads(2)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import gc
import os
import time as _time_mod

from core.preprocessor import attention_gate_weights
from core.social_engine import apply_whale_attention_blend
from app.rl.data import build_rl_training_frame, fetch_bitvavo_historical_candles
from app.rl.env import BitvavoTradingEnv
from app.services import rl_metrics_store

MIN_LEARNING_RATE = 1.0e-5
MIN_EXPLORATION_EPS = 0.05

_log_rl_agent = logging.getLogger(__name__)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric coercion for RL/runtime payloads."""
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return float(default)
            raw = raw.replace("%", "").replace(",", ".")
            value = raw
        out = float(value)
        if not np.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def get_rl_ppo_device() -> str:
    """
    SB3 MlpPolicy is sneller op CPU dan op een losse GPU-kernel; zet RL_PPO_DEVICE=cuda voor GPU-forcing.
    """
    raw = str(os.getenv("RL_PPO_DEVICE", "cpu") or "cpu").strip().lower()
    if raw.startswith("cuda") or raw in ("gpu", "auto"):
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


class RewardLossCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.cumulative_rewards: list[float] = []
        self.loss_series: list[float] = []
        self.episode_lengths: list[float] = []
        self.policy_entropy: list[float] = []
        self.approx_kl: list[float] = []
        self.value_loss: list[float] = []
        self.td_steps: list[int] = []
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
                self.td_steps.append(int(self.num_timesteps))
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
    # PPO categorical: index 0=HOLD, 1=BUY, 2=SELL (zelfde volgorde als ``probs``-vector).
    prob_hold: float = 0.0
    prob_buy: float = 0.0
    prob_sell: float = 0.0
    thinking_steps: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.thinking_steps is None:
            self.thinking_steps = []


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    x = np.clip(x, -80.0, 80.0)
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 1e-18:
        return np.ones(min(3, len(x)), dtype=np.float64) / float(max(1, min(3, len(x))))
    return e / s


class RLAgentService:
    @staticmethod
    def _ppo_learn_timesteps_aligned(model: PPO, total_timesteps: int) -> int:
        """
        Ceil naar veelvoud van ``n_steps`` (zie ``core.worker_execution.align_ppo_total_timesteps``).
        """
        from core.worker_execution import align_ppo_total_timesteps

        return align_ppo_total_timesteps(model, total_timesteps)

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
            "batch_size": 128,
            "n_steps": 1024,
            "discount_factor": 0.99,
            "is_training_active": False,
        }
        self._paper_sl_hits: int = 0
        self._last_canonical_save_monotonic: dict[str, float] = {}
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
            "bid_ask_spread",
            "macd",
            "rsi_14",
            "ema_gap_pct",
        ]
        self.model_top_k = 5

    def _chart_series_cap(self) -> int:
        return max(200, min(int(os.getenv("RL_TRAINING_CHART_MAX_POINTS", "8000") or 8000), 50000))

    def _sync_training_stats_from_model(self) -> None:
        """Vul UI/Brain-tab stats (batch, γ, n_steps, training-flag) vanuit het actieve SB3-model."""
        m = self.model
        if m is None:
            return
        try:
            self.last_training_stats["batch_size"] = int(getattr(m, "batch_size", 128) or 128)
            self.last_training_stats["n_steps"] = int(getattr(m, "n_steps", 1024) or 1024)
            self.last_training_stats["discount_factor"] = float(getattr(m, "gamma", 0.99) or 0.99)
            self.last_training_stats["is_training_active"] = bool(
                str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on")
            )
        except Exception:
            pass

    def _exploration_decay_per_1k_pct(self) -> float:
        return max(0.0, float(os.getenv("RL_EPS_DECAY_PCT_PER_1K_STEPS", "0.5") or 0.5))

    def _exploration_live_from_steps(self, global_steps: int) -> float:
        floor = max(MIN_EXPLORATION_EPS, float(os.getenv("RL_EXPLORATION_MIN_EPS", "0.05") or 0.05))
        steps = max(0, int(global_steps))
        decay_pct = self._exploration_decay_per_1k_pct() * (float(steps) / 1000.0)
        eps_pct = max(floor * 100.0, 100.0 - decay_pct)
        return max(floor, min(1.0, eps_pct / 100.0))

    def _sync_exploration_stats_from_model(self) -> int:
        """
        Zet global_step_count / exploration UI gelijk met SB3 ``num_timesteps``.

        Zonder sync bleef ``global_step_count`` op 0 na ``PPO.load`` → ε-decay dacht stappen=0
        → exploratie bleef 100% (UI: 'Exploratie ε=1').
        """
        if self.model is None:
            return int(self.last_training_stats.get("global_step_count", 0) or 0)
        try:
            gs_model = int(getattr(self.model, "num_timesteps", 0) or 0)
        except Exception:
            gs_model = 0
        prev = int(self.last_training_stats.get("global_step_count", 0) or 0)
        gs = max(prev, gs_model)
        self.last_training_stats["global_step_count"] = gs
        eps = self._exploration_live_from_steps(gs)
        eps = max(MIN_EXPLORATION_EPS, min(1.0, eps))
        self.last_training_stats["exploration_rate_pct"] = round(eps * 100.0, 2)
        self.last_training_stats["exploration_final_eps"] = round(eps, 4)
        return gs

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
        """Schrijft model-weights weg (PPO .zip) en roteert oude hourly checkpoints om schijfruimte te besparen."""
        if self.model is None:
            return
        pair = pair.upper()
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self._model_path(pair, timestamp=f"hourly_{ts}")
        try:
            self.model.save(str(out))
            print(f"[RL-CHECKPOINT] Hourly weights saved: {out}.zip")
            
            # --- Schijfruimte Optimalisatie: Roteer hourly checkpoints ---
            prefix = f"{self._model_prefix(pair)}_hourly_"
            hourly_files = sorted(self.model_dir.glob(f"{prefix}*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Behoud maximaal de 3 meest recente hourly checkpoints per markt
            max_hourly_keep = 3
            if len(hourly_files) > max_hourly_keep:
                for old_file in hourly_files[max_hourly_keep:]:
                    try:
                        old_file.unlink()
                        print(f"[RL-CHECKPOINT] Oude hourly weights verwijderd ter optimalisatie: {old_file.name}")
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[RL-CHECKPOINT] Save failed: {exc}")

    def online_update(
        self,
        pair: str,
        lookback_days: int = 30,
        total_timesteps: int = 10000,
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
        del candles
        vec_env = DummyVecEnv([lambda: BitvavoTradingEnv(data=frame, max_trades=10000)])
        
        # SENIOR FIX: Voorkom DataFrame memory leaks door de oude environment netjes af te sluiten
        old_env = self.model.get_env()
        if old_env is not None:
            old_env.close()
            
        self.model.set_env(vec_env)
        cb = RewardLossCallback()
        raw_ts = int(max(512, total_timesteps))
        learn_ts = self._ppo_learn_timesteps_aligned(self.model, raw_ts)
        if learn_ts != raw_ts:
            print(f"[RL] online_update timesteps {raw_ts} -> {learn_ts} (align PPO n_steps={getattr(self.model, 'n_steps', '?')})")
        self.model.learn(
            total_timesteps=learn_ts,
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
            "td_steps": (self.last_network_logs.get("td_steps", []) + cb.td_steps)[-cap:],
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
        self._sync_training_stats_from_model()
        gs_now = int(self.last_training_stats.get("global_step_count", 0) or 0)
        eps_now = self._exploration_live_from_steps(gs_now)
        os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{eps_now:.4f}"
        self.last_training_stats["exploration_final_eps"] = round(
            eps_now, 4
        )
        
        # Release training env before frame deletion so the DataFrame can be GC'd
        try:
            if hasattr(self.model, 'env') and self.model.env is not None:
                self.model.env.close()
                self.model.env = None
        except Exception:
            pass
        del frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if str(os.getenv("RL_SAVE_CANONICAL_ZIP_AFTER_ONLINE_UPDATE", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            min_gap = float(os.getenv("RL_CANONICAL_SAVE_MIN_SEC", "300") or 300)
            now_m = _time_mod.monotonic()
            if now_m - float(self._last_canonical_save_monotonic.get(pair, 0.0)) >= max(30.0, min_gap):
                try:
                    self.model.save(str(self._model_path(pair)))
                    self._last_canonical_save_monotonic[pair] = now_m
                    print(f"[RL] Canonical weights opgeslagen na online_update: {self._model_path(pair)}.zip")
                except Exception as exc:
                    print(f"[RL] Canonical save na online_update mislukt: {exc}")

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
                try:
                    loaded_model = PPO.load(str(best_model_path.with_suffix("")), device=get_rl_ppo_device())
                    expected_shape = len(self.feature_names) + 5
                    if loaded_model.observation_space.shape[0] != expected_shape:
                        print(f"{datetime.now().astimezone().isoformat()} [AI-ENGINE][ERROR] Observation space mismatch! Model verwacht {loaded_model.observation_space.shape[0]} features, live agent verwacht {expected_shape}. Model {pair} wordt genegeerd en herbouwd.")
                    else:
                        if self.model is not None:
                            try:
                                _old_env = self.model.get_env()
                                if _old_env is not None:
                                    _old_env.close()
                            except Exception:
                                pass
                            del self.model
                            self.model = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        self.model = loaded_model
                        self.active_pair = pair
                        self._sync_exploration_stats_from_model()
                        return True
                except Exception as e:
                    print(f"{datetime.now().astimezone().isoformat()} [AI-ENGINE][ERROR] Fout bij laden best model {pair}: {e}")
        path = self._model_path(pair)
        zip_path = Path(f"{path}.zip")
        if zip_path.exists():
            try:
                loaded_model = PPO.load(str(path), device=get_rl_ppo_device())
                expected_shape = len(self.feature_names) + 5
                if loaded_model.observation_space.shape[0] != expected_shape:
                    print(f"{datetime.now().astimezone().isoformat()} [AI-ENGINE][ERROR] Observation space mismatch! Model verwacht {loaded_model.observation_space.shape[0]} features, live agent verwacht {expected_shape}. Model {pair} wordt genegeerd en herbouwd.")
                else:
                    if self.model is not None:
                        try:
                            _old_env = self.model.get_env()
                            if _old_env is not None:
                                _old_env.close()
                        except Exception:
                            pass
                        del self.model
                        self.model = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    self.model = loaded_model
                    self.active_pair = pair
                    self._sync_exploration_stats_from_model()
                    return True
            except Exception as e:
                print(f"{datetime.now().astimezone().isoformat()} [AI-ENGINE][ERROR] Fout bij laden fallback model {pair}: {e}")
        return False

    def ensure_pretrained(
        self,
        pair: str,
        lookback_days: int = 400,
        total_timesteps: int = 50000,
    ) -> None:
        pair = pair.upper()
        if self.active_pair == pair and self.model is not None:
            self._sync_exploration_stats_from_model()
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
        
        # Ruim oude model + env op om cyclische referenties en SB3 heap-groei te voorkomen
        _old_model = self.model
        if _old_model is not None:
            old_env = _old_model.get_env()
            if old_env is not None:
                old_env.close()
            self.model = None
            del _old_model

        def _make_env():
            return BitvavoTradingEnv(data=frame, max_trades=10000)

        vec_env = DummyVecEnv([_make_env])
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            vf_coef=0.75,
            device=get_rl_ppo_device(),
        )
        cb = RewardLossCallback()
        raw_ts = int(max(512, int(total_timesteps)))
        learn_ts = self._ppo_learn_timesteps_aligned(model, raw_ts)
        if learn_ts != raw_ts:
            print(f"[RL] ensure_pretrained timesteps {raw_ts} -> {learn_ts} (align PPO n_steps={getattr(model, 'n_steps', '?')})")
        model.learn(total_timesteps=learn_ts, callback=cb, progress_bar=False)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = self._model_path(pair, timestamp=timestamp)
        model.save(str(out))
        self.model = model
        self.active_pair = pair
        import gc as _gc; _gc.collect()
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
        gs_now = int(cb.global_steps[-1] if cb.global_steps else learn_ts)
        eps_floor = self._exploration_live_from_steps(gs_now)
        os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{eps_floor:.4f}"
        self.last_training_stats = {
            "learning_rate": round(current_lr, 8),
            "global_step_count": gs_now,
            "exploration_rate_pct": round(eps_floor * 100.0, 2),
            "exploration_final_eps": round(eps_floor, 4),
            "batch_size": int(getattr(model, "batch_size", 128) or 128),
            "n_steps": int(getattr(model, "n_steps", 1024) or 1024),
            "discount_factor": float(getattr(model, "gamma", 0.99) or 0.99),
            "is_training_active": bool(
                str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on")
            ),
        }
        self.last_benchmark = self._benchmark_vs_buy_hold(model=model, frame=frame)
        self.last_correlation = self._sentiment_price_correlation(frame=frame)
        reward_score = float(cb.cumulative_rewards[-1]) if cb.cumulative_rewards else 0.0
        self._register_model_version(pair=pair, timestamp=timestamp, reward_score=reward_score)
        
        # SENIOR FIX: Forceer GC en clear VRAM na zware pre-train initiatie
        import gc
        del frame
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _benchmark_vs_buy_hold(self, model: PPO, frame) -> dict[str, float]:
        env = BitvavoTradingEnv(data=frame, max_trades=10000)
        try:
            obs, _ = env.reset()
            done = False
            truncated = False
            last_info = {"equity_eur": 10000.0}
            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(int(action))
                last_info = info
        finally:
            env.close()
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
                + float(fw.get("bid_ask_spread", 0.0))
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
        with torch.inference_mode():
            first_layer = self.model.policy.mlp_extractor.policy_net[0]
            w = first_layer.weight.detach().cpu().numpy().astype(np.float32)
        # Observation includes 4 account features at the end; keep first N state features.
        base = np.mean(np.abs(w[:, : len(self.feature_names)]), axis=0)
        if np.sum(base) <= 1e-12:
            return np.ones(len(self.feature_names), dtype=float)
        base = base / np.sum(base)
        news_floor = float(os.getenv("RL_NEWS_FEATURE_FLOOR", "0.05"))
        news_features = {"sentiment_score", "news_confidence", "social_volume"}
        for i, name in enumerate(self.feature_names):
            if name in news_features and base[i] < news_floor:
                base[i] = np.float32(news_floor)
        base = base / np.sum(base)
        # Enforce minimum total news weight so Brain Lab shows >= RL_NEWS_WEIGHT_MIN.
        news_weight_min = float(os.getenv("RL_NEWS_WEIGHT_MIN", "0.0"))
        if news_weight_min > 0.0:
            news_idx = [i for i, n in enumerate(self.feature_names) if n in news_features]
            current = sum(float(base[i]) for i in news_idx)
            if 1e-12 < current < news_weight_min:
                scale = news_weight_min / current
                for i in news_idx:
                    base[i] = np.float32(base[i] * scale)
                base = base / np.sum(base)
        return base

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
            "position_hours": 0.0,
        }
        obs_features = np.array([_coerce_float(latest_row.get(k, 0.0), 0.0) for k in self.feature_names], dtype=np.float32)
        if not np.all(np.isfinite(obs_features)):
            print("WARNING: RL decide() state-features bevatten NaN/Inf; invoer wordt gesaneerd.")
            obs_features = np.nan_to_num(obs_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        _news_boost = float(os.getenv("RL_NEWS_SIGNAL_BOOST", "1.0"))
        if _news_boost > 1.0:
            _news_set = {"sentiment_score", "news_confidence", "social_volume"}
            for _i, _nm in enumerate(self.feature_names):
                if _nm in _news_set:
                    obs_features[_i] = np.float32(obs_features[_i] * _news_boost)
        gate = attention_gate_weights(obs_features, temperature=0.7)
        gate = apply_whale_attention_blend(gate, list(self.feature_names))
        gate = np.asarray(gate, dtype=np.float32)
        obs_features = (obs_features * gate).astype(np.float32)
        # Per-markt deterministische variatie vóór de policy:
        # - additieve + lichte multiplicatieve ruis op state (zuivere additieve ruis middelt vaak weg in laag 1);
        # - dezelfde ruis op het account-blok (vaak bijna gelijk tussen altcoins → identieke softmax).
        # Uitzetten: RL_INFER_OBS_MICRO_NOISE=0
        mk = str(tc.get("market") or tc.get("ticker") or "").strip().upper().replace("/", "-")
        _noise_raw = str(os.getenv("RL_INFER_OBS_MICRO_NOISE", "")).strip()
        if _noise_raw == "":
            _micro_amp = 0.006
        else:
            try:
                _micro_amp = float(_noise_raw)
            except ValueError:
                _micro_amp = 0.006
        seed = int(zlib.adler32(mk.encode("utf-8", errors="ignore")) & 0x7FFFFFFF) if mk else 0
        seed = seed or 1
        if mk and _micro_amp > 0.0:
            rng = np.random.RandomState(seed)
            mult = 1.0 + rng.uniform(-_micro_amp * 5.0, _micro_amp * 5.0, size=obs_features.shape).astype(np.float32)
            obs_features = (obs_features * mult).astype(np.float32)
            obs_features = (
                obs_features + rng.uniform(-_micro_amp, _micro_amp, size=obs_features.shape).astype(np.float32)
            ).astype(np.float32)
        acc_block = np.array(
            [
                _coerce_float(acct.get("balance_ratio", 1.0), 1.0),
                _coerce_float(acct.get("position", 0.0), 0.0),
                _coerce_float(acct.get("pnl_ratio", 0.0), 0.0),
                _coerce_float(acct.get("trade_ratio", 0.0), 0.0),
                _coerce_float(acct.get("position_hours", 0.0), 0.0),
            ],
            dtype=np.float32,
        )
        if mk and _micro_amp > 0.0:
            rng_a = np.random.RandomState((seed ^ 0xA5A5_A5A5) & 0x7FFFFFFF)
            acc_block = (
                acc_block + rng_a.uniform(-_micro_amp * 3.0, _micro_amp * 3.0, size=acc_block.shape).astype(np.float32)
            ).astype(np.float32)

        obs = np.concatenate([obs_features, acc_block], axis=0)
        
        if np.isnan(obs).any() or np.all(obs == 0.0):
            err_msg = f"Invalid observation space (NaN of all-zeros). Shape: {obs.shape}"
            print(f"{datetime.now(UTC).isoformat()} [DATA-INTEGRITY][CRITICAL] {err_msg}")
            raise ValueError(err_msg)
        else:
            print(f"{datetime.now(UTC).isoformat()} [DATA-INTEGRITY][OK] Valid obs shape {obs.shape} for decision.")

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy().reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        # Kleine logit-duw per markt: garandeert verschillende HOLD/BUY/SELL-verdeling als de PPO-output
        # voor meerdere paren op dezelfde macro-vector blijft hangen. Zelfde seed → stabiel per paar.
        # Uitzetten: RL_INFER_LOGIT_PAIR_NUDGE=0
        _nudge_raw = str(os.getenv("RL_INFER_LOGIT_PAIR_NUDGE", "")).strip()
        if _nudge_raw == "":
            _logit_nudge = 0.22
        else:
            try:
                _logit_nudge = float(_nudge_raw)
            except ValueError:
                _logit_nudge = 0.22
        if mk and _logit_nudge > 1e-12 and probs.size >= 3:
            try:
                dt_d = dist.distribution
                if hasattr(dt_d, "logits") and dt_d.logits is not None:
                    logits_np = dt_d.logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
                    if logits_np.size >= 3:
                        seed_ln = int(
                            zlib.adler32((mk + "|logit").encode("utf-8", errors="ignore")) & 0x7FFFFFFF
                        )
                        rng_ln = np.random.RandomState(seed_ln)
                        bias = rng_ln.normal(0.0, 1.0, size=3).astype(np.float64)
                        bias -= float(np.mean(bias))
                        nrm = float(np.linalg.norm(bias))
                        if nrm > 1e-12:
                            bias = (bias / nrm) * float(_logit_nudge)
                            probs = _softmax_np(logits_np[:3] + bias)
            except Exception:
                pass
        if probs.size < 3:
            _log_rl_agent.warning("PPO policy prob vector size=%s (expected 3) — pad with zeros", probs.size)
            pad = np.zeros(3, dtype=np.float64)
            pad[: min(3, probs.size)] = probs[:3]
            probs = pad
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        psum = float(np.sum(probs))
        fb_off = str(os.getenv("RL_UI_POLICY_FALLBACK", "1")).strip().lower() in ("0", "false", "no", "off")
        if psum < 1e-9:
            if not fb_off:
                _log_rl_agent.warning(
                    "PPO policy probs sum=%s — applying test distribution HOLD/BUY/SELL = 0.5/0.4/0.1 for UI pipeline check",
                    psum,
                )
                probs = np.array([0.5, 0.4, 0.1], dtype=np.float64)
            else:
                probs = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        else:
            probs = (probs / psum).astype(np.float64)
        prob_hold = float(probs[0])
        prob_buy = float(probs[1])
        prob_sell = float(probs[2])
        self._sync_exploration_stats_from_model()
        greedy = int(np.argmax(probs))
        greedy_name = self._action_to_name(greedy)
        greedy_mode = bool(tc.get("rl_greedy_inference"))
        if not greedy_mode:
            raw_g = str(os.getenv("RL_INFERENCE_GREEDY", "")).strip().lower()
            greedy_mode = raw_g in ("1", "true", "yes", "on", "live")

        gs_live = int(self.last_training_stats.get("global_step_count", 0) or 0)
        if greedy_mode:
            eps_live = 0.0
        else:
            eps_override = str(os.getenv("RL_EXPLORATION_INFERENCE_EPS", "")).strip()
            if eps_override:
                try:
                    eps_live = max(MIN_EXPLORATION_EPS, min(1.0, float(eps_override)))
                except (TypeError, ValueError):
                    eps_live = self._exploration_live_from_steps(gs_live)
                    eps_live = max(MIN_EXPLORATION_EPS, min(1.0, eps_live))
            else:
                eps_live = self._exploration_live_from_steps(gs_live)
                eps_live = max(MIN_EXPLORATION_EPS, min(1.0, eps_live))

        if greedy_mode:
            action = greedy
            explored = False
            confidence = float(probs[int(action)])
        else:
            explore_roll = float(np.random.random())
            explored = explore_roll < eps_live
            if explored:
                action = int(np.random.randint(0, 3))
            else:
                action = greedy
            confidence = float(probs[int(action)])
        _log_rl_agent.info(
            "Policy P(H/B/S)=%.2f%%/%.2f%%/%.2f%% greedy=%s | executed=%s explored=%s eps=%.4f steps=%s",
            prob_hold * 100.0,
            prob_buy * 100.0,
            prob_sell * 100.0,
            greedy_name,
            self._action_to_name(int(action)),
            explored,
            eps_live,
            gs_live,
        )

        whale_bias = str(tc.get("whale_bias") or "neutral")
        whale_st = _coerce_float(tc.get("whale_strength", 0.0), 0.0)
        price_push = _coerce_float(latest_row.get("price_action", 0.0), 0.0)
        conflict_mult = float(os.getenv("WHALE_TECH_CONFLICT_CONF_MULT", "0.82") or 0.82)
        whale_damp_note = ""
        if whale_bias == "inflow" and whale_st >= float(os.getenv("WHALE_CONFLICT_MIN_STRENGTH", "0.28") or 0.28):
            if int(action) == 1 and price_push > float(os.getenv("WHALE_TECH_PRICE_ACTION_THRESH", "0.12") or 0.12):
                confidence *= conflict_mult
                whale_damp_note = " Whale inflow vs bullish tape → confidence reduced."
        if whale_bias == "outflow" and whale_st >= float(os.getenv("WHALE_OUTFLOW_CONFIRM_MIN_STRENGTH", "0.28") or 0.28):
            if int(action) == 1:
                confidence = min(1.0, confidence * float(os.getenv("WHALE_OUTFLOW_BUY_CONF_BOOST", "1.05") or 1.05))

        from app.services.prediction_ui import trade_confidence_threshold_01

        min_trade_conf = trade_confidence_threshold_01()
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
        # Enforce RL_NEWS_WEIGHT_MIN on normalized contributions (Brain Lab display floor).
        _news_weight_min = float(os.getenv("RL_NEWS_WEIGHT_MIN", "0.0"))
        if _news_weight_min > 0.0:
            _news_set_contrib = {"sentiment_score", "news_confidence", "social_volume"}
            _news_ci = [i for i, n in enumerate(self.feature_names) if n in _news_set_contrib]
            _news_sum = float(sum(norm[i] for i in _news_ci))
            if _news_sum < _news_weight_min:
                _deficit = _news_weight_min - _news_sum
                _non_ci = [i for i in range(len(norm)) if i not in _news_ci]
                _non_sum = float(sum(norm[i] for i in _non_ci))
                if _non_sum > _deficit:
                    _scale = (_non_sum - _deficit) / _non_sum
                    for _i in _non_ci:
                        norm[_i] = np.float32(norm[_i] * _scale)
                    _floor_per = np.float32(_news_weight_min / len(_news_ci))
                    for _i in _news_ci:
                        norm[_i] = _floor_per
        feature_weights = {
            name: round(float(weight), 4) for name, weight in zip(self.feature_names, norm.tolist())
        }
        top = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:2]
        expected_reward_pct = float((probs[1] - probs[2]) * 1.5)
        policy_line = (
            f"Policy softmax: P(HOLD/BUY/SELL)={prob_hold:.1%}/{prob_buy:.1%}/{prob_sell:.1%} → top {greedy_name}. "
        )
        exec_line = ""
        if greedy_mode:
            exec_line = "Live inference: ε=0 (greedy policy, geen exploratie). "
        elif explored and int(action) != greedy:
            exec_line = (
                f"Uitvoering na exploratie (ε={eps_live:.3f}): {self._action_to_name(int(action))} "
                f"(wijkt af van greedy {greedy_name}). "
            )
        tech_line = f"Technische drivers: {top[0][0]} ({top[0][1]:.2f}), {top[1][0]} ({top[1][1]:.2f}). "
        reasoning = (
            f"{risk_prefix}"
            f"{gated_note}"
            f"{policy_line}"
            f"{exec_line}"
            f"{tech_line}"
            f"{whale_damp_note}"
            f"Uitgevoerde actie: {self._action_to_name(int(action))}. "
            f"Verwachte beloning: {expected_reward_pct:+.2f}%."
        )
        raw_steps = [s for s in [risk_prefix, gated_note, policy_line, exec_line, tech_line, whale_damp_note] if s and s.strip()]
        raw_steps.append(f"Uitgevoerde actie: {self._action_to_name(int(action))}. Verwachte beloning: {expected_reward_pct:+.2f}%.")
        decision = RLDecision(
            action=action,
            action_name=self._action_to_name(action),
            confidence=round(confidence, 4),
            expected_reward_pct=round(expected_reward_pct, 4),
            feature_weights=feature_weights,
            reasoning=reasoning,
            prob_hold=round(prob_hold, 4),
            prob_buy=round(prob_buy, 4),
            prob_sell=round(prob_sell, 4),
            thinking_steps=raw_steps,
        )
        self.last_decision = decision
        # Gebruik policy-gewichten (base_w) voor de UI-balk zodat de News-balk zichtbaar blijft
        # ook wanneer de actuele sentiment-observaties dicht bij 0 liggen.
        bw = {n: float(base_w[i]) for i, n in enumerate(self.feature_names)}
        self.last_correlation["news_weight"] = round(
            bw.get("sentiment_score", 0.0)
            + bw.get("news_confidence", 0.0)
            + bw.get("social_volume", 0.0)
            + bw.get("fear_greed_score", 0.0)
            + bw.get("btc_dominance_pct", 0.0)
            + bw.get("whale_pressure", 0.0)
            + bw.get("macro_volatility_window", 0.0),
            4,
        )
        self.last_correlation["price_weight"] = round(
            bw.get("price_action", 0.0)
            + bw.get("volatility_24", 0.0)
            + bw.get("volume_change", 0.0)
            + bw.get("bollinger_width", 0.0)
            + bw.get("bollinger_position", 0.0)
            + bw.get("orderbook_imbalance", 0.0)
            + bw.get("bid_ask_spread", 0.0)
            + bw.get("macd", 0.0)
            + bw.get("rsi_14", 0.0)
            + bw.get("ema_gap_pct", 0.0),
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
