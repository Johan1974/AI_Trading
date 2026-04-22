"""
Bestand: app/rl/train.py
Relatief pad: ./app/rl/train.py
Functie: Train RL-agent met Stable Baselines3 op Bitvavo data + event-features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from app.rl.data import build_rl_training_frame, default_training_period, fetch_bitvavo_historical_candles
from app.rl.env import BitvavoTradingEnv
from app.rl.agent_rl import get_rl_ppo_device


def train_rl_agent(
    market: str = "BTC-EUR",
    interval: str = "1h",
    total_trades: int = 10000,
    total_timesteps: int = 120000,
    model_output_path: str = "artifacts/rl_ppo_bitvavo",
) -> dict:
    start_dt, end_dt = default_training_period()
    candles_df = fetch_bitvavo_historical_candles(
        market=market,
        interval=interval,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    train_df = build_rl_training_frame(candles_df)

    def _make_env():
        return BitvavoTradingEnv(data=train_df, max_trades=total_trades)

    vec_env = DummyVecEnv([_make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        device=get_rl_ppo_device(),
    )
    model.learn(total_timesteps=total_timesteps)

    output = Path(model_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))

    metrics = evaluate_trained_model(model, train_df, max_trades=total_trades)
    metrics["model_path"] = str(output)
    return metrics


def evaluate_trained_model(model: PPO, frame: pd.DataFrame, max_trades: int = 10000) -> dict:
    env = BitvavoTradingEnv(data=frame, max_trades=max_trades)
    obs, _ = env.reset()
    done = False
    truncated = False
    final_info = {}
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(int(action))
        final_info = info
    return {
        "equity_eur": final_info.get("equity_eur"),
        "pnl_eur_total": final_info.get("pnl_eur_total"),
        "drawdown": final_info.get("drawdown"),
        "trade_count": final_info.get("trade_count"),
    }
