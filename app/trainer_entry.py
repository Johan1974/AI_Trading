"""
BESTANDSNAAM: app/trainer_entry.py
FUNCTIE: Zelfstandige PPO-trainer container entry point. Draait onafhankelijk van Worker/Portal.
         Leest paren uit TICKERS env var, traint RLAgentService periodiek op historische
         Bitvavo-data, slaat checkpoints op in artifacts/. Deelt storage/ en artifacts/
         volumes met de Worker maar heeft geen Redis- of STATE-afhankelijkheid.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from app.datetime_util import UTC  # noqa: F401 — timezone.utc wrapper voor Python 3.10

_OVERRIDES_FILE = Path(os.getenv("OVERRIDES_FILE", "data/optimizer_overrides.json"))


def _apply_optimizer_overrides(rl_agent) -> None:
    if not _OVERRIDES_FILE.exists():
        return
    try:
        overrides = json.loads(_OVERRIDES_FILE.read_text())
    except Exception:
        return
    changed = {}
    if "learning_rate" in overrides and rl_agent.model is not None:
        new_lr = float(overrides.pop("learning_rate"))
        rl_agent.model.lr_schedule = lambda _: new_lr
        changed["learning_rate"] = new_lr
    if changed:
        _OVERRIDES_FILE.write_text(json.dumps(overrides, indent=2))
        print(f"[TRAINER] optimizer override toegepast: {changed}", flush=True)


async def _trainer_loop() -> None:
    from app.rl.agent_rl import RLAgentService  # noqa: PLC0415

    rl_agent = RLAgentService()

    tickers_raw = os.getenv("TICKERS", os.getenv("DEFAULT_TICKER", "BTC-EUR"))
    pairs = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    interval_sec = max(30, int(os.getenv("RL_TRAIN_INTERVAL_SEC", "120") or 120))
    chunk = max(512, int(os.getenv("RL_TRAIN_CHUNK_STEPS", "1000") or 1000))
    lookback = max(30, int(os.getenv("RL_LOOKBACK_DAYS", "60") or 60))
    conc = max(1, int(os.getenv("RL_BG_TRAIN_CONCURRENCY", "1") or 1))
    checkpoint_interval = float(os.getenv("RL_CHECKPOINT_INTERVAL_SEC", "3600") or 3600)

    print(
        f"[TRAINER] Gestart | paren={pairs} | interval={interval_sec}s "
        f"| chunk={chunk} | lookback={lookback}d | concurrency={conc}"
    )

    last_checkpoint = 0.0

    while True:
        _apply_optimizer_overrides(rl_agent)
        t0 = time.monotonic()
        try:
            sem = asyncio.Semaphore(conc)

            async def _train_one(pair: str) -> None:
                async with sem:
                    await asyncio.to_thread(
                        rl_agent.online_update,
                        pair,
                        max(1, lookback // 16),
                        chunk,
                        None,
                    )

            await asyncio.gather(*[_train_one(p) for p in pairs])
            elapsed = time.monotonic() - t0
            print(f"[TRAINER] PPO chunk OK | {', '.join(pairs)} | {elapsed:.1f}s")
        except Exception as exc:
            print(f"[TRAINER] Training mislukt: {exc}")

        now = time.monotonic()
        if now - last_checkpoint >= checkpoint_interval:
            try:
                await asyncio.to_thread(rl_agent.save_hourly_checkpoint, pairs[0])
                print(f"[TRAINER] Checkpoint opgeslagen voor {pairs[0]}")
            except Exception as exc:
                print(f"[TRAINER] Checkpoint mislukt: {exc}")
            last_checkpoint = now

        await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    asyncio.run(_trainer_loop())
