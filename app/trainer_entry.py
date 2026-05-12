"""
BESTANDSNAAM: app/trainer_entry.py
FUNCTIE: Zelfstandige PPO-trainer container entry point. Draait onafhankelijk van Worker/Portal.
         Leest paren uit TICKERS env var, traint RLAgentService periodiek op historische
         Bitvavo-data, slaat checkpoints op in artifacts/. Deelt storage/ en artifacts/
         volumes met de Worker maar heeft geen Redis- of STATE-afhankelijkheid.
"""

from __future__ import annotations

import asyncio
import os
import time

from app.datetime_util import UTC  # noqa: F401 — timezone.utc wrapper voor Python 3.10


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
