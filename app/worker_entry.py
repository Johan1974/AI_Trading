"""
Worker-proces: trading/AI-bootstrap zonder FastAPI of Jinja2.

Start: ``python -m app.worker_entry`` (docker-compose worker service).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv


async def main_loop() -> None:
    load_dotenv()
    os.environ.setdefault("AI_TRADING_PROCESS", "worker")
    from app.trading_core import run_trading_worker_forever

    await run_trading_worker_forever()


if __name__ == "__main__":
    asyncio.run(main_loop())
