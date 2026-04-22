"""
Bestand: app/worker.py
Functie: Compat-entrypoint; nieuwe start is ``python -m app.worker_entry``.
"""

from __future__ import annotations

import asyncio

from app.worker_entry import main_loop


if __name__ == "__main__":
    asyncio.run(main_loop())
