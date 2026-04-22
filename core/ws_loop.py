"""
WebSocket helpers: heartbeat loop and throttle loops.
Bevat de afhankelijkheden naar Starlette en mag alleen door de Portal (FastAPI) worden gebruikt.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from starlette.websockets import WebSocket, WebSocketDisconnect

HEARTBEAT_INTERVAL_S = 30.0
TRADES_WS_MIN_INTERVAL_S = 0.5

async def websocket_json_loop_with_heartbeat(
    websocket: WebSocket,
    *,
    tick_interval: float,
    build_payload: Callable[[], dict[str, Any]],
    on_sent: Callable[[], None] | None = None,
) -> None:
    """
    Combineert periodieke JSON-push met receive-timeout voor heartbeat.
    Client antwoordt met tekst die 'hb_ack' bevat na een {"t":"hb"} frame.
    """
    next_tick = 0.0
    next_hb = time.monotonic() + HEARTBEAT_INTERVAL_S
    while True:
        now = time.monotonic()
        deadline = min(next_tick, next_hb)
        timeout = max(0.02, deadline - now)
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            if raw and "hb_ack" in raw.lower():
                next_hb = time.monotonic() + HEARTBEAT_INTERVAL_S
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            raise
        now = time.monotonic()
        if now >= next_hb:
            await websocket.send_json({"t": "hb", "ts": int(now * 1000)})
            next_hb = now + HEARTBEAT_INTERVAL_S
            if on_sent:
                on_sent()
        if now >= next_tick:
            payload = build_payload()
            await websocket.send_json(payload)
            if on_sent:
                on_sent()
            next_tick = now + tick_interval


async def websocket_trades_loop_with_throttle(
    websocket: WebSocket,
    *,
    fetch_rows: Callable[[], list[dict[str, Any]]],
    on_sent: Callable[[], None] | None = None,
) -> None:
    """Max ~2 pushes/s; bij gewijzigde trade-lijst direct (critical)."""
    next_tick = 0.0
    next_hb = time.monotonic() + HEARTBEAT_INTERVAL_S
    prev_sig: str | None = None
    while True:
        now = time.monotonic()
        deadline = min(next_tick, next_hb)
        timeout = max(0.02, deadline - now)
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
            if raw and "hb_ack" in raw.lower():
                next_hb = time.monotonic() + HEARTBEAT_INTERVAL_S
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            raise
        now = time.monotonic()
        if now >= next_hb:
            await websocket.send_json({"t": "hb", "ts": int(now * 1000)})
            next_hb = now + HEARTBEAT_INTERVAL_S
            if on_sent:
                on_sent()
        rows = fetch_rows()
        sig = json.dumps(rows, default=str, separators=(",", ":"))
        critical = prev_sig is None or sig != prev_sig
        prev_sig = sig
        if not critical and now < next_tick:
            continue
        await websocket.send_json({"t": "tr", "d": rows})
        if on_sent:
            on_sent()
        next_tick = now + TRADES_WS_MIN_INTERVAL_S