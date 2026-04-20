"""
Bestand: app/services/telegram_notifier.py
Relatief pad: ./app/services/telegram_notifier.py
Functie: Verstuurt operationele meldingen naar Telegram.
"""

from __future__ import annotations

import socket
from html import escape
from datetime import datetime

import pytz
import requests

_tz = pytz.timezone("Europe/Amsterdam")


def _telegram_timestamp() -> str:
    return datetime.now(_tz).strftime("%Y-%m-%d %H:%M:%S")


class TelegramNotifier:
    def __init__(self, token: str | None, chat_id: str | None, enabled: bool = True) -> None:
        self.token = str(token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.enabled = bool(enabled) and bool(self.token) and bool(self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        msg = str(text or "").strip()
        if not msg:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = requests.post(
                url,
                data={
                    "chat_id": self.chat_id,
                    "text": msg,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=8,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _host_line(self) -> str:
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    def send_start(self) -> None:
        ts = _telegram_timestamp()
        host = escape(self._host_line())
        self.send(
            "🟢 <b>AI Trading Bot gestart</b>\n"
            f"Tijd: <code>{escape(ts)}</code>\n"
            f"Host: <code>{host}</code>\n"
            "Engine-loop en API zijn actief."
        )

    def send_stop(self, reason: str = "shutdown") -> None:
        ts = _telegram_timestamp()
        host = escape(self._host_line())
        safe_reason = escape(str(reason or "shutdown"))
        self.send(
            "🔴 <b>AI Trading Bot gestopt</b>\n"
            f"Tijd: <code>{escape(ts)}</code>\n"
            f"Host: <code>{host}</code>\n"
            f"Reden: <code>{safe_reason}</code>"
        )

    def send_bot_status(self, status: str) -> None:
        emoji = "⏸️" if status == "paused" else ("🚨" if status == "panic_stop" else "▶️")
        self.send(f"{emoji} <b>Bot status</b>: <code>{status}</code>")

    def send_trade(self, market: str, signal: str, price: float, qty: float, equity: float) -> None:
        side = str(signal or "").upper()
        icon = "🟢" if side == "BUY" else "🔴"
        self.send(
            f"{icon} <b>Paper Trade {side}</b>\n"
            f"Pair: <code>{market}</code>\n"
            f"Qty: <code>{qty:.8f}</code>\n"
            f"Prijs: <code>EUR {price:.4f}</code>\n"
            f"Equity: <code>EUR {equity:.2f}</code>"
        )

