"""
Bestand: core/audit_system_health.py
Functie: System Health Audit — container-uptime, API-latencies, resource-verbruik.
Data direct uit Redis (worker_snapshot / ai_trading_snapshot) en psutil.
Verstuurd via Telegram + optionele SMTP (EMAIL_ENABLED=1).
"""

from __future__ import annotations

import json
import os
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from typing import Any

import psutil

UTC = timezone.utc

_PROCESS_START = time.time()


def _redis_client():
    import redis as _redis
    host = str(os.getenv("REDIS_HOST", "redis")).strip()
    port = str(os.getenv("REDIS_PORT", "6379")).strip()
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    if "localhost" in url or "127.0.0.1" in url:
        url = f"redis://{host}:{port}/0"
    return _redis.Redis.from_url(url, decode_responses=True,
                                 socket_connect_timeout=3, socket_timeout=6)


def _read_redis_snapshot() -> dict[str, Any]:
    try:
        r = _redis_client()
        raw = r.hget("worker_snapshot", "data") or r.get("ai_trading_snapshot")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def _redis_ping_ms() -> float | None:
    try:
        r = _redis_client()
        t0 = time.monotonic()
        r.ping()
        return round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        return None


def _process_uptime_seconds() -> float:
    try:
        return time.time() - psutil.Process(os.getpid()).create_time()
    except Exception:
        return time.time() - _PROCESS_START


def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}u {m}m {sec}s"
    return f"{m}m {sec}s"


def collect_system_health() -> dict[str, Any]:
    snap = _read_redis_snapshot()
    ping_ms = _redis_ping_ms()

    # Resource usage
    try:
        cpu_pct = float(psutil.cpu_percent(interval=0.1))
    except Exception:
        cpu_pct = 0.0
    try:
        ram_pct = float(psutil.virtual_memory().percent)
        ram_used_mb = int(psutil.virtual_memory().used / 1024 / 1024)
    except Exception:
        ram_pct, ram_used_mb = 0.0, 0
    try:
        disk_pct = float(psutil.disk_usage("/").percent)
    except Exception:
        disk_pct = 0.0

    # GPU via cached snapshot
    health = snap.get("system_health") or snap.get("health") or {}
    gpu_util = float(health.get("gpu_util_effective") or health.get("gpu_util_pct") or 0.0)
    gpu_mem_pct = float(health.get("gpu_memory_pct") or 0.0)

    # API latencies uit de snapshot (worker zet ze er in als het er zijn)
    latencies: dict[str, float] = {}
    lat_raw = snap.get("api_latencies") or snap.get("latencies") or {}
    if isinstance(lat_raw, dict):
        for k, v in lat_raw.items():
            try:
                latencies[str(k)] = round(float(v), 1)
            except (TypeError, ValueError):
                pass
    if ping_ms is not None:
        latencies["redis_ping_ms"] = ping_ms

    # Container uptime (proces-niveau)
    uptime_sec = _process_uptime_seconds()

    # WebSocket health
    ws_ts = snap.get("ws_last_tick_ts") or snap.get("last_ws_ts")
    ws_age_s: float | None = None
    if ws_ts:
        try:
            ws_age_s = round((datetime.now(UTC) - datetime.fromisoformat(
                str(ws_ts).replace("Z", "+00:00")
            )).total_seconds(), 0)
        except Exception:
            pass

    return {
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "process_uptime_sec": round(uptime_sec, 0),
        "process_uptime_fmt": _fmt_uptime(uptime_sec),
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "ram_used_mb": ram_used_mb,
        "disk_pct": disk_pct,
        "gpu_util_pct": gpu_util,
        "gpu_mem_pct": gpu_mem_pct,
        "latencies_ms": latencies,
        "ws_age_seconds": ws_age_s,
        "redis_snapshot_keys": len(snap),
    }


def format_system_health_telegram(data: dict[str, Any]) -> str:
    ts = escape(str(data.get("ts_utc", "?"))[:19])
    up = escape(str(data.get("process_uptime_fmt", "?")))
    cpu = float(data.get("cpu_pct", 0))
    ram = float(data.get("ram_pct", 0))
    disk = float(data.get("disk_pct", 0))
    gpu = float(data.get("gpu_util_pct", 0))
    gpu_m = float(data.get("gpu_mem_pct", 0))
    ws_age = data.get("ws_age_seconds")
    redis_keys = int(data.get("redis_snapshot_keys", 0))
    lats = data.get("latencies_ms") or {}

    cpu_e = "✅" if cpu < 80 else ("⚠️" if cpu < 95 else "❌")
    ram_e = "✅" if ram < 80 else ("⚠️" if ram < 92 else "❌")
    disk_e = "✅" if disk < 80 else ("⚠️" if disk < 92 else "❌")
    gpu_e = "✅" if gpu < 85 else ("⚠️" if gpu < 97 else "❌")
    ws_e = "✅" if (ws_age is None or float(ws_age) < 120) else "⚠️"

    lat_lines = ""
    for k, v in list(lats.items())[:4]:
        tag = "✅" if float(v) < 200 else ("⚠️" if float(v) < 1000 else "❌")
        lat_lines += f"\n  {tag} {escape(k)}: <code>{v:.0f} ms</code>"

    ws_line = (
        f"\n🔌 WS lag: <code>{int(ws_age)}s</code> {ws_e}"
        if ws_age is not None else ""
    )

    return (
        f"🖥️ <b>System Health Audit</b> <code>[{ts} UTC]</code>\n"
        f"⏱️ Uptime: <code>{up}</code> · Redis keys: <code>{redis_keys}</code>\n"
        f"{cpu_e} CPU <code>{cpu:.0f}%</code> · "
        f"{ram_e} RAM <code>{ram:.0f}%</code> · "
        f"{disk_e} Disk <code>{disk:.0f}%</code>\n"
        f"{gpu_e} GPU compute <code>{gpu:.0f}%</code> · mem <code>{gpu_m:.0f}%</code>"
        f"{ws_line}"
        f"{lat_lines}"
    )


def format_system_health_email(data: dict[str, Any]) -> tuple[str, str]:
    ts = str(data.get("ts_utc", "?"))[:19]
    subject = f"AI Trading — System Health Audit [{ts} UTC]"

    def row(label: str, value: str, ok: bool | None = None) -> str:
        bg = ""
        if ok is False:
            bg = "background:#fee2e2"
        elif ok is True:
            bg = "background:#dcfce7"
        return f'<tr style="{bg}"><th align="left">{escape(label)}</th><td>{escape(value)}</td></tr>'

    cpu = float(data.get("cpu_pct", 0))
    ram = float(data.get("ram_pct", 0))
    disk = float(data.get("disk_pct", 0))
    gpu = float(data.get("gpu_util_pct", 0))
    gpu_m = float(data.get("gpu_mem_pct", 0))
    ws_age = data.get("ws_age_seconds")
    lats = data.get("latencies_ms") or {}

    lat_rows = ""
    for k, v in lats.items():
        ok = float(v) < 500
        lat_rows += row(f"  {k}", f"{v:.0f} ms", ok)

    ws_row = ""
    if ws_age is not None:
        ws_ok = float(ws_age) < 120
        ws_row = row("WebSocket lag", f"{int(ws_age)}s", ws_ok)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/></head>
<body style="font-family:Arial,sans-serif;color:#111;max-width:680px;margin:auto">
<h2>AI Trading — System Health Audit</h2>
<p style="color:#475569">Generated: {escape(str(data.get("ts_utc","?")))} UTC</p>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
{row("Process uptime", str(data.get("process_uptime_fmt","?")), True)}
{row("Redis snapshot keys", str(data.get("redis_snapshot_keys", 0)))}
{row("CPU", f"{cpu:.1f}%", cpu < 80)}
{row("RAM", f"{ram:.1f}%  ({data.get('ram_used_mb',0)} MB)", ram < 80)}
{row("Disk", f"{disk:.1f}%", disk < 80)}
{row("GPU compute", f"{gpu:.1f}%", gpu < 85)}
{row("GPU memory", f"{gpu_m:.1f}%", gpu_m < 85)}
{ws_row}
</table>
<h3>API / service latencies</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>Endpoint</th><th>Latency (ms)</th></tr>
{lat_rows if lat_rows else '<tr><td colspan="2">Geen latency-data beschikbaar</td></tr>'}
</table>
</body></html>"""
    return subject, html


def send_system_health_audit() -> bool:
    from core.notifier import send_telegram_message, system_alerts_email_enabled

    data = collect_system_health()
    tg_text = format_system_health_telegram(data)
    tg_ok = send_telegram_message(tg_text)

    if not system_alerts_email_enabled():
        return tg_ok

    smtp_server = str(os.getenv("PRIVATE_SMTP", "")).strip()
    smtp_port = int(os.getenv("PRIVATE_PORT") or "587")
    email_user = str(os.getenv("PRIVATE_EMAIL", "")).strip()
    email_pass = str(os.getenv("PRIVATE_PASS", "")).strip()
    receiver = str(os.getenv("PRIVATE_EMAIL", "")).strip()

    if not (smtp_server and email_user and receiver):
        return tg_ok

    subject, html = format_system_health_email(data)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = receiver
    msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as srv:
            srv.starttls()
            if email_pass:
                srv.login(email_user, email_pass)
            srv.sendmail(email_user, [receiver], msg.as_string())
        print("[AUDIT-HEALTH] Email verstuurd.")
        return True
    except Exception as exc:
        print(f"[AUDIT-HEALTH] Email mislukt: {exc}")
        return tg_ok
