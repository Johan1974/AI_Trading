"""
Bestand: core/audit_rl_intelligence.py
Functie: RL Intelligence Audit — global steps delta, TD-error stabiliteit, shadow performance.
Data direct uit Redis (worker_snapshot / ai_trading_snapshot).
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

UTC = timezone.utc

_GLOBAL_STEP_BASELINE_KEY = "_audit_rl_step_baseline"
_step_baseline: dict[str, int] = {}


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


def _stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def _trend_slope(values: list[float]) -> float:
    """Lineaire regressiehelling (laatste N punten), genormaliseerd op de gemiddelde absolute waarde."""
    n = len(values)
    if n < 3:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((xs[i] - mx) * (values[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    avg_abs = sum(abs(v) for v in values) / n
    return slope / avg_abs if avg_abs > 1e-9 else slope


def collect_rl_intelligence() -> dict[str, Any]:
    snap = _read_redis_snapshot()
    tm: dict[str, Any] = snap.get("tm") or snap.get("training_monitor") or {}
    stats: dict[str, Any] = tm.get("stats") or {}
    network_logs: dict[str, Any] = tm.get("network_logs") or {}

    # --- Global Steps delta ---
    global_step = int(stats.get("global_step_count") or 0)
    baseline = _step_baseline.get("value", 0)
    if baseline == 0:
        _step_baseline["value"] = global_step
    step_delta = global_step - baseline

    # --- TD-error (value_loss) stabiliteit ---
    value_loss_series: list[float] = []
    raw_vl = network_logs.get("value_loss") or []
    if isinstance(raw_vl, list):
        for v in raw_vl:
            try:
                value_loss_series.append(float(v))
            except (TypeError, ValueError):
                pass
    tail = value_loss_series[-50:] if len(value_loss_series) >= 3 else value_loss_series
    td_mean = sum(tail) / len(tail) if tail else 0.0
    td_std = _stddev(tail)
    td_cv = td_std / td_mean if td_mean > 1e-9 else 0.0
    td_slope = _trend_slope(tail)

    # --- Shadow Trades overzicht ---
    shadow_raw = snap.get("sh") or snap.get("shadow_trades") or []
    shadow_trades: list[dict[str, Any]] = []
    if isinstance(shadow_raw, list):
        for item in shadow_raw:
            if isinstance(item, dict):
                shadow_trades.append(item)

    by_signal: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    markets_blocked: dict[str, int] = {}
    for t in shadow_trades:
        sig = str(t.get("signal", "?")).upper()
        reason = str(t.get("reason", "?"))
        market = str(t.get("market", "?"))
        by_signal[sig] = by_signal.get(sig, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
        markets_blocked[market] = markets_blocked.get(market, 0) + 1

    top_reasons = sorted(by_reason.items(), key=lambda x: -x[1])[:4]
    last_shadow = shadow_trades[-1] if shadow_trades else {}

    # --- Extra RL stats ---
    lr = float(stats.get("learning_rate") or 0.0)
    exploration_pct = float(stats.get("exploration_rate_pct") or 0.0)
    is_training = bool(stats.get("is_training_active"))
    batch_size = int(stats.get("batch_size") or 0)
    approx_kl_series: list[float] = []
    raw_kl = network_logs.get("approx_kl") or []
    if isinstance(raw_kl, list):
        for v in raw_kl:
            try:
                approx_kl_series.append(float(v))
            except (TypeError, ValueError):
                pass
    approx_kl_last = approx_kl_series[-1] if approx_kl_series else None

    return {
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "global_step": global_step,
        "step_delta": step_delta,
        "is_training_active": is_training,
        "learning_rate": lr,
        "exploration_rate_pct": exploration_pct,
        "batch_size": batch_size,
        "td_mean": round(td_mean, 6),
        "td_std": round(td_std, 6),
        "td_cv": round(td_cv, 4),
        "td_slope": round(td_slope, 4),
        "td_series_len": len(value_loss_series),
        "approx_kl_last": round(approx_kl_last, 5) if approx_kl_last is not None else None,
        "shadow_total": len(shadow_trades),
        "shadow_by_signal": by_signal,
        "shadow_by_reason": top_reasons,
        "shadow_markets": markets_blocked,
        "shadow_last": last_shadow,
    }


def _stability_label(cv: float, slope: float) -> str:
    if cv < 0.10 and abs(slope) < 0.02:
        return "✅ stabiel"
    if cv < 0.25 or slope < -0.01:
        return "⚠️ licht instabiel"
    return "❌ instabiel"


def format_rl_intelligence_telegram(data: dict[str, Any]) -> str:
    ts = escape(str(data.get("ts_utc", "?"))[:19])
    gs = int(data.get("global_step", 0))
    delta = int(data.get("step_delta", 0))
    training = "✅ actief" if data.get("is_training_active") else "⏸️ inactief"
    lr = float(data.get("learning_rate") or 0.0)
    eps = float(data.get("exploration_rate_pct") or 0.0)
    batch = int(data.get("batch_size") or 0)

    td_mean = float(data.get("td_mean") or 0.0)
    td_std = float(data.get("td_std") or 0.0)
    td_cv = float(data.get("td_cv") or 0.0)
    td_slope = float(data.get("td_slope") or 0.0)
    td_n = int(data.get("td_series_len") or 0)
    stab = _stability_label(td_cv, td_slope)
    kl = data.get("approx_kl_last")

    shadow_total = int(data.get("shadow_total") or 0)
    by_signal = data.get("shadow_by_signal") or {}
    top_reasons = data.get("shadow_by_reason") or []

    shadow_sig_line = " · ".join(
        f"{sig}: {cnt}" for sig, cnt in sorted(by_signal.items())
    ) or "—"

    reason_lines = ""
    for reason, cnt in top_reasons:
        reason_lines += f"\n  • {escape(str(reason))}: <code>{cnt}×</code>"

    kl_line = f"\n📐 Approx KL: <code>{kl:.5f}</code>" if kl is not None else ""

    delta_arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "—")

    return (
        f"🧠 <b>RL Intelligence Audit</b> <code>[{ts} UTC]</code>\n"
        f"🔢 Global Steps: <code>{gs:,}</code> {delta_arrow} <code>Δ{delta:+,}</code>\n"
        f"🏋️ Training: {training} · LR: <code>{lr:.2e}</code> · Batch: <code>{batch}</code>\n"
        f"🎲 Exploratie: <code>{eps:.1f}%</code>"
        f"{kl_line}\n"
        f"\n📉 <b>TD-Error (value_loss)</b> — laatste <code>{td_n}</code> punten\n"
        f"  μ: <code>{td_mean:.4f}</code> · σ: <code>{td_std:.4f}</code> · "
        f"CV: <code>{td_cv:.3f}</code>\n"
        f"  Helling: <code>{td_slope:+.4f}</code> · {stab}\n"
        f"\n🚫 <b>Shadow Trades</b>: <code>{shadow_total}</code> geblokkeerd\n"
        f"  Signalen: {escape(shadow_sig_line)}"
        f"{reason_lines if reason_lines else chr(10) + '  Geen redenen gevonden.'}"
    )


def format_rl_intelligence_email(data: dict[str, Any]) -> tuple[str, str]:
    ts = str(data.get("ts_utc", "?"))[:19]
    subject = f"AI Trading — RL Intelligence Audit [{ts} UTC]"

    def row(label: str, value: str, ok: bool | None = None) -> str:
        bg = ""
        if ok is False:
            bg = "background:#fee2e2"
        elif ok is True:
            bg = "background:#dcfce7"
        return f'<tr style="{bg}"><th align="left">{escape(label)}</th><td>{escape(value)}</td></tr>'

    gs = int(data.get("global_step", 0))
    delta = int(data.get("step_delta", 0))
    is_training = bool(data.get("is_training_active"))
    lr = float(data.get("learning_rate") or 0.0)
    eps = float(data.get("exploration_rate_pct") or 0.0)
    batch = int(data.get("batch_size") or 0)
    td_mean = float(data.get("td_mean") or 0.0)
    td_std = float(data.get("td_std") or 0.0)
    td_cv = float(data.get("td_cv") or 0.0)
    td_slope = float(data.get("td_slope") or 0.0)
    td_n = int(data.get("td_series_len") or 0)
    kl = data.get("approx_kl_last")
    shadow_total = int(data.get("shadow_total") or 0)
    by_signal = data.get("shadow_by_signal") or {}
    top_reasons = data.get("shadow_by_reason") or []
    shadow_markets = data.get("shadow_markets") or {}

    stab = _stability_label(td_cv, td_slope)

    reason_rows = ""
    for reason, cnt in top_reasons:
        reason_rows += f"<tr><td>{escape(str(reason))}</td><td>{cnt}×</td></tr>"

    signal_rows = ""
    for sig, cnt in sorted(by_signal.items()):
        signal_rows += f"<tr><td>{escape(sig)}</td><td>{cnt}</td></tr>"

    market_rows = ""
    for mkt, cnt in sorted(shadow_markets.items(), key=lambda x: -x[1])[:5]:
        market_rows += f"<tr><td>{escape(mkt)}</td><td>{cnt}</td></tr>"

    kl_row = row("Approx KL (laatste)", f"{kl:.5f}") if kl is not None else ""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/></head>
<body style="font-family:Arial,sans-serif;color:#111;max-width:680px;margin:auto">
<h2>AI Trading — RL Intelligence Audit</h2>
<p style="color:#475569">Generated: {escape(str(data.get("ts_utc","?")))} UTC</p>

<h3>Training Status</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
{row("Global Steps", f"{gs:,}")}
{row("Steps delta (deze run)", f"{delta:+,}", delta >= 0)}
{row("Training actief", "Ja" if is_training else "Nee", is_training)}
{row("Learning rate", f"{lr:.2e}")}
{row("Exploratie %", f"{eps:.1f}%")}
{row("Batch size", str(batch))}
{kl_row}
</table>

<h3>TD-Error Stabiliteit (value_loss — {td_n} punten)</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
{row("Gemiddelde (μ)", f"{td_mean:.6f}")}
{row("Standaarddev (σ)", f"{td_std:.6f}")}
{row("Variatiecoëfficiënt (CV)", f"{td_cv:.4f}", td_cv < 0.25)}
{row("Helling (genormaliseerd)", f"{td_slope:+.4f}", abs(td_slope) < 0.02)}
{row("Stabiliteitsklasse", stab.replace("✅ ", "").replace("⚠️ ", "").replace("❌ ", ""))}
</table>

<h3>Shadow Trades — {shadow_total} geblokkeerde signalen</h3>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>Signaal</th><th>Aantal</th></tr>
{signal_rows if signal_rows else '<tr><td colspan="2">Geen data</td></tr>'}
</table>

<h4>Top blokkeringsredenen</h4>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>Reden</th><th>Frequentie</th></tr>
{reason_rows if reason_rows else '<tr><td colspan="2">Geen data</td></tr>'}
</table>

<h4>Meest geblokkeerde markten</h4>
<table border="1" cellspacing="0" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>Markt</th><th>Blokkades</th></tr>
{market_rows if market_rows else '<tr><td colspan="2">Geen data</td></tr>'}
</table>
</body></html>"""
    return subject, html


def send_rl_intelligence_audit() -> bool:
    from core.notifier import send_telegram_message, system_alerts_email_enabled

    data = collect_rl_intelligence()
    tg_text = format_rl_intelligence_telegram(data)
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

    subject, html = format_rl_intelligence_email(data)
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
        print("[AUDIT-RL] Email verstuurd.")
        return True
    except Exception as exc:
        print(f"[AUDIT-RL] Email mislukt: {exc}")
        return tg_ok
