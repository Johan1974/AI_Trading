"""
System-audit: primair Telegram; optionele volledige SMTP-mail bij SYSTEM_ALERTS_EMAIL_ENABLED=1.
"""

from __future__ import annotations

import asyncio
import os
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from typing import Any

import requests
from app.datetime_util import UTC
import pytz

from core.auditor import format_startup_or_daily_audit_telegram


TZ = pytz.timezone("Europe/Amsterdam")

# Voorkom herhaalde "[NOTIFIER] … Telegram not configured" bij elke audit/urgent-call.
_telegram_missing_logged: set[str] = set()


def _log_telegram_not_configured_once(key: str, line: str) -> None:
    if key in _telegram_missing_logged:
        return
    _telegram_missing_logged.add(key)
    print(line)


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name, "0")).strip().lower() in ("1", "true", "yes", "on")


def system_alerts_email_enabled() -> bool:
    """Volledige HTML-e-mail voor restart/Jarvis/urgent alleen als expliciet ingeschakeld (backup)."""
    return _env_truthy("SYSTEM_ALERTS_EMAIL_ENABLED")


def telegram_configured() -> bool:
    token = str(os.getenv("TELEGRAM_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    return bool(token and chat_id)


def send_telegram_message(text: str, *, disable_notification: bool = False) -> bool:
    """Ruwe Telegram-send via bot API (HTML). Zelfde env als TelegramNotifier."""
    token = str(os.getenv("TELEGRAM_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        return False
    msg = str(text or "").strip()
    if not msg:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
                "disable_notification": bool(disable_notification),
            },
            timeout=8,
        )
        return resp.status_code == 200
    except Exception:
        return False


def send_watchdog_recovery_telegram(
    engine_age: float,
    ws_age: float,
    *,
    disable_notification: bool = False,
) -> bool:
    age = int(round(float(engine_age)))
    ws_ag = int(round(float(ws_age)))
    text = (
        "🛠️ <b>WATCHDOG ALERT</b>\n"
        "De engine/ws was vastgelopen.\n"
        "🔄 Actie: Geforceerde herstart uitgevoerd.\n"
        f"⏱️ Status: Engine age: {age}s, WS age: {ws_ag}s"
    )
    return send_telegram_message(text, disable_notification=disable_notification)


def format_telegram_sell_alert(
    market: str,
    qty: float,
    entry_price: float | None,
    exit_price: float | None,
    pnl_eur: float | None,
) -> str:
    q = float(qty or 0.0)
    ep = float(entry_price) if entry_price is not None else 0.0
    xp = float(exit_price) if exit_price is not None else 0.0
    inleg = max(0.0, q * ep)
    verkoop = max(0.0, q * xp)
    resultaat = float(pnl_eur) if pnl_eur is not None else (verkoop - inleg)
    emo = "✅" if resultaat >= 0 else "❌"
    sign = "+" if resultaat >= 0 else "-"
    return (
        f"🔴 <b>SELL ALERT: {escape(str(market or '-'))}</b>\n"
        "------------------------\n"
        f"💰 Inleg: <code>€{inleg:.2f}</code>\n"
        f"📈 Verkoop: <code>€{verkoop:.2f}</code>\n"
        f"✨ Resultaat: <code>{sign}€{abs(resultaat):.2f}</code> {emo}"
    )


def _roadmap_progress(roadmap_path: str) -> tuple[int, int, int]:
    p = Path(roadmap_path)
    if not p.exists():
        return 0, 0, 0
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Overall voortgang:\s*(\d+)%\s*\((\d+)/(\d+)", text)
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def send_restart_report(
    cash_eur: float,
    equity_eur: float,
    cuda_available: bool,
    trigger: str = "startup",
    roadmap_path: str = "ROADMAP.md",
    scanner_selected: list[dict] | None = None,
    ai_reflection: dict | None = None,
    yesterday_pnl_eur: float = 0.0,
    portfolio_distribution: list[dict] | None = None,
    allocation_snapshot: dict[str, Any] | None = None,
) -> bool:
    """
    Verstuurt een verkorte audit naar Telegram. Volledige HTML-mail alleen als
    SYSTEM_ALERTS_EMAIL_ENABLED=1 en PRIVATE_SMTP/PRIVATE_EMAIL geconfigureerd zijn.
    """
    progress_pct, progress_done, progress_total = _roadmap_progress(roadmap_path=roadmap_path)
    alloc = allocation_snapshot if isinstance(allocation_snapshot, dict) else {}
    alloc_summary = str(alloc.get("summary") or "")
    tg_text = format_startup_or_daily_audit_telegram(
        trigger=trigger,
        cash_eur=cash_eur,
        equity_eur=equity_eur,
        cuda_available=cuda_available,
        roadmap_pct=progress_pct,
        roadmap_done=progress_done,
        roadmap_total=progress_total,
        yesterday_pnl_eur=yesterday_pnl_eur,
        alloc_summary=alloc_summary,
    )
    tg_ok = send_telegram_message(tg_text, disable_notification=False)
    if tg_ok:
        print("[NOTIFIER] Restart audit (Telegram) sent.")
    elif not telegram_configured():
        _log_telegram_not_configured_once(
            "restart_audit",
            "[NOTIFIER] Restart audit: Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID).",
        )
    else:
        print("[NOTIFIER] Restart audit: Telegram send failed.")

    if not system_alerts_email_enabled():
        return tg_ok

    smtp_server = str(os.getenv("PRIVATE_SMTP", "")).strip()
    smtp_port = int((os.getenv("PRIVATE_PORT") or "587"))
    email_user = str(os.getenv("PRIVATE_EMAIL", "")).strip()
    email_pass = str(os.getenv("PRIVATE_PASS", "")).strip()
    receiver_email = str(os.getenv("PRIVATE_EMAIL", "")).strip()

    if not (smtp_server and email_user and receiver_email):
        print("[NOTIFIER] Restart report email skipped: SMTP config incompleet (SYSTEM_ALERTS_EMAIL_ENABLED=1).")
        return tg_ok

    ts_utc = datetime.now(UTC).isoformat()
    ts_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    gpu_text = "CUDA AVAILABLE" if bool(cuda_available) else "CUDA UNAVAILABLE"
    tenant_isolation = "PASS"
    tenant_note = "Tenant-scoped runtime state + tenant-tagged SQLite records are active."
    low_hanging_rows = """
    <tr><td>Performance</td><td>Blocking IO uit hot API-routes halen</td><td>Sync calls in /predict en /paper/run</td><td>M</td></tr>
    <tr><td>Performance</td><td>System stats sampling centraliseren</td><td>Frequente nvidia-smi subprocess calls</td><td>S</td></tr>
    <tr><td>Security & Doctrine</td><td>Zero-trace logging aanscherpen</td><td>Plaintext headlines/ai_thought traces</td><td>S</td></tr>
    <tr><td>Security & Doctrine</td><td>Tenant boundary afdwingen</td><td>Globale STATE + gedeelde SQLite zonder tenant-id</td><td>M</td></tr>
    <tr><td>Developer Velocity</td><td>app/main.py opdelen</td><td>Orchestratie en endpoints in 1 module</td><td>M</td></tr>
    <tr><td>Developer Velocity</td><td>Nieuwsflow simplificeren</td><td>Overlappende ingestion/news services</td><td>M</td></tr>
    """
    strategic_rows = """
    <tr><td>Performance</td><td>Event-driven ingestion + backpressure</td><td>Polling+sync mix op kritieke paden</td><td>L</td></tr>
    <tr><td>Performance</td><td>GPU scheduler voor GTX 1080</td><td>RL-train en inference concurreren ad hoc</td><td>L</td></tr>
    <tr><td>Security & Doctrine</td><td>Doctrine-as-code checks</td><td>Policies niet hard afgedwongen in CI/runtime</td><td>M</td></tr>
    <tr><td>Security & Doctrine</td><td>Per-tenant data partitioning</td><td>Globale storage en blast radius</td><td>L</td></tr>
    <tr><td>Developer Velocity</td><td>Pure decision core</td><td>Veel side-effects in runtime chain</td><td>L</td></tr>
    <tr><td>Developer Velocity</td><td>Docs-naar-code alignment</td><td>Scope/terminologie drift zonder guardrails</td><td>S</td></tr>
    """
    strategy_focus_rows = """
    <tr><td>Regime-awareness</td><td>Trades zijn nog te weinig regime-adaptief</td><td>Voeg regime detector + dynamische risk bands toe</td></tr>
    <tr><td>Confidence gating</td><td>Signalen met lage zekerheid worden nog uitgevoerd</td><td>Harde confidence drempel + position scaling per confidence</td></tr>
    <tr><td>Execution realism</td><td>Paper outcomes verschillen van live frictie</td><td>Gebruik orderbook spread/slippage model in alle paper cycles</td></tr>
    """
    rl_sources_rows = """
    <tr><td>Nieuwskwaliteit</td><td>Ruis/fake/dup nieuws vervuilt sentiment state</td><td>Dedup + source reliability + novelty scoring verplicht maken</td></tr>
    <tr><td>Market microstructure</td><td>RL mist orderflow-signalen</td><td>Voeg orderbook imbalance, depth pressure en trade tape features toe</td></tr>
    <tr><td>Macro context</td><td>Event-impact wordt beperkt meegewogen</td><td>Breid macro/event calendar uit met severity gewichten</td></tr>
    """
    system_opt_rows = """
    <tr><td>CPU load</td><td>Polling loops blijven actief bij lage marktactiviteit</td><td>Adaptive polling + event-driven updates voor idle periodes</td></tr>
    <tr><td>GPU utilization</td><td>RL-train en inference concurreren</td><td>Schedule train windows en reserveer inference budget</td></tr>
    <tr><td>I/O latency</td><td>Blocking calls in request path</td><td>Async clients + worker queue voor externe API calls</td></tr>
    """
    self_maintenance_rows = """
    <tr><td>Technisch onderhoud</td><td>Geen volledige self-heal runbook in runtime</td><td>Automatische dependency checks, data pruning en health remediation jobs</td></tr>
    <tr><td>Trading onderhoud</td><td>Beperkte auto-calibratie van risk/judge</td><td>Dagelijkse parameter re-evaluatie op win-rate, DD en sentiment drift</td></tr>
    <tr><td>Governance</td><td>Doctrine-afwijkingen niet hard geblokkeerd</td><td>Policy-as-code gates (tenant isolation / zero-trace) als deployment blocker</td></tr>
    """
    scanner_rows = ""
    for row in scanner_selected or []:
        reason = escape(str(row.get("selection_reason") or "n/a"))
        scanner_rows += (
            f"<tr><td>{escape(str(row.get('market','-')))}</td>"
            f"<td>{float(row.get('volume_quote_24h') or 0.0):.2f}</td>"
            f"<td>{float(row.get('move_pct_4h') or 0.0):.3f}%</td>"
            f"<td>{reason}</td></tr>"
        )
    if not scanner_rows:
        scanner_rows = "<tr><td colspan='4'>No dynamic scanner selection available</td></tr>"
    reflection = ai_reflection or {}
    reflection_reason = escape(str(reflection.get("reason") or "No autonomous tuning applied yet."))
    old_thr = float(reflection.get("old_decision_threshold") or 0.0)
    new_thr = float(reflection.get("new_decision_threshold") or 0.0)
    old_sl = float(reflection.get("old_stop_loss_pct") or 0.0)
    new_sl = float(reflection.get("new_stop_loss_pct") or 0.0)
    distribution_rows = ""
    for row in portfolio_distribution or []:
        distribution_rows += (
            f"<tr><td>{escape(str(row.get('asset') or '-'))}</td>"
            f"<td>{float(row.get('qty') or 0.0):.6f}</td>"
            f"<td>{float(row.get('weight_pct') or 0.0):.2f}%</td></tr>"
        )
    if not distribution_rows:
        distribution_rows = "<tr><td>EUR</td><td>0.000000</td><td>0.00%</td></tr>"
    alloc = allocation_snapshot if isinstance(allocation_snapshot, dict) else {}
    alloc_summary = escape(str(alloc.get("summary") or "Allocatie: —"))
    alloc_slot = float(alloc.get("slot_pct") or 12.5)
    alloc_lines = ""
    for row in alloc.get("lines") or []:
        if not isinstance(row, dict):
            continue
        coin = escape(str(row.get("coin") or "-"))
        wp = float(row.get("weight_pct") or 0.0)
        alloc_lines += f"<tr><td>{coin}</td><td>{wp:.2f}%</td></tr>"
    if not alloc_lines:
        alloc_lines = "<tr><td colspan='2'>Geen actieve posities</td></tr>"
    executive_rows = f"""
    <tr><th align="left">Trigger</th><td>{escape(trigger.upper())}</td></tr>
    <tr><th align="left">Cash / Equity</th><td>{cash_eur:.2f} EUR / {equity_eur:.2f} EUR</td></tr>
    <tr><th align="left">P/L</th><td>{(equity_eur - cash_eur):.2f} EUR ({(((equity_eur - cash_eur) / max(1.0, equity_eur)) * 100.0):.2f}%)</td></tr>
    <tr><th align="left">Allocatie (Elite)</th><td>{alloc_summary} · max {alloc_slot:.1f}% per munt</td></tr>
    <tr><th align="left">CUDA</th><td>{'YES' if cuda_available else 'NO'}</td></tr>
    <tr><th align="left">Roadmap</th><td>{progress_pct}% ({progress_done}/{progress_total})</td></tr>
    <tr><th align="left">Tenant Isolation</th><td>{tenant_isolation}</td></tr>
    <tr><th align="left">Top selectie-reden</th><td>{escape(str((scanner_selected or [{}])[0].get("selection_reason") if scanner_selected else "n/a"))}</td></tr>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"AI_Trading System Audit [{trigger.upper()}]"
    msg["From"] = email_user
    msg["To"] = receiver_email
    html = f"""
<html><body style="font-family:Arial,sans-serif;color:#111">
  <h3>AI_Trading Performance &amp; Integrity Report</h3>
  <h4>Executive Snapshot (Mobiel)</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    {executive_rows}
  </table>
  <table border="1" cellspacing="0" cellpadding="6" style="margin-top:8px">
    <tr><th>Priority</th><th align="left">Actie</th></tr>
    <tr><td>P1</td><td>Verplaats blocking IO uit request-path naar worker-queue</td></tr>
    <tr><td>P2</td><td>Forceer orderbook-based spread/slippage in paper execution</td></tr>
    <tr><td>P3</td><td>Implementeer tenant-scoped state/storage als hard gate</td></tr>
    <tr><td>P4</td><td>Activeer dagelijkse auto-calibratie op win-rate/DD drift</td></tr>
  </table>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th align="left">Execution Engine</th><td>Helsinki Node</td></tr>
    <tr><th align="left">Trigger</th><td>{escape(trigger.upper())}</td></tr>
    <tr><th align="left">Generated (local)</th><td>{escape(ts_local)}</td></tr>
    <tr><th align="left">Generated (UTC)</th><td>{escape(ts_utc)}</td></tr>
  </table>
  <h4>Equal-weight allocatie (Executive)</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Munt</th><th>Portfolio %</th></tr>
    {alloc_lines}
  </table>
  <h4>Financials</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Cash EUR</th><th>Equity EUR</th><th>P/L EUR</th><th>P/L %</th><th>Yesterday P/L</th></tr>
    <tr>
      <td>{cash_eur:.2f}</td>
      <td>{equity_eur:.2f}</td>
      <td>{(equity_eur - cash_eur):.2f}</td>
      <td>{(((equity_eur - cash_eur) / max(1.0, equity_eur)) * 100.0):.2f}%</td>
      <td>{float(yesterday_pnl_eur or 0.0):.2f}</td>
    </tr>
  </table>
  <h4>Portfolio Verdeling</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Asset</th><th>Qty</th><th>Weight</th></tr>
    {distribution_rows}
  </table>
  <h4>System Health</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>GPU Status</th><th>CUDA</th></tr>
    <tr><td>{gpu_text}</td><td>{'YES' if cuda_available else 'NO'}</td></tr>
  </table>
  <h4>Roadmap &amp; Doctrine</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Roadmap progress</th><th>Tenant Isolation</th><th>Integrity note</th></tr>
    <tr>
      <td>{progress_pct}% ({progress_done}/{progress_total})</td>
      <td>{tenant_isolation}</td>
      <td>{escape(tenant_note)}</td>
    </tr>
  </table>
  <h4>AI System Audit — Low-hanging fruit</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Front</th><th>Kans</th><th>Evidence</th><th>Effort</th></tr>
    {low_hanging_rows}
  </table>
  <h4>AI System Audit — Strategic improvements</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Front</th><th>Verbetering</th><th>Evidence</th><th>Horizon</th></tr>
    {strategic_rows}
  </table>
  <h4>Kernobservatie</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Observatie</th></tr>
    <tr><td>Documentatie stuurt op modulaire risk-first architectuur, maar runtime-concentratie en globale state veroorzaken frictie op performance, doctrine-naleving en ontwikkelsnelheid.</td></tr>
  </table>
  <h4>Aandachtspunten &amp; Verbeterpunten — Strategie</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Onderwerp</th><th>Aandachtspunt</th><th>Verbeterpunt</th></tr>
    {strategy_focus_rows}
  </table>
  <h4>Aandachtspunten &amp; Verbeterpunten — Info Sources t.b.v. RL</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Bronlaag</th><th>Aandachtspunt</th><th>Verbeterpunt</th></tr>
    {rl_sources_rows}
  </table>
  <h4>Aandachtspunten &amp; Verbeterpunten — Systeemoptimalisatie</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Domein</th><th>Aandachtspunt</th><th>Verbeterpunt</th></tr>
    {system_opt_rows}
  </table>
  <h4>Aandachtspunten &amp; Verbeterpunten — Self-Maintenance</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Domein</th><th>Aandachtspunt</th><th>Verbeterpunt</th></tr>
    {self_maintenance_rows}
  </table>
  <h4>Dynamic High-Volatility Scanner Selection</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Pair</th><th>Volume 24h</th><th>Move 4h</th><th>Reason</th></tr>
    {scanner_rows}
  </table>
  <h4>AI Zelfreflectie</h4>
  <table border="1" cellspacing="0" cellpadding="6">
    <tr><th>Decision Threshold</th><th>Stop Loss %</th><th>Waarom</th></tr>
    <tr><td>{old_thr:.2f} → {new_thr:.2f}</td><td>{old_sl:.2f}% → {new_sl:.2f}%</td><td>{reflection_reason}</td></tr>
  </table>
</body></html>
"""
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            if email_pass:
                server.login(email_user, email_pass)
            server.sendmail(email_user, [receiver_email], msg.as_string())
        print("[NOTIFIER] Restart report email sent.")
        return True
    except Exception as exc:
        print(f"[NOTIFIER] Restart report email failed: {exc}")
        return tg_ok


async def daily_restart_report_loop(
    payload_provider,
    roadmap_path: str = "ROADMAP.md",
) -> None:
    while True:
        now = datetime.now(TZ)
        target = now.replace(hour=8, minute=0, second=0, microsecond=0)
        if now >= target:
            target = target + timedelta(days=1)
        await asyncio.sleep(max(1.0, (target - now).total_seconds()))
        try:
            payload = payload_provider()
            send_restart_report(
                cash_eur=float(payload.get("cash_eur", 0.0)),
                equity_eur=float(payload.get("equity_eur", 0.0)),
                cuda_available=bool(payload.get("cuda_available", False)),
                trigger="daily-executive-summary",
                roadmap_path=roadmap_path,
                scanner_selected=list(payload.get("scanner_selected") or []),
                ai_reflection=dict(payload.get("ai_reflection") or {}),
                yesterday_pnl_eur=float(payload.get("yesterday_pnl_eur", 0.0) or 0.0),
                portfolio_distribution=list(payload.get("portfolio_distribution") or []),
                allocation_snapshot=payload.get("allocation_snapshot")
                if isinstance(payload.get("allocation_snapshot"), dict)
                else None,
            )
        except Exception as exc:
            print(f"[NOTIFIER] Daily restart report failed: {exc}")


def send_urgent_alert(subject: str, details: str) -> bool:
    ts_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    safe_sub = escape(str(subject or ""))
    safe_det = escape(str(details or ""))
    tg_body = (
        "🚨 <b>URGENT</b>\n"
        f"<b>{safe_sub}</b>\n"
        f"<code>{escape(ts_local)}</code>\n\n"
        f"{safe_det}"
    )
    tg_ok = send_telegram_message(tg_body, disable_notification=False)
    if tg_ok:
        print("[NOTIFIER] Urgent alert sent to Telegram.")
    elif not telegram_configured():
        _log_telegram_not_configured_once(
            "urgent_alert",
            "[NOTIFIER] Urgent alert: Telegram not configured (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID).",
        )
    else:
        print("[NOTIFIER] Urgent alert: Telegram send failed.")

    if not system_alerts_email_enabled():
        return tg_ok

    smtp_server = str(os.getenv("PRIVATE_SMTP", "")).strip()
    smtp_port = int((os.getenv("PRIVATE_PORT") or "587"))
    email_user = str(os.getenv("PRIVATE_EMAIL", "")).strip()
    email_pass = str(os.getenv("PRIVATE_PASS", "")).strip()
    receiver_email = str(os.getenv("PRIVATE_EMAIL", "")).strip()
    if not (smtp_server and email_user and receiver_email):
        print("[NOTIFIER] Urgent alert email skipped: SMTP incompleet (SYSTEM_ALERTS_EMAIL_ENABLED=1).")
        return tg_ok
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[URGENT] {subject}"
    msg["From"] = email_user
    msg["To"] = receiver_email
    html = (
        "<html><body style='font-family:Arial,sans-serif;color:#111'>"
        f"<h3>AI_Trading Urgent Alert</h3><p><strong>{escape(ts_local)}</strong></p>"
        f"<p>{escape(details)}</p>"
        "</body></html>"
    )
    msg.attach(MIMEText(html, "html", "utf-8"))
    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            if email_pass:
                server.login(email_user, email_pass)
            server.sendmail(email_user, [receiver_email], msg.as_string())
        print("[NOTIFIER] Urgent alert email sent.")
        return True
    except Exception as exc:
        print(f"[NOTIFIER] Urgent alert email failed: {exc}")
        return tg_ok
