"""
Korte system-auditteksten voor Telegram (Performance & Integrity — verkorte versie).
Volledige HTML-rapporten blijven optioneel via SMTP (zie env SYSTEM_ALERTS_EMAIL_ENABLED).
"""

from __future__ import annotations

from html import escape
from typing import Any


def format_startup_or_daily_audit_telegram(
    *,
    trigger: str,
    cash_eur: float,
    equity_eur: float,
    cuda_available: bool,
    roadmap_pct: int,
    roadmap_done: int,
    roadmap_total: int,
    yesterday_pnl_eur: float,
    alloc_summary: str = "",
) -> str:
    """Startup/dagelijkse notifier-mail vervangen door compact Telegram-bericht."""
    tg = escape(str(trigger or "?").upper())
    cuda_e = "✅" if cuda_available else "❌"
    road_e = "✅" if int(roadmap_pct or 0) >= 90 else ("⚠️" if int(roadmap_pct or 0) >= 50 else "❌")
    eq_ok = "✅" if float(equity_eur or 0.0) > 0 else "❌"
    alloc_line = ""
    if str(alloc_summary or "").strip():
        alloc_line = f"\n📊 {escape(str(alloc_summary).strip())}"
    return (
        f"📋 <b>AI_Trading audit</b> <code>[{tg}]</code>\n"
        f"{eq_ok} Equity: <code>{float(equity_eur):.2f}</code> EUR · Cash: <code>{float(cash_eur):.2f}</code> EUR\n"
        f"{cuda_e} CUDA beschikbaar\n"
        f"{road_e} Roadmap: <code>{int(roadmap_pct)}%</code> ({int(roadmap_done)}/{int(roadmap_total)})\n"
        f"📅 Yesterday P/L: <code>{float(yesterday_pnl_eur):.2f}</code> EUR"
        f"{alloc_line}"
    )


def format_jarvis_integrity_telegram(payload: dict[str, Any]) -> str:
    """Jarvis Performance & Integrity Report — verkort voor Telegram."""
    trigger = escape(str(payload.get("trigger") or "?").upper())
    f = payload.get("financials") or {}
    wr = float(f.get("win_rate_pct") or 0.0)
    pnl = float(f.get("total_pnl_eur") or 0.0)
    wins = int(f.get("wins") or 0)
    losses = int(f.get("losses") or 0)
    wr_e = "✅" if wr >= 50.0 else ("⚠️" if wr >= 45.0 else "❌")
    pnl_e = "✅" if pnl >= 0 else "❌"
    health = payload.get("health") or {}
    cpu = float(health.get("cpu_pct") or 0.0)
    gpu = float(health.get("gpu_util_effective") or 0.0)
    cpu_e = "✅" if cpu < 85 else "⚠️" if cpu < 95 else "❌"
    gpu_e = "✅" if gpu < 90 else "⚠️" if gpu < 98 else "❌"
    doctrine = payload.get("doctrine") or {}
    tenant_ok = str(doctrine.get("tenant_isolation", "")).upper() == "PASS"
    ten_e = "✅" if tenant_ok else "❌"
    conf = payload.get("go_live_confidence") or {}
    audit_ok = bool(conf.get("audit_ok"))
    aud_e = "✅" if audit_ok else "❌"
    score = float(conf.get("score") or 0.0)
    score_e = "✅" if score >= 75 else ("⚠️" if score >= 50 else "❌")
    roadmap = payload.get("roadmap") or {}
    rp = int(roadmap.get("percent") or 0)
    rd = int(roadmap.get("done") or 0)
    rt = int(roadmap.get("total") or 0)
    road_e = "✅" if rp >= 90 else ("⚠️" if rp >= 50 else "❌")
    sig = escape(str(payload.get("live_trading_signal") or "—")[:120])
    paper = payload.get("paper_profit_status") or {}
    equity = float(paper.get("equity_eur") or 0.0)
    return (
        f"📋 <b>AI_Trading Performance &amp; Integrity</b> <code>[{trigger}]</code>\n"
        f"{wr_e} Win rate: <code>{wr:.1f}%</code> ({wins}W/{losses}L) · {pnl_e} P/L (ledger): <code>{pnl:.2f}</code> EUR\n"
        f"{cpu_e} CPU <code>{cpu:.0f}%</code> · {gpu_e} GPU <code>{gpu:.0f}%</code>\n"
        f"{ten_e} Tenant isolation · {aud_e} Audit · {score_e} Go-live score <code>{score:.1f}%</code>\n"
        f"{road_e} Roadmap <code>{rp}%</code> ({rd}/{rt})\n"
        f"💶 Paper equity: <code>{equity:.2f}</code> EUR\n"
        f"🎯 Signal: <code>{sig}</code>"
    )
