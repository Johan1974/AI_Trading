#!/usr/bin/env python3
"""
Genereert AUTO_BRIEFING.md in de project-root.
Bronnen: _logs_hub/, data/rl_replay_buffer.db, artifacts/rl/, Redis (optioneel).
Aanroep: python3 scripts/generate_briefing.py
         of automatisch via _rl_hourly_checkpoint_and_metrics_loop in trading_core.py.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "_logs_hub"
ARTIFACTS_RL = ROOT / "artifacts" / "rl"
OUT = ROOT / "AUTO_BRIEFING.md"
TZ_LOCAL = ZoneInfo("Europe/Amsterdam")
UTC = timezone.utc
NOW = datetime.now(UTC)
NOW_LOCAL = NOW.astimezone(TZ_LOCAL)


def _iso(ts: str | None) -> str:
    return ts[:16].replace("T", " ") + " UTC" if ts else "—"


def _load_json(p: Path) -> dict | list | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── Data loaders ───────────────────────────────────────────────────────────────

def replay_stats() -> dict:
    db = ROOT / "data" / "rl_replay_buffer.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    try:
        total, first_ts, last_ts = con.execute(
            "SELECT COUNT(*), MIN(ts_utc), MAX(ts_utc) FROM rl_replay_experience"
        ).fetchone()
        with_reward = con.execute(
            "SELECT COUNT(*) FROM rl_replay_experience WHERE reward_pct IS NOT NULL"
        ).fetchone()[0]
        kinds = dict(con.execute(
            "SELECT kind, COUNT(*) FROM rl_replay_experience GROUP BY kind"
        ).fetchall())
        signals_24h = dict(con.execute(
            "SELECT executed_signal, COUNT(*) FROM rl_replay_experience"
            " WHERE ts_utc >= datetime('now','-24 hours') GROUP BY executed_signal"
        ).fetchall())
        prob_avg = con.execute(
            "SELECT AVG(prob_buy), AVG(prob_hold), AVG(prob_sell)"
            " FROM rl_replay_experience WHERE ts_utc >= datetime('now','-24 hours')"
        ).fetchone()
        return {
            "total": total, "first_ts": first_ts, "last_ts": last_ts,
            "with_reward": with_reward, "kinds": kinds,
            "signals_24h": signals_24h, "prob_avg_24h": prob_avg,
        }
    finally:
        con.close()


def training_chunks_stats() -> dict:
    db = LOGS / "rl_training_metrics.sqlite"
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    try:
        total = con.execute("SELECT COUNT(*) FROM rl_training_chunks").fetchone()[0]
        last_24h = con.execute(
            "SELECT COUNT(*) FROM rl_training_chunks WHERE ts_utc >= datetime('now','-24 hours')"
        ).fetchone()[0]
        last_row = con.execute(
            "SELECT ts_utc, pair, global_step FROM rl_training_chunks ORDER BY ts_utc DESC LIMIT 1"
        ).fetchone()
        steps_series = con.execute(
            "SELECT ts_utc, global_step FROM rl_training_chunks ORDER BY ts_utc DESC LIMIT 20"
        ).fetchall()
        # Per-pair: max global_step + count (last 24h)
        pair_rows = con.execute(
            """
            SELECT pair, MAX(global_step) as max_step,
                   COUNT(*) as total_chunks,
                   SUM(CASE WHEN ts_utc >= datetime('now','-24 hours') THEN 1 ELSE 0 END) as chunks_24h
            FROM rl_training_chunks
            GROUP BY pair
            ORDER BY max_step DESC
            """
        ).fetchall()
        # Step growth per pair: compare last vs first in last 20 chunks
        pair_growth = {}
        for pair, max_step, total_c, chunks_24h in (pair_rows or []):
            pair_series = con.execute(
                "SELECT global_step FROM rl_training_chunks WHERE pair=? ORDER BY ts_utc DESC LIMIT 20",
                (pair,)
            ).fetchall()
            series_vals = [r[0] for r in reversed(pair_series) if r[0]]
            if len(series_vals) >= 2:
                delta = series_vals[-1] - series_vals[0]
            else:
                delta = 0
            pair_growth[str(pair)] = {
                "max_step": max_step, "delta": delta,
                "chunks_24h": chunks_24h, "total_chunks": total_c,
            }
        return {
            "total": total, "last_24h": last_24h,
            "last_row": last_row, "steps_series": list(reversed(steps_series)),
            "pair_growth": pair_growth,
        }
    finally:
        con.close()


def hourly_metrics() -> list[dict]:
    p = LOGS / "rl_hourly_metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line.strip()))
        except Exception:
            pass
    return out


def model_versions() -> list[dict]:
    out = []
    for jf in sorted(ARTIFACTS_RL.glob("*_models.json")):
        data = _load_json(jf)
        if isinstance(data, list) and data:
            out.append({"file": jf.name, "entries": data, "last": data[-1]})
        elif isinstance(data, dict):
            out.append({"file": jf.name, "entries": [data], "last": data})
    return out


def redis_full_snapshot() -> dict:
    """Leest worker_snapshot / ai_trading_snapshot uit Redis."""
    try:
        import redis as r
        url = str(os.getenv("REDIS_URL") or "").strip()
        host = str(os.getenv("REDIS_HOST", "redis")).strip()
        port = str(os.getenv("REDIS_PORT", "6379")).strip()
        if not url:
            url = f"redis://{host}:{port}/0"
        if "localhost" in url or "127.0.0.1" in url:
            url = f"redis://{host}:{port}/0"
        rc = r.Redis.from_url(url, decode_responses=True,
                              socket_connect_timeout=2, socket_timeout=3)
        raw = rc.hget("worker_snapshot", "data") or rc.get("ai_trading_snapshot")
        rc.close()
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def system_state() -> dict:
    return _load_json(LOGS / "system_state.json") or {}


def recent_crash_lines(max_age_hours: float = 24.0, max_lines: int = 20) -> list[str]:
    """Lees de laatste N uur aan crash/error-regels uit persistent_crash.log."""
    candidates = [
        Path("/app/logs/persistent_crash.log"),
        LOGS / "persistent_crash.log",
    ]
    cutoff = NOW - timedelta(hours=max_age_hours)
    results: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            if len(results) >= max_lines:
                break
            # Zoek ISO-timestamp in de regel
            m = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    ts = datetime.fromisoformat(m.group(1).replace(" ", "T")).replace(tzinfo=UTC)
                    if ts >= cutoff:
                        results.append(line.strip())
                        continue
                except Exception:
                    pass
            # geen timestamp → voeg toe als context als vorige regel al binnen window viel
            if results:
                results.append(line.strip())
        break  # gebruik eerste bestaand bestand
    return list(reversed(results))


# ── Analysis ───────────────────────────────────────────────────────────────────

def collect_critical_alerts(snap: dict, ss: dict, crash_lines: list[str]) -> list[str]:
    """Verzamel CRITICAL ALERTS: API-fouten, container-herstarts, extreme prijsafwijking."""
    alerts: list[str] = []

    # 1. API errors in crash log
    api_errors = [l for l in crash_lines if re.search(r"\b(401|403|429|timeout|auth|forbidden|unauthorized)\b", l, re.I)]
    if api_errors:
        alerts.append(
            f"**API-FOUT ({len(api_errors)} regels):** "
            + " | ".join(api_errors[:2])
        )

    # 2. Overige crashes/errors
    other_errors = [l for l in crash_lines if l not in api_errors and "ERROR" in l.upper()]
    if other_errors:
        alerts.append(
            f"**CRASH LOG ({len(other_errors)} ERROR-regels in 24u):** "
            + other_errors[-1][:200]
        )

    # 3. Container herstart detectie via startup_mode
    startup = str(ss.get("startup_mode") or "manual").lower()
    if startup == "auto":
        alerts.append(
            "**CONTAINER HERSTART:** startup_mode='auto' → de worker is onverwacht herstart "
            "(crash-loop of restart:always). Controleer de logs."
        )

    # 4. Extreme prijsafwijking (predicted vs current)
    # snap.predicted_price is een list; snap.current_price is een getal
    try:
        pred_series = snap.get("predicted_price") or []
        current_px = float(snap.get("current_price") or snap.get("price") or 0.0)
        if pred_series and current_px > 0:
            first_pred = float(pred_series[0]) if isinstance(pred_series[0], (int, float)) else 0.0
            if first_pred > 0:
                dev_pct = abs(first_pred - current_px) / current_px * 100.0
                if dev_pct > 5.0:
                    alerts.append(
                        f"**PRIJS-AFWIJKING {dev_pct:.1f}%:** "
                        f"voorspeld {first_pred:.2f} vs actueel {current_px:.2f}. "
                        "Controleer model freshness of data feed."
                    )
    except Exception:
        pass

    # 5. Rate limit hit
    rl_status = snap.get("rate_limit") or {}
    if isinstance(rl_status, dict) and rl_status.get("is_limited"):
        alerts.append(
            f"**RATE LIMIT ACTIEF:** {rl_status.get('reason','?')} — "
            f"backoff={rl_status.get('backoff_s','?')}s"
        )

    # 6. Redis snapshot ouderdom
    snap_ts = snap.get("ts_utc") or snap.get("generated_at") or ""
    if snap_ts:
        try:
            snap_age = (NOW - datetime.fromisoformat(
                str(snap_ts).replace("Z", "+00:00")
            )).total_seconds()
            if snap_age > 600:
                alerts.append(
                    f"**STALE SNAPSHOT:** worker_snapshot is {int(snap_age/60)}m oud — "
                    "worker loopt mogelijk vast of is gestopt."
                )
        except Exception:
            pass

    return alerts


def global_steps_growth(chunks: dict) -> dict:
    series = chunks.get("steps_series", [])
    if len(series) < 2:
        return {"growth": None, "series": series}
    first_step = series[0][1] or 0
    last_step = series[-1][1] or 0
    growth = last_step - first_step
    return {"growth": growth, "first": first_step, "last": last_step, "series": series}


def prediction_accuracy(snap: dict) -> dict:
    """
    Schat AI-prediction accuraatheid:
    - Berekent afwijking predicted_price[0] t.o.v. current_price (instant accuracy proxy).
    - Dominant policy_prob per markt als confidence indicator.
    """
    results: dict[str, dict] = {}

    # Instant deviation van huidige markt (snap van geselecteerde markt)
    try:
        pred_series = snap.get("predicted_price") or []
        current_px = float(snap.get("current_price") or snap.get("price") or 0.0)
        market = str(snap.get("market") or snap.get("selected_market") or "?").upper()
        if pred_series and current_px > 0:
            first_pred = float(pred_series[0]) if isinstance(pred_series[0], (int, float)) else 0.0
            if first_pred > 0:
                dev_pct = (first_pred - current_px) / current_px * 100.0
                results[market] = {
                    "predicted": first_pred,
                    "actual": current_px,
                    "dev_pct": round(dev_pct, 2),
                }
    except Exception:
        pass

    # Policy confidence per markt (uit elite-lite of policy probs)
    lite = snap.get("lite") or snap.get("elite_lite") or []
    if isinstance(lite, list):
        for row in lite:
            if not isinstance(row, dict):
                continue
            mkt = str(row.get("market") or row.get("m") or "").upper()
            if not mkt:
                continue
            dom = row.get("dominant_policy_prob") or row.get("dpp")
            if dom is not None:
                entry = results.setdefault(mkt, {})
                entry["policy_confidence"] = round(float(dom), 3)

    return results


def detect_learning_plateau(chunks: dict, hm: list[dict]) -> str | None:
    series = chunks.get("steps_series", [])
    if len(series) < 5:
        return None
    recent = [s[1] for s in series[-5:] if s[1] is not None]
    if len(recent) < 2:
        return None
    growth_pct = (recent[-1] - recent[0]) / max(1, recent[0]) * 100
    if growth_pct < 10:
        return (
            f"Global step groeide slechts {growth_pct:.1f}% over de laatste "
            f"{len(recent)} chunks ({recent[0]:,} → {recent[-1]:,})."
        )
    if len(hm) >= 4:
        losses = [h.get("avg_loss") for h in hm[-4:] if h.get("avg_loss") is not None]
        if len(losses) >= 4:
            spread = max(losses) - min(losses)
            if spread < 0.01:
                return f"avg_loss stagneert: {losses} (spread {spread:.4f} < 0.01)."
    return None


def detect_anomalies(rb: dict, chunks: dict, hm: list[dict], ss: dict) -> list[str]:
    anomalies: list[str] = []

    if chunks.get("last_24h", 0) == 0:
        last = chunks.get("last_row")
        anomalies.append(
            f"Geen training chunks in 24u (laatste: {_iso(last[0] if last else None)}). Model leert niet."
        )

    probs = rb.get("prob_avg_24h") or (None, None, None)
    if probs[0] is not None and max(probs) - min(probs) < 0.05:
        anomalies.append(
            f"Policy probs uniform (buy={probs[0]:.3f} hold={probs[1]:.3f} sell={probs[2]:.3f}) — "
            "agent gebruikt fallback-prior, geen geleerd signaal."
        )

    if rb.get("total", 0) > 0 and rb.get("with_reward", 0) == 0:
        anomalies.append(
            f"Alle {rb['total']} replay-entries hebben reward_pct=NULL — geen trade ooit gesloten."
        )

    sigs = rb.get("signals_24h", {})
    total_sigs = sum(sigs.values()) or 1
    hold_pct = sigs.get("HOLD", 0) / total_sigs * 100
    if hold_pct > 65:
        anomalies.append(
            f"Te veel HOLD: {hold_pct:.0f}% van signalen 24u. "
            "decision_threshold waarschijnlijk te hoog of exploration_eps te laag."
        )

    if ss.get("portfolio_equity_integrity_ok") is False:
        anomalies.append(
            f"portfolio_equity_integrity_ok=False: {ss.get('portfolio_equity_integrity_detail','?')}"
        )

    plateau = detect_learning_plateau(chunks, hm)
    if plateau:
        anomalies.append(f"Learning plateau gedetecteerd: {plateau}")

    if len(hm) >= 2:
        rewards = [h.get("avg_final_cumulative_reward") for h in hm if h.get("avg_final_cumulative_reward") is not None]
        if len(rewards) >= 2 and rewards[-1] < rewards[-2]:
            anomalies.append(
                f"Reward dalend: {rewards[-2]:.2f} → {rewards[-1]:.2f}. Controleer learning_rate."
            )

    return anomalies


def generate_action_plan(alerts: list[str], anomalies: list[str],
                         rb: dict, chunks: dict, hm: list[dict]) -> list[str]:
    actions: list[str] = []

    # API-fout (401)
    if any("401" in a or "unauthorized" in a.lower() or "API-FOUT" in a for a in alerts):
        actions.append(
            "**API-authenticatie vernieuwen** — `docker-compose.env`: controleer `BITVAVO_API_KEY` "
            "en `BITVAVO_API_SECRET`. Draai `python3 scripts/verify_api.py` om de credentials "
            "te testen zonder de container te herstarten."
        )

    # Container herstart
    if any("HERSTART" in a or "auto" in a.lower() for a in alerts):
        actions.append(
            "**Crashoorzaak opsporen** — `cat _logs_hub/persistent_crash.log | tail -50`. "
            "Zet `EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART=1` in `docker-compose.env` "
            "voor uitgebreid startup-rapport bij volgende herstart."
        )

    # Stale snapshot
    if any("STALE SNAPSHOT" in a for a in alerts):
        actions.append(
            "**Worker-status controleren** — `docker compose ps` + "
            "`docker compose logs --tail=30 worker`. "
            "Als de worker vastloopt: `docker compose restart worker`."
        )

    # Geen training
    if chunks.get("last_24h", 0) == 0:
        actions.append(
            "**Forceer RL-training** — `POST /api/v1/rl-train` met `{\"force\": true}`, "
            "of zet `RL_BACKGROUND_TRAIN=1` + `RL_TRAIN_INTERVAL_SEC=60` in `docker-compose.env`."
        )

    # Plateau
    plateau = detect_learning_plateau(chunks, hm)
    if plateau:
        actions.append(
            "**Doorbreek learning plateau** — `docker-compose.env`: verhoog "
            "`RL_TRAIN_CHUNK_STEPS` (bijv. naar 2000) en reset "
            "`RL_EXPLORATION_FINAL_EPS=0.15`. Grotere rollouts dwingen PPO tot "
            "meer gradiëntstappen per update."
        )

    # Uniform prior
    probs = rb.get("prob_avg_24h") or (None, None, None)
    if probs[0] is not None and max(probs) - min(probs) < 0.05:
        actions.append(
            "**Model is niet geleerd** — uniform policy-prior. Zet `RL_BACKGROUND_TRAIN=1` "
            "en wacht minimaal 2 uur. Als het model net geladen is: controleer of "
            "`artifacts/rl/*_models.json` een geldig pad bevat."
        )

    # Reward daalt
    if len(hm) >= 2:
        rewards = [h.get("avg_final_cumulative_reward") for h in hm if h.get("avg_final_cumulative_reward") is not None]
        if len(rewards) >= 3 and all(rewards[i] > rewards[i + 1] for i in range(-3, -1)):
            actions.append(
                "**Reward daalt consistent** — overweeg `learning_rate` te halveren in "
                "`docker-compose.env` (`RL_LEARNING_RATE`). Controleer ook of "
                "`reward_function.py` de beloning niet te zwaar straft voor kleine winsten."
            )

    # Prijs-afwijking
    if any("PRIJS-AFWIJKING" in a for a in alerts):
        actions.append(
            "**Hoge prijs-afwijking** — controleer of de prediction-feed actuele candles "
            "ontvangt (`GET /api/v1/predictions/{market}`). Mogelijk is `_build_predicted_price_fields` "
            "in `main.py` gebaseerd op verouderde OHLCV-data."
        )

    if not actions:
        actions.append(
            "✅ Geen kritieke acties gedetecteerd. "
            "Controleer morgen of `training_chunks_in_window` groeit en reward stabiel blijft."
        )

    return actions[:5]


# ── Render ─────────────────────────────────────────────────────────────────────

def render(rb: dict, chunks: dict, hm: list[dict], mv: list[dict],
           ss: dict, snap: dict, crash_lines: list[str]) -> str:

    alerts = collect_critical_alerts(snap, ss, crash_lines)
    anomalies = detect_anomalies(rb, chunks, hm, ss)
    actions = generate_action_plan(alerts, anomalies, rb, chunks, hm)
    steps = global_steps_growth(chunks)
    acc = prediction_accuracy(snap)
    ts = NOW_LOCAL.strftime("%Y-%m-%d %H:%M Europe/Amsterdam")

    L: list[str] = [f"# AUTO_BRIEFING — {ts}", ""]

    # ── CRITICAL ALERTS (altijd bovenaan) ──
    L.append("## ⚠️ CRITICAL ALERTS")
    L.append("")
    if alerts:
        for a in alerts:
            L.append(f"> 🚨 {a}")
            L.append("")
    else:
        L.append("> ✅ Geen kritieke meldingen.")
        L.append("")

    # ── Global Steps Growth per markt ──
    L += ["---", "", "## Global Steps Groei per Markt", ""]
    pair_growth = chunks.get("pair_growth") or {}
    if pair_growth:
        L.append("| Markt | Max Steps | Δ (20 chunks) | Chunks 24u |")
        L.append("|-------|-----------|---------------|------------|")
        for pair, pg in sorted(pair_growth.items(), key=lambda x: -x[1].get("max_step", 0)):
            delta = pg.get("delta", 0)
            arrow = "📈" if delta > 500 else ("⚠️" if delta < 100 else "→")
            L.append(
                f"| {pair} | {pg.get('max_step', 0):,} | {arrow} {delta:+,} | {pg.get('chunks_24h', 0)} |"
            )
    else:
        # Fallback: totale groei
        if steps.get("growth") is not None:
            L.append(f"- Groei over laatste {len(steps['series'])} chunks: **{steps['growth']:+,} stappen** "
                     f"({steps['first']:,} → {steps['last']:,})")
            growth_per_chunk = steps["growth"] / max(1, len(steps["series"]) - 1)
            L.append(f"- Gemiddeld per chunk: **{growth_per_chunk:.0f} stappen**")
            if steps["growth"] < 1000:
                L.append("- ⚠️ Groei traag — model traint nauwelijks.")
        else:
            L.append("- Onvoldoende data (< 2 chunks beschikbaar).")

    # ── AI Prediction Accuraatheid ──
    L += ["", "## AI Prediction Accuraatheid", ""]
    L.append("*Afwijking = predicted_price[0] t.o.v. huidige prijs. Policy confidence = dominant_policy_prob.*")
    L.append("")
    if acc:
        L.append("| Markt | Voorspeld | Actueel | Afwijking | Confidence |")
        L.append("|-------|-----------|---------|-----------|------------|")
        for market, info in sorted(acc.items()):
            pred = f"{info['predicted']:.2f}" if "predicted" in info else "—"
            actual = f"{info['actual']:.2f}" if "actual" in info else "—"
            dev = f"{info['dev_pct']:+.2f}%" if "dev_pct" in info else "—"
            flag = " ⚠️" if "dev_pct" in info and abs(info["dev_pct"]) > 3 else ""
            conf = f"{info['policy_confidence']:.3f}" if "policy_confidence" in info else "—"
            L.append(f"| {market} | {pred} | {actual} | {dev}{flag} | {conf} |")
    else:
        L.append("- Redis niet beschikbaar of geen prediction data.")

    # ── Learning Plateau ──
    L += ["", "## Learning Plateau Detectie", ""]
    plateau = detect_learning_plateau(chunks, hm)
    if plateau:
        L.append(f"- ⚠️ **Plateau gedetecteerd:** {plateau}")
        L.append("")
        L.append("  **Indicatoren:**")
        if len(hm) >= 4:
            losses = [h.get("avg_loss") for h in hm[-4:] if h.get("avg_loss") is not None]
            if losses:
                L.append(f"  - avg_loss laatste 4 uur: `{losses}`")
    else:
        L.append("- ✅ Geen plateau gedetecteerd op basis van beschikbare data.")

    # ── RL Intelligence Samenvatting ──
    L += ["", "## RL Intelligence Samenvatting", "", "| Metric | Waarde |", "|--------|--------|"]
    L.append(f"| Replay buffer entries | {rb.get('total', '?')} |")
    L.append(f"| Entries met reward_pct | {rb.get('with_reward', '?')} |")
    L.append(f"| Training chunks totaal | {chunks.get('total', '?')} |")
    L.append(f"| Training chunks (24u) | {chunks.get('last_24h', '?')} |")
    lr = chunks.get("last_row")
    L.append(f"| Laatste training | {_iso(lr[0] if lr else None)} |")
    step_fmt = f"{lr[2]:,}" if lr and lr[2] else "—"
    L.append(f"| Global step (laatste) | {step_fmt} |")
    if hm:
        last_h = hm[-1]
        L.append(f"| Avg loss (laatste uur) | {last_h.get('avg_loss', '—')} |")
        L.append(f"| Avg cumulative reward | {last_h.get('avg_final_cumulative_reward', '—')} |")
    probs = rb.get("prob_avg_24h") or (None, None, None)
    if probs[0] is not None:
        L.append(f"| Policy probs buy/hold/sell | {probs[0]:.3f} / {probs[1]:.3f} / {probs[2]:.3f} |")
    sigs = rb.get("signals_24h", {})
    if sigs:
        L.append(f"| Signalen 24u BUY/HOLD/SELL | {sigs.get('BUY',0)} / {sigs.get('HOLD',0)} / {sigs.get('SELL',0)} |")

    # ── Anomalies ──
    L += ["", "## Anomaly Detection", ""]
    if anomalies:
        for a in anomalies:
            L.append(f"- ⚠️ {a}")
    else:
        L.append("- ✅ Geen anomalieën gedetecteerd.")

    # ── Action Plan ──
    L += ["", "## 🎯 Action Plan", "",
          "> **Voer deze acties uit bij het begin van je volgende sessie.**", ""]
    for i, a in enumerate(actions, 1):
        L.append(f"**{i}.** {a}")
        L.append("")

    # ── Model Versies ──
    if mv:
        L += ["## Model Versies", ""]
        for m in mv:
            e = m.get("last") or {}
            L.append(
                f"- **{m['file']}** | ts: `{e.get('timestamp','?')}` "
                f"| reward_score: `{e.get('reward_score','?')}` | `{e.get('model_path','?')}`"
            )

    # ── Reward Trend ──
    reward_hist = [h.get("avg_final_cumulative_reward") for h in hm if h.get("avg_final_cumulative_reward") is not None]
    if len(reward_hist) >= 2:
        L += ["", "## Reward Trend (laatste uurblokken)", "", "```"]
        for h in hm[-10:]:
            rv = h.get("avg_final_cumulative_reward")
            ts_h = _iso(h.get("ts_utc"))
            if rv is not None:
                bar = "█" * max(0, min(40, int((rv + 30))))
                L.append(f"{ts_h}  {rv:+8.2f}  {bar}")
        L.append("```")

    L += ["", "---",
          f"*Gegenereerd: {ts} — `scripts/generate_briefing.py`*",
          f"*Volgende update: over ~1 uur (via `_rl_hourly_checkpoint_and_metrics_loop`)*",
          ""]
    return "\n".join(L)


if __name__ == "__main__":
    rb = replay_stats()
    chunks = training_chunks_stats()
    hm = hourly_metrics()
    mv = model_versions()
    ss = system_state()
    snap = redis_full_snapshot()
    crash_lines = recent_crash_lines()
    content = render(rb, chunks, hm, mv, ss, snap, crash_lines)
    OUT.write_text(content)
    print(f"✅ AUTO_BRIEFING.md geschreven ({len(content)} bytes) → {OUT}")
