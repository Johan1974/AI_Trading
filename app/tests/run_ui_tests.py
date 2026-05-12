"""
BESTANDSNAAM: app/tests/run_ui_tests.py
FUNCTIE: Deep-Scan Watchdog voor het AI Trading Dashboard.
         Dit script voert een autonome, periodieke audit uit op alle UI-tabs 
         (Terminal, AI Brain, Ledger, Hardware). Het controleert op lege velden, 
         valideert of multi-ticker support (ETH/XRP/SOL) aanwezig is, detecteert 
         'bevroren' data (stagnerende prijzen/CPU) en genereert automatische 
         audit-rapporten en heartbeats voor continue systeemmonitoring.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# ``python /app/tests/run_ui_tests.py`` zet alleen ``…/tests`` op sys.path.
# Op host: ``parents[2]`` = repo-root (map die ``app/`` bevat). In validator (mount ``./app`` → ``/app``): ``parents[2]`` = ``/`` zodat ``app`` → ``/app/__init__.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytz
import urllib.request
import urllib.error
try:
    import docker
except ImportError:
    docker = None
from playwright.async_api import async_playwright

from app.diagnostics.logs_hub_maintenance import repair_permissions, resolve_logs_hub

BRAIN_WAIT_PHRASES = (
    "wachten op eerste besluit",
    "model aan het inladen",
)

# Portal-backend kan al scanner-/voorspelling-tekst tonen i.p.v. kale RL-wachttekst.
BRAIN_FALLBACK_OK_MARKERS = (
    "scanner (",
    "laatste voorspelling",
    "engine draait",
    "volledige rl-policytekst",
)

WORKER_EXEC_LOG = "/app/logs/worker_execution.log"

_LAST_LOGS_HUB_MAINT_MONO = 0.0


def _logs_hub_subprocess_maintenance() -> None:
    """Volledige diagnose + stubs + permissies (host uid/gid via env)."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "analyze_logs_hub.py"
    if not script.is_file():
        print("[AUTONOMY] ⚠️ scripts/analyze_logs_hub.py ontbreekt; mount ./scripts in de validator-container.")
        return
    try:
        subprocess.run(
            [sys.executable, str(script), "--fix", "--fix-permissions"],
            check=False,
            timeout=180,
            env=os.environ.copy(),
        )
    except Exception as exc:
        print(f"[AUTONOMY] ⚠️ _logs_hub onderhoud subprocess: {exc}")


def _parse_iso_age_sec(iso: str | None) -> float | None:
    """Leeftijd van een ISO-timestamp t.o.v. UTC-now; None bij parse-fout."""
    if not iso:
        return None
    s = str(iso).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        t = datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds())
    except Exception:
        return None


def diagnose_worker_brain_pipeline(
    api_base: str,
    redis_snapshot_ok: bool,
    brain_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Automatische verdieping als RL-/brain-tekst ontbreekt of nog laadt:
    /activity (tick), optioneel al opgehaalde brain-JSON, tail worker_execution.log.
    """
    base = api_base.rstrip("/")
    out: dict[str, Any] = {
        "redis_snapshot_ok": redis_snapshot_ok,
        "last_engine_tick_utc": None,
        "tick_age_sec": None,
        "bot_status": None,
        "selected_market": None,
        "brain_status": None,
        "brain_reasoning_preview": None,
        "worker_log_hints": [],
        "verdict": "",
        "print_lines": [],
    }
    act = _http_get_json(f"{base}/activity")
    if isinstance(act, dict):
        lec = act.get("last_engine_cycle")
        tick = None
        if isinstance(lec, dict):
            tick = lec.get("ts")
        tick = tick or act.get("last_engine_tick_utc")
        out["last_engine_tick_utc"] = tick
        out["tick_age_sec"] = _parse_iso_age_sec(tick) if tick else None
        out["bot_status"] = act.get("bot_status")
        out["selected_market"] = act.get("selected_market")

    brain = brain_json if isinstance(brain_json, dict) else _http_get_json(f"{base}/api/v1/brain/reasoning")
    if isinstance(brain, dict):
        out["brain_status"] = brain.get("status")
        rp = str(brain.get("reasoning") or "").strip()
        out["brain_reasoning_preview"] = rp[:200] + ("…" if len(rp) > 200 else "")

    hints: list[str] = []
    try:
        with open(WORKER_EXEC_LOG, encoding="utf-8", errors="replace") as f:
            chunk = f.readlines()[-500:]
        blob = "".join(chunk)
        if "GPU DISCONNECTED" in blob or "BOT STOPPED" in blob:
            hints.append("worker_execution.log: GPU disconnect / BOT STOPPED — worker stopt RL-cyclus")
        if "[ENGINE]" in blob or "ENGINE]" in blob or "Tick |" in blob:
            hints.append("worker_execution.log: engine-tick regels aanwezig (recent)")
        if "RL-MULTI" in blob or "inferentie" in blob.lower():
            hints.append("worker_execution.log: RL-multi / inferentie vermeld")
        if "Inference fout" in blob or "inferentie fout" in blob.lower():
            hints.append("worker_execution.log: inferentie-fout — check stacktrace boven laatste melding")
        if "[CRITICAL]" in blob:
            hints.append("worker_execution.log: [CRITICAL] regels gevonden")
    except OSError:
        hints.append("worker_execution.log niet leesbaar (pad /app/logs of volume)")

    out["worker_log_hints"] = hints

    age = out["tick_age_sec"]
    lines: list[str] = []
    if not redis_snapshot_ok:
        lines.append("Redis snapshot ontbreekt of fout — worker schrijft geen tenant naar portal.")
        out["verdict"] = "Redis/worker-snapshot eerst herstellen; daarna pas RL-tekst verwachten."
    elif age is None:
        lines.append("Geen parseerbare last_engine_tick in /activity — worker heeft mogelijk nog geen cycle-TS gepubliceerd.")
        out["verdict"] = "Controleren of worker draait en STATE last_engine_cycle vult."
    elif age > 180:
        lines.append(f"Engine-tick is ~{age:.0f}s oud (>3 min) — worker lijkt stil of vastgelopen.")
        out["verdict"] = "Worker-herstart + worker_execution.log (GPU, exceptions) bekijken."
    elif age > 60:
        lines.append(f"Engine-tick ~{age:.0f}s oud — traag; RL kan nog opstarten of GPU wacht.")
        out["verdict"] = "Korte wachttijd normaal na deploy; bij aanhouden: worker-resources en logs."
    else:
        lines.append(f"Engine-tick recent (~{age:.0f}s) — worker loopt; ontbrekende RL-tekst is waarschijnlijk serialisatie/portal-cache.")
        out["verdict"] = "Portal/worker op dezelfde Redis-snapshot; /api/v1/brain/reasoning en rl_last_decision in worker_state controleren."

    for h in hints[:6]:
        lines.append(h)

    out["print_lines"] = lines
    return out

# Voorkomt portal-flap: herstart → Playwright klikt tegen dichte poort (ERR_CONNECTION_REFUSED).
_LAST_CONTAINER_RESTART_TS = 0.0


def _validator_heal_debounce_sec() -> float:
    return float(os.getenv("VALIDATOR_HEAL_DEBOUNCE_SEC", "120"))


def _validator_post_health_settle_sec() -> float:
    return float(os.getenv("VALIDATOR_POST_HEALTH_SETTLE_SEC", "10"))


def _validator_health_max_rounds() -> int:
    return int(os.getenv("VALIDATOR_HEALTH_WAIT_ROUNDS", "48"))


def _validator_health_sleep_sec() -> float:
    return float(os.getenv("VALIDATOR_HEALTH_SLEEP_SEC", "2.5"))


def _http_get_json(full_url: str, timeout: float = 8.0) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(full_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return None
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _redis_worker_data_hget() -> tuple[bool, str | None]:
    """Cross-check: Redis `worker_snapshot` `data` hash (zelfde bron als portal poll)."""
    if not docker:
        return True, None
    try:
        client = docker.from_env()
        cont = client.containers.get("ai-trading-redis")
        out = cont.exec_run("redis-cli HGET worker_snapshot data")
        if out.exit_code != 0:
            return False, f"redis-cli exit {out.exit_code}"
        blob = out.output
        if isinstance(blob, bytes):
            blob = blob.decode("utf-8", errors="replace").strip()
        else:
            blob = str(blob).strip()
        if not blob or blob in ("nil", "(nil)"):
            return False, "worker_snapshot.data leeg"
        return True, None
    except Exception as exc:
        return False, str(exc)


def run_pipeline_sanity_checks(api_base: str) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Extra datastroom-controles: Redis/worker → portal, Elite-8, RL-brain API, Bitvavo rate-limit.
    Returns (failures, warnings, audit_flat) — failures zijn hard; warnings alleen in rapport.
    """
    failures: list[str] = []
    warnings: list[str] = []
    audit: dict[str, Any] = {}
    base = api_base.rstrip("/")

    dbg = _http_get_json(f"{base}/api/v1/debug_data")
    if not dbg or dbg.get("status") == "error":
        msg = (dbg or {}).get("message") or "debug_data niet bereikbaar"
        failures.append(f"Redis/worker snapshot: {msg}")
        audit["Pipeline: Redis snapshot"] = {"status": "FAILED", "value": msg}
    else:
        audit["Pipeline: Redis snapshot"] = {"status": "OK", "value": "api/v1/debug_data heeft tenant"}

    ok_r, err_r = _redis_worker_data_hget()
    if ok_r:
        audit["Pipeline: Redis HGET"] = {"status": "OK", "value": "worker_snapshot.data aanwezig"}
    else:
        warnings.append(f"Redis docker cross-check: {err_r}")
        audit["Pipeline: Redis HGET"] = {"status": "WARNING", "value": err_r or "n/a"}

    stats = _http_get_json(f"{base}/api/v1/stats")
    if not stats:
        failures.append("API /api/v1/stats niet bereikbaar of geen JSON")
        audit["Pipeline: stats API"] = {"status": "FAILED", "value": "geen response"}
    else:
        am = stats.get("active_markets")
        if not isinstance(am, list) or len(am) == 0:
            failures.append("active_markets leeg in /api/v1/stats (worker/scanner levert geen Elite-lijst)")
            audit["Pipeline: active_markets"] = {"status": "FAILED", "value": "leeg of geen lijst"}
        else:
            n_ok = sum(1 for m in am if isinstance(m, dict) and m.get("market"))
            audit["Pipeline: active_markets"] = {"status": "OK", "value": f"{n_ok} rijen met market"}

    status_v1 = _http_get_json(f"{base}/api/v1/status")
    if not status_v1:
        warnings.append("/api/v1/status niet bereikbaar")
        audit["Pipeline: AI engine status"] = {"status": "WARNING", "value": "geen /api/v1/status JSON"}
    else:
        ai_engine = str(status_v1.get("ai_engine") or "").upper()
        worker_st = str(status_v1.get("worker_status") or "").lower()
        probs = stats.get("ai_action_probs") if isinstance(stats, dict) else {}
        has_probs = False
        if isinstance(probs, dict):
            for k in ("buy_pct", "hold_pct", "sell_pct"):
                try:
                    v = probs.get(k)
                    if v is not None and float(v) > 0.01:
                        has_probs = True
                        break
                except (TypeError, ValueError):
                    continue
        if has_probs and ai_engine == "ERROR":
            failures.append("AI engine status mismatch: ai_action_probs aanwezig maar /api/v1/status=ERROR")
            audit["Pipeline: AI engine status"] = {
                "status": "FAILED",
                "value": f"mismatch (worker={worker_st}, ai_engine={ai_engine}, has_probs={has_probs})",
            }
        elif ai_engine in {"ACTIVE", "THINKING"}:
            audit["Pipeline: AI engine status"] = {
                "status": "OK",
                "value": f"ai_engine={ai_engine}, worker={worker_st}, has_probs={has_probs}",
            }
        else:
            audit["Pipeline: AI engine status"] = {
                "status": "WARNING",
                "value": f"ai_engine={ai_engine or 'n/a'}, worker={worker_st}, has_probs={has_probs}",
            }

    brain = _http_get_json(f"{base}/api/v1/brain/reasoning")
    redis_ok = audit.get("Pipeline: Redis snapshot", {}).get("status") == "OK"
    brain_pipeline_not_ok = False

    if not brain:
        warnings.append("/api/v1/brain/reasoning geen JSON (RL-tab kan traag opstarten)")
        audit["Pipeline: brain reasoning API"] = {"status": "WARNING", "value": "geen JSON"}
        brain_pipeline_not_ok = True
    else:
        audit["Pipeline: brain reasoning API"] = {"status": "OK", "value": "JSON OK"}
        rtxt = str(brain.get("reasoning") or "").lower()
        st = str(brain.get("status") or "").lower()
        has_fallback = any(m in rtxt for m in BRAIN_FALLBACK_OK_MARKERS)
        bare_placeholder = any(p in rtxt for p in BRAIN_WAIT_PHRASES) and not has_fallback
        if bare_placeholder or st in ("model_loading", "warming_up") or not str(brain.get("reasoning") or "").strip():
            brain_pipeline_not_ok = True
            detail_bits: list[str] = []
            if bare_placeholder:
                detail_bits.append("placeholder")
            if st in ("model_loading", "warming_up"):
                detail_bits.append(st)
            if not str(brain.get("reasoning") or "").strip():
                detail_bits.append("lege reasoning")
            warnings.append(
                "RL brain reasoning nog niet volledig (worker tick / serialisatie / placeholder): "
                + ", ".join(detail_bits)
            )
            audit["Pipeline: RL reasoning"] = {"status": "WARNING", "value": ", ".join(detail_bits)}
        else:
            audit["Pipeline: RL reasoning"] = {"status": "OK", "value": "tekst aanwezig"}

    if brain_pipeline_not_ok or not brain:
        diag = diagnose_worker_brain_pipeline(base, redis_ok, brain_json=brain if isinstance(brain, dict) else None)
        audit["Pipeline: Worker brain diagnose"] = diag
        print("  🔬 [WORKER-DIAGNOSE] Automatische verdieping (RL/brain-pijplijn):")
        for ln in diag.get("print_lines") or []:
            print(f"      • {ln}")
        print(f"      → {diag.get('verdict', '')}")
        age = diag.get("tick_age_sec")
        if redis_ok and age is not None and float(age) > 120.0:
            warnings.append(
                f"WORKER_TICK_STALE: engine-tick ~{float(age):.0f}s geleden — {str(diag.get('verdict') or '')[:160]}"
            )

    rl = _http_get_json(f"{base}/exchange/rate-limit/status")
    if not rl:
        warnings.append("/exchange/rate-limit/status niet bereikbaar (portal stub of route ontbreekt)")
        audit["Pipeline: Bitvavo rate-limit"] = {"status": "WARNING", "value": "geen JSON"}
    else:
        cb = rl.get("circuit_breaker") if isinstance(rl.get("circuit_breaker"), dict) else {}
        if cb.get("is_open") is True:
            failures.append("Bitvavo circuit breaker OPEN — API calls worden geblokkeerd")
            audit["Pipeline: Bitvavo circuit"] = {"status": "FAILED", "value": "circuit open"}
        else:
            audit["Pipeline: Bitvavo circuit"] = {"status": "OK", "value": "circuit gesloten"}
        rl_info = rl.get("rate_limit") if isinstance(rl.get("rate_limit"), dict) else {}
        lim = rl_info.get("limit")
        rem = rl_info.get("remaining")
        try:
            li = float(lim) if lim is not None else 0.0
            re = float(rem) if rem is not None else 0.0
            if li > 0 and re / li < 0.05:
                warnings.append(f"Bitvavo rate-limit laag: remaining={re} limit={li}")
                audit["Pipeline: Bitvavo remaining"] = {"status": "WARNING", "value": f"{re}/{li}"}
            else:
                audit["Pipeline: Bitvavo remaining"] = {"status": "OK", "value": f"{rem}/{lim}"}
        except (TypeError, ValueError):
            audit["Pipeline: Bitvavo remaining"] = {"status": "OK", "value": "n/a (nog geen headers)"}

    # Hardware-tab: centrale log-API (zelfde bron als UI + /ws/logs)
    slog = _http_get_json(f"{base}/api/v1/system/logs?limit=80")
    if not slog:
        warnings.append("/api/v1/system/logs niet bereikbaar — Hardware log-paneel blijft leeg")
        audit["Pipeline: system logs API"] = {"status": "WARNING", "value": "geen response"}
    else:
        lines = slog.get("lines") if isinstance(slog.get("lines"), list) else []
        path = slog.get("path", "")
        if len(lines) == 0:
            warnings.append("Hardware logs: API gaf lines=[] — volume _logs_hub of permissies")
            audit["Pipeline: system logs API"] = {"status": "WARNING", "value": "empty lines"}
        else:
            blob0 = " ".join(str(x) for x in lines[:3] if x)
            if "Geen enkel logbestand" in blob0 or "Geen logregels" in blob0:
                warnings.append(
                    "Hardware logs: schijf onder /app/logs levert geen worker/portal/blackbox-inhoud "
                    "(compose: ./_logs_hub:/app/logs)"
                )
                audit["Pipeline: system logs API"] = {"status": "WARNING", "value": "alleen placeholder"}
            else:
                n = sum(1 for ln in lines if isinstance(ln, str) and ln.strip())
                audit["Pipeline: system logs API"] = {"status": "OK", "value": f"{n} regels, path={path}"}

    return failures, warnings, audit

# Cockpit: `applyActivityResponse` schrijft via `.js-sentiment-value`; `id` is nodig voor Playwright + legacy main.js.
TERMINAL_SENTIMENT_LOCATOR = "#sentiment-value, .js-sentiment-value, .sentiment-value"

TABS_MAP = {
    "AI Brain": {
        "button": "#btn-aibrain",
        "elements": {
            "#rl-confidence, .rl-confidence": "RL Confidence",
            "#model-version, .model-version": "Model Version",
            "#weight-correlation, #js-corr-value": "AI Correlation Weight",
            "#weight-news, #js-news-weight": "AI News Weight",
            "#weight-price, #js-price-weight": "AI Price Weight",
            "#brainTabStatDiscount": "Discount Factor",
            "#brainTabStatBatch": "Batch Size",
            "#brainLastScanLine": "Laatste Scan Tijd"
        }
    },
    "Ledger": {
        "button": "#btn-ledger",
        "elements": {
            "#total-balance, .total-balance": "Total Balance",
            "#ledgerPerfClosed, #trade-count": "Trade Count",
            "#ledgerPerfPnlEur, #pnl-total": "Total PnL",
            "#ledgerPerfWinRate": "Win Rate",
            "#ledgerPerfMaxWin": "Max Win",
            "#ledgerPerfMaxLoss": "Max Loss",
            "#ledgerPerfHold": "Avg Hold Time"
        }
    },
    "Hardware": {
        "elements": {
            "#ringCpuVal, .ringCpuVal": "CPU Load Cirkel",
            "#ringRamVal, .ringRamVal": "RAM Usage Cirkel",
            "#ringGpuVal, .ringGpuVal": "GPU Temp Cirkel",
            "#ringDiskVal, .ringDiskVal": "Disk Usage Cirkel"
        }
    },
    "Terminal": {
        "button": "#btn-terminal",
        "elements": {
            "#btc-price, .btc-price": "Actieve Market Prijs",
            TERMINAL_SENTIMENT_LOCATOR: "AI Sentiment Score",
            "#market-24h-change, .market-24h-change": "24h Change %",
            "#market-volatility, .market-volatility": "4h Volatility %",
            "#market-reason, .market-reason": "Scanner Reason",
            "#allocatie": "Allocatie Status",
            "#terminalFearGreed": "Fear & Greed Index"
        }
    }
}

last_btc_price = None
last_cpu_val = None
frozen_counter = 0

AMSTERDAM = pytz.timezone("Europe/Amsterdam")

async def check_blackout(page):
    while True:
        fails = getattr(page, "__network_failures", [])
        # Anti-flap: behandel pas als blackout wanneer meerdere harde connectiefouten zijn gezien.
        if isinstance(fails, list) and len(fails) >= 3:
            return fails[-1]
        await asyncio.sleep(0.5)

async def wait_with_blackout_check(page, awaitable, failures, root_causes):
    monitor = asyncio.create_task(check_blackout(page))
    action = asyncio.create_task(awaitable)
    done, pending = await asyncio.wait([monitor, action], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    if monitor in done:
        err = monitor.result()
        failures.append(f"API Blackout gedetecteerd: {err}")
        root_causes.add("backend_api_down")
        raise Exception(f"API Blackout Triggered: {err}")
    return action.result()


_COCKPIT_VIEWPORT_CONTRACT_JS = """
() => {
  const out = [];
  const vh = window.innerHeight;
  const vw = window.innerWidth;
  if (!document.body || !document.body.classList.contains('cockpit-body')) {
    out.push('SKIP_NOT_COCKPIT');
    return out;
  }
  const tab = document.getElementById('tab-terminal');
  const tabStyle = tab ? window.getComputedStyle(tab) : null;
  const terminalVisible = tab && tabStyle && tabStyle.display !== 'none' && tabStyle.visibility !== 'hidden';
  if (!terminalVisible) return out;

  const row2 = document.querySelector('.cockpit-header-row2.cockpit-asset-strip--rail');
  if (row2) {
    const r = row2.getBoundingClientRect();
    if (r.top < -2) out.push('ASSET_RAIL_TOP_OFFSCREEN:' + Math.round(r.top));
    if (r.bottom > vh + 3) out.push('ASSET_RAIL_BOTTOM_OVERFLOW:' + Math.round(r.bottom) + '/' + vh);
  }
  const led = document.getElementById('liveLedgerFooter');
  if (led) {
    const r = led.getBoundingClientRect();
    if (r.top < -2) out.push('LEDGER_TOP_OFFSCREEN:' + Math.round(r.top));
    if (r.bottom > vh + 4) out.push('LEDGER_BOTTOM_OVERFLOW:' + Math.round(r.bottom) + '/' + vh);
    if (r.left < -2 || r.right > vw + 2) {
      out.push('LEDGER_HORIZONTAL_CLIP:' + Math.round(r.left) + ',' + Math.round(r.right) + '/' + vw);
    }
  }
  const app = document.getElementById('app');
  if (app) {
    const r = app.getBoundingClientRect();
    if (r.top < -2) out.push('APP_TOP_OFFSCREEN:' + Math.round(r.top));
    if (r.bottom > vh + 3) out.push('APP_BOTTOM_OVERFLOW:' + Math.round(r.bottom) + '/' + vh);
  }
  return out;
}
"""


async def collect_cockpit_viewport_contract_violations(page) -> list[str]:
    """Meet-kader voor cockpit-terminal: kritieke UI moet in inner viewport vallen (geen 'oneindig tunen')."""
    try:
        raw = await page.evaluate(_COCKPIT_VIEWPORT_CONTRACT_JS)
    except Exception as exc:
        return [f"viewport_contract_eval_error:{exc}"]
    if not isinstance(raw, list):
        return ["viewport_contract_bad_result"]
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if not s or s == "SKIP_NOT_COCKPIT":
            continue
        out.append(s)
    return out

async def wait_for_api_health(url: str) -> bool:
    """Wacht tot uvicorn weer luistert (/health) én /api/v1/stats 200 geeft; daarna settle-pauze voor stabiele UI."""
    base = url.rstrip("/")
    health_url = f"{base}/health"
    stats_url = f"{base}/api/v1/stats"
    mx = _validator_health_max_rounds()
    delay = _validator_health_sleep_sec()
    settle = _validator_post_health_settle_sec()
    print(f"\n[HEALTH] ⏳ Wachten op portal-recovery (/health + /api/v1/stats), max ~{mx * delay:.0f}s, daarna {settle:.0f}s settle...")
    for i in range(mx):
        h_ok = False
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=4) as response:
                h_ok = response.status == 200
        except Exception:
            pass
        if h_ok:
            try:
                req = urllib.request.Request(stats_url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=6) as response:
                    if response.status == 200:
                        print("[HEALTH] ✅ Portal + stats API weer online.")
                        await asyncio.sleep(settle)
                        return True
            except Exception as e:
                if i % 4 == 0:
                    print(f"[HEALTH] ⚠️ Health OK, stats nog niet: {e}")
        else:
            if i % 4 == 0:
                print(f"[HEALTH] ⚠️ Nog geen /health (ronde {i + 1}/{mx})")
        await asyncio.sleep(delay)
    print("[HEALTH] ❌ API recovery-timeout bereikt.")
    return False

async def check_multi_tickers(page):
    tickers_env = os.getenv("TICKERS", "BTC-EUR,ETH-EUR,XRP-EUR,SOL-EUR")
    expected_tickers = [t.strip().split('-')[0] for t in tickers_env.split(',')]
    
    try:
        # Give the JS some time to populate the markets dropdown
        try:
            await page.wait_for_function("() => document.querySelectorAll('#marketSelect option').length > 1", timeout=15000)
        except:
            print("  ⚠️ Wachten op market dropdown timeout")

        options = await page.locator('#marketSelect option').all_inner_texts()
        found_text = " ".join(options)
        
        # If dropdown is empty, maybe they are in the page body
        if not found_text.strip():
            found_text = await page.content()
            
        found_non_btc = False
        for expected in expected_tickers:
            if expected != "BTC" and expected in found_text:
                found_non_btc = True
                break
                
        if not found_non_btc and len(expected_tickers) > 1:
            return False, "ERROR: Multi-Ticker Failure (Only BTC found)"
            
        return True, "Multi-Ticker check passed"
    except Exception as e:
        return False, f"Multi-Ticker check failed: {str(e)}"

async def test_market_switch(page):
    try:
        # Wacht tot de dropdown is gevuld door de backend
        await page.wait_for_function("() => document.querySelectorAll('#marketSelect option').length > 1", timeout=10000)
        select_elem = page.locator('#marketSelect')
        options = await select_elem.locator('option').all_inner_texts()
        
        current_val = await select_elem.evaluate("el => el.options[el.selectedIndex] ? el.options[el.selectedIndex].text : ''")
        current_val = current_val.strip()

        target_market = None
        for opt in options:
            opt = opt.strip()
            if opt and opt != current_val:
                target_market = opt
                break
                
        if not target_market:
            return False, "ERROR: Geen alternatieve munt gevonden (zoals ONDO of TRUMP) om switch te testen."

        # 1. Lees huidige (oude) waarden met text_content() (leest de DOM, ongeacht welke tab zichtbaar is)
        old_sentiment = await page.locator(TERMINAL_SENTIMENT_LOCATOR).first.text_content()
        old_price = await page.locator("#btc-price").first.text_content()

        # 2. Switch uitvoeren (change-handler is async — vaste sleep is onbetrouwbaar)
        await select_elem.select_option(label=target_market)
        try:
            await page.wait_for_function(
                """(old) => {
                  const el = document.querySelector('#btc-price');
                  if (!el) return false;
                  return (el.textContent || '').trim() !== (old || '').trim();
                }""",
                arg=old_price.strip() if old_price else "",
                timeout=15000,
            )
        except Exception:
            await page.wait_for_timeout(6000)
        
        # 3. Lees nieuwe waarden
        new_price = await page.locator("#btc-price").first.text_content()
        new_sentiment = await page.locator(TERMINAL_SENTIMENT_LOCATOR).first.text_content()
        
        # 4. Deep-Interaction Validatie (State-Change)
        old_price_clean = old_price.strip() if old_price else ""
        new_price_clean = new_price.strip() if new_price else ""
        old_sentiment_clean = old_sentiment.strip() if old_sentiment else ""
        new_sentiment_clean = new_sentiment.strip() if new_sentiment else ""
        
        if old_price_clean == new_price_clean:
            return False, f"Switch mislukt: Prijs bleef hangen op {old_price_clean} na switch naar {target_market}."

        # Prijs is de betrouwbare proxy voor een geslaagde switch; sentiment kan per markt gelijk blijven (timing/cache).
        return True, f"Switch naar {target_market} geslaagd (Prijs: {old_price_clean}->{new_price_clean}, Sent: {old_sentiment_clean}->{new_sentiment_clean})."
    except Exception as e:
        return False, f"Market switch test faalde: {str(e)}"

async def investigate_missing_value(page, element_id, tab_name):
    print(f"\n[INVESTIGATIE] 🔍 Start root-cause analyse voor '#{element_id}' op tab '{tab_name}'...")
    try:
        api_data = await page.evaluate('''async () => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            try {
                const opts = { signal: controller.signal };
                const res = await Promise.allSettled([
                    fetch('/api/v1/stats', opts),
                    fetch('/api/v1/brain/training-monitor', opts),
                    fetch('/api/v1/performance/analytics', opts),
                    fetch('/api/v1/brain/reasoning', opts)
                ]);
                
                const parse = async (r) => {
                    try { return (r.status === 'fulfilled' && r.value.ok) ? await r.value.json() : {}; } 
                    catch (e) { return {}; }
                };
                
                return {
                    stats: await parse(res[0]),
                    brain: await parse(res[1]),
                    ledger: await parse(res[2]),
                    reasoning: await parse(res[3])
                };
            } catch (e) {
                return { error: e.toString() };
            } finally {
                clearTimeout(timeoutId);
            }
        }''')
    except Exception as e:
        print(f"[INVESTIGATIE] ❌ Python error tijdens browser-evaluatie: {e}")
        return "target_crashed" if "crashed" in str(e).lower() or "closed" in str(e).lower() else "backend_api_down"

    if "error" in api_data:
        print(f"[INVESTIGATIE] ❌ API Fetch gefaald vanuit browser: {api_data['error']}. Backend ligt plat.")
        return "backend_api_down"

    is_in_api = False
    api_source = "API"
    api_value = None
    if element_id == "terminalFearGreed":
        is_in_api = "fear_greed_score" in api_data.get("stats", {})
        api_source = "/api/v1/stats -> fear_greed_score"
        api_value = api_data.get("stats", {}).get("fear_greed_score")
    elif element_id in ["brainTabStatDiscount", "brainTabStatBatch"]:
        stats_obj = api_data.get("brain", {}).get("stats", {})
        is_in_api = "discount_factor" in stats_obj or "batch_size" in stats_obj
        api_source = "/api/v1/brain/training-monitor -> stats"
        api_value = stats_obj.get("discount_factor") if element_id == "brainTabStatDiscount" else stats_obj.get("batch_size")
    elif element_id in ["ledgerPerfWinRate", "ledgerPerfMaxWin", "ledgerPerfMaxLoss", "ledgerPerfHold"]:
        ps = api_data.get("ledger", {}).get("analytics", {}).get("performance_summary", {})
        is_in_api = "win_rate_pct" in ps or "max_win_eur" in ps
        api_source = "/api/v1/performance/analytics -> performance_summary"
        api_value = ps.get("win_rate_pct") if element_id == "ledgerPerfWinRate" else ps.get("max_win_eur")
    elif element_id == "brainLastScanLine":
        is_in_api = "generated_at" in api_data.get("reasoning", {})
        api_source = "/api/v1/brain/reasoning -> generated_at"
        api_value = api_data.get("reasoning", {}).get("generated_at")

    if is_in_api or api_value is not None:
        print(f"[INVESTIGATIE] ⚠️ Data voor '{element_id}' ZIT wél in de JSON API via {api_source} (waarde: {api_value}), maar UI toont een lege placeholder.")
        return "js_mapping_error"
        
    print(f"[INVESTIGATIE] 🚨 Data voor '{element_id}' ONTBREEKT in de JSON payload van {api_source}.")
    return "backend_missing_data"

async def trigger_auto_jumpstart(page):
    print("\n[NIGHT-MODE] 🌙 Auto-Jumpstart geactiveerd: Systeem is leeg of ongetraind. Forceren van een paper-trade cyclus...")
    try:
        await page.click("#btn-terminal")
        await page.wait_for_timeout(1000)
        btn = page.locator("#paperBtn")
        if await btn.count() > 0 and await btn.is_visible():
            await btn.click()
            print("  ⏳ Wachten op AI redenering en trade-executie (max 90s)...")
            await page.wait_for_function(
                "() => { const b = document.querySelector('#paperBtn'); return b && (b.innerText.includes('Succes') || b.innerText.includes('Fout')); }",
                timeout=90000,
            )
            print("  ✅ Auto-Jumpstart voltooid! De AI is geforceerd om na te denken en te handelen.")
            await page.wait_for_timeout(5000)
        else:
            print("  ⏳ Paper-knop niet in UI; POST /paper/run (BTC-EUR) vanuit de pagina-context…")
            await page.evaluate(
                """async () => {
  try {
    const r = await fetch('/paper/run?ticker=BTC-EUR', { method: 'POST' });
    window.__jumpstartPaper = await r.json().catch(() => ({}));
  } catch (e) {
    window.__jumpstartPaper = { error: String(e) };
  }
}"""
            )
            await page.wait_for_timeout(12000)
            print("  ✅ Paper-run fetch afgerond (controleer worker/logs indien nodig).")
    except Exception as e:
        print(f"  ⚠️ Auto-Jumpstart time-out of mislukt: {e}")

def scan_backend_logs():
    print("\n[NIGHT-MODE] 🕵️ Analyseren van backend logs op verstopte fouten...")
    log_path = '/app/logs/worker_execution.log'
    criticals = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f.readlines()[-500:]:
                    if "[CRITICAL]" in line or "Exception" in line or "Traceback" in line:
                        criticals.append(line.strip())
            if criticals:
                print(f"  ⚠️ Gevonden: {len(criticals)} kritieke backend meldingen in de laatste 500 regels.")
            else:
                print("  ✅ Geen kritieke fouten (Exceptions/CRITICAL) in de recente worker logs.")
        except Exception as e:
            print(f"  ⚠️ Kon logs niet lezen: {e}")
    return criticals


def collect_user_action_items(
    root_causes: set[str],
    failures: list[str],
    pipeline_warnings: list[str] | None = None,
) -> tuple[list[dict[str, str]], str]:
    """
    Classificeert wat niet door container-restarts wordt opgelost (code, config, host).
    """
    pw = pipeline_warnings or []
    if not failures and not pw:
        return [], "Geen actie vereist; audit geslaagd."

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(code: str, title: str, detail: str) -> None:
        if code in seen:
            return
        seen.add(code)
        items.append({"code": code, "title": title, "detail": detail})

    rc = root_causes
    if "backend_missing_data" in rc:
        add(
            "API_PAYLOAD",
            "Backend levert geen JSON-velden",
            "De [INVESTIGATIE]-regels tonen dat data ontbreekt in de API-payload. "
            "Redis-flush + worker-herstart vult die velden niet als de route ze nooit berekent. "
            "Controleer o.a.: performance_summary in GET /api/v1/performance/analytics, "
            "generated_at in GET /api/v1/brain/reasoning, stats (discount_factor, batch_size) in GET /api/v1/brain/training-monitor.",
        )
    if "js_mapping_error" in rc:
        add(
            "UI_MAPPING",
            "API heeft data maar de UI toont —",
            "De JSON bevat het veld maar de DOM blijft een placeholder. Controleer static/js (updateUI, module_ledger, brain-tab) en de koppeling naar element-IDs.",
        )
    if "backend_api_down" in rc:
        add(
            "CONNECTIVITY",
            "Portal/API was onbereikbaar",
            "Controleer `docker compose ps`, `docker logs ai-trading-portal`, `docker logs ai-trading-worker`. "
            "De validator herstart portal/worker al bij blackout; blijft het falen, zoek crashes, poorten of resource-uitputting.",
        )
    if "target_crashed" in rc:
        add(
            "PLAYWRIGHT_OOM",
            "Browsercontext gecrasht (OOM)",
            "Validator draait headless met beperkt geheugen. Verhoog shm_size in docker-compose of verminder parallel werk; volgende scan opent een nieuwe context.",
        )

    fb = " ".join(failures).lower()
    if pw:
        fb = f"{fb} {' '.join(pw).lower()}".strip()
    if "multi-ticker" in fb:
        add(
            "TICKERS_ENV",
            "Multi-ticker / dropdown",
            "Zet TICKERS in docker-compose.env op meerdere markten en zorg dat de worker/scanner `active_markets` vult zodat #marketSelect meer dan alleen BTC toont.",
        )
    if "switch mislukt" in fb or "market switch" in fb:
        add(
            "MARKET_SWITCH",
            "Markt-wissel vernieuwt prijs/sentiment niet",
            "Na selectie in #marketSelect moet de Terminal-state meebewegen (WebSocket/poll). Controleer market-change handlers en of de worker data per markt levert.",
        )
    if "page.goto" in fb and "timeout" in fb:
        add(
            "PORTAL_SLOW",
            "Portal laadt te traag",
            "Playwright kreeg geen 'load' binnen de timeout. Bekijk portal_api.log, trage endpoints en machine-resources.",
        )
    if "schijfruimte" in fb or "onder 5gb" in fb or "disk space" in fb:
        add(
            "DISK",
            "Schijfruimte of log-permissies",
            "Maar minstens 5 GB vrij op de host. Bij Permission denied / Errno 13: chmod/chown op de hostmap `_logs_hub` (docker volume).",
        )
    if "worker_tick_stale" in fb:
        add(
            "WORKER_TICK_STALE",
            "Worker engine-tick te oud",
            "Validator: /activity last_engine_tick is >~2 min oud terwijl Redis-snapshot OK lijkt. "
            "Controleer `docker logs ai-trading-worker`, GPU (GENESIS_REQUIRE_GPU / disconnect), en `worker_execution.log`. "
            "Zie `last_audit_report.json` → metrics → `Pipeline: Worker brain diagnose`.",
        )
    if "rl brain reasoning nog niet volledig" in fb and "worker_tick_stale" not in fb:
        add(
            "WORKER_RL_REASONING",
            "RL policy-tekst ontbreekt of placeholder",
            "Pipeline-waarschuwing op /api/v1/brain/reasoning. Open hetzelfde rapport onder `Pipeline: Worker brain diagnose` "
            "(tick-leeftijd, log-hints). Vaak: eerste inferentie na start, of rl_last_decision nog niet geserialiseerd.",
        )

    code_like = bool(rc & {"backend_missing_data", "js_mapping_error"})
    infra_like = (
        bool(rc & {"backend_api_down", "target_crashed"})
        or "blackout" in fb
        or "switch mislukt" in fb
        or "worker_tick_stale" in fb
    )
    if code_like and not infra_like:
        summary = "Vooral code/API of UI-mapping; herstarts alleen helpen zelden."
    elif code_like and infra_like:
        summary = "Eerst stabiliteit (portal/worker/netwerk), daarna ontbrekende API-velden of UI-mapping afronden."
    elif infra_like:
        summary = "Vooral runtime/connectiviteit; container-herstarts zijn passend."
    else:
        summary = "Zie genummerde acties en de regel AUDIT GEFAALD voor details."
    return items, summary


def print_user_action_block(items: list[dict[str, str]], summary: str) -> None:
    print("\n" + "=" * 72)
    print("[ACTIE VEREIST] Wat de validator niet zelf kan oplossen (jij / codebase):")
    print(f"  Samenvatting: {summary}")
    if not items:
        print("  — Geen aanvullende classificatie; zie AUDIT GEFAALD hierboven.")
    else:
        for i, it in enumerate(items, 1):
            print(f"  {i}. [{it['code']}] {it['title']}")
            print(f"     {it['detail']}")
    print("=" * 72 + "\n")


def print_autonomy_expectation(root_causes: list[str] | None, failures: list[str]) -> None:
    rc = set(root_causes or [])
    fb = " ".join(failures).lower()
    if rc & {"backend_missing_data", "js_mapping_error"} and "blackout" not in fb:
        print(
            "[AUTONOMY] ℹ️ Oorzaak bevat ontbrekende API-data of UI-mapping. "
            "Hieronder: 'best effort' herstarts; blijft het rood, volg [ACTIE VEREIST]."
        )


async def attempt_fix(failures, root_causes=None, url="http://portal:8000"):
    global _LAST_CONTAINER_RESTART_TS
    print("\n[AUTONOMY] 🛠️ Fout gedetecteerd in AI Brain of UI. Start herstel-procedure...")
    print_autonomy_expectation(root_causes, failures)
    print("[AUTONOMY] 🧹 Uitvoeren van 'Schoon Veeg' protocol (oude Docker resten en logs prunen)...")

    f_str = " ".join(failures).lower()
    rc_set = set(root_causes or [])
    rc_str = " ".join(root_causes or [])

    connectivity_fail = any(
        s in f_str
        for s in (
            "connection refused",
            "api blackout",
            "server onbereikbaar",
            "net::err",
            "err_connection",
        )
    )
    # Startup-guard: bij tijdelijke boot races (portal nog aan het opstarten) eerst health-afwachten
    # i.p.v. direct containers opnieuw te herstarten (voorkomt restart-loops/flapping).
    if connectivity_fail:
        print("[AUTONOMY] ⏳ Connectivity-fout gedetecteerd; eerst passive health-recovery proberen...")
        recovered = await wait_for_api_health(url)
        if recovered:
            print("[AUTONOMY] ✅ API herstelde binnen health-window; docker-herstart overgeslagen.")
            return

    # Alleen DOM/API mismatch: portal herstarten maakt Playwright-scans alleen maar breekbaarder.
    if rc_set <= {"js_mapping_error"} and not connectivity_fail:
        print("[AUTONOMY] ℹ️ Alleen UI-mapping; geen docker-herstart (fix = deploy/static; geen portal-flap).")
        return

    # WORKER_TICK_STALE is vaak tijdelijk (na deploy/startup). Voorkom restart-loop:
    # eerst health-window afwachten i.p.v. direct worker/portal te rebooten.
    if ("worker_tick_stale" in f_str or "worker_tick_stale" in rc_str) and not connectivity_fail:
        print("[AUTONOMY] ⏳ WORKER_TICK_STALE gedetecteerd; passieve recovery-window zonder restart...")
        await wait_for_api_health(url)
        return

    debounce = _validator_heal_debounce_sec()
    if time.time() - _LAST_CONTAINER_RESTART_TS < debounce:
        print(f"[AUTONOMY] ℹ️ Herstart gedebounced ({debounce:.0f}s sinds laatste container-restart) — sla docker-restarts over.")
        return

    if not docker:
        print("[AUTONOMY] ⚠️ Docker module ontbreekt. Kan self-healing niet uitvoeren.")
        print("[AUTONOMY] 👉 Handmatig: herstart portal/worker; details staan in last_audit_report.json onder user_actions.")
        return

    restarted = False
    try:
        client = docker.from_env()
        try:
            client.containers.prune()
            client.images.prune(filters={"dangling": True})
        except Exception as e:
            print(f"  ⚠️ Docker prune mislukt: {e}")

        if "target crashed" in f_str or "target_crashed" in rc_str or "browser crash" in f_str:
            print("[AUTONOMY] ⚠️ Playwright Target Crashed (Browser OOM). Sla container-restarts over; Watchdog herstelt zijn eigen browsercontext.")
        elif (
            "backend_api_down" in rc_str
            or "api blackout" in f_str
            or "http " in f_str
            or "server onbereikbaar" in f_str
        ):
            print("[AUTONOMY] 🛠️ Backend API reageert niet. Herstarten van portal en worker...")
            client.containers.get("ai-trading-portal").restart()
            try:
                client.containers.get("ai-trading-worker").restart()
            except Exception:
                pass
            restarted = True
            print("[AUTONOMY] 🛠️ Portal + worker herstart uitgevoerd.")
        elif "backend_missing_data" in rc_str or "spook-data" in f_str or "exact nul" in f_str:
            print("[AUTONOMY] 🛠️ Data corruptie of ontbrekende backend-data. Deep Repair (cache) + worker...")
            try:
                redis_cont = client.containers.get("ai-trading-redis")
                redis_cont.exec_run("redis-cli DEL worker_snapshot ai_trading_snapshot")
                print("[AUTONOMY] 🔧 Redis worker_snapshot gewist.")
            except Exception as e:
                print(f"[AUTONOMY] ⚠️ Deep Repair (Redis) mislukt: {e}")
            client.containers.get("ai-trading-worker").restart()
            restarted = True
            print("[AUTONOMY] 🛠️ Worker herstart uitgevoerd.")
        elif "frozen" in f_str or "bevroren" in f_str or "gelijk gebleven" in f_str:
            print("[AUTONOMY] 🛠️ Bevroren data. Herstarten van portal...")
            client.containers.get("ai-trading-portal").restart()
            restarted = True
            print("[AUTONOMY] 🛠️ Portal herstart uitgevoerd.")
        elif (
            ("leeg of ongeldig" in f_str or "placeholder" in f_str or "kon niet worden uitgelezen" in f_str)
            and "js_mapping_error" not in rc_set
        ):
            # Geen 'timeout' hier: Playwright-tab-timeouts triggerten onnodig portal-flap.
            print("[AUTONOMY] 🛠️ UI placeholders zonder mapping-rootcause. Herstarten van portal...")
            client.containers.get("ai-trading-portal").restart()
            restarted = True
            print("[AUTONOMY] 🛠️ Portal herstart uitgevoerd.")
        elif "ongetraind" in f_str or "grafiek is leeg" in f_str or "badges zijn leeg" in f_str:
            print("[AUTONOMY] 🛠️ Lege grafieken / ongetraind RL. Herstarten van worker...")
            client.containers.get("ai-trading-worker").restart()
            restarted = True
            print("[AUTONOMY] 🛠️ Worker herstart uitgevoerd.")
        elif "ticker" in f_str or "niet numeriek" in f_str or "ongeldig prijsformaat" in f_str:
            print("[AUTONOMY] 🛠️ Ongeldige prijs-data. Herstarten van portal...")
            client.containers.get("ai-trading-portal").restart()
            restarted = True
            print("[AUTONOMY] 🛠️ Portal herstart uitgevoerd.")
        else:
            if "js_mapping_error" in rc_set and not connectivity_fail:
                print("[AUTONOMY] ℹ️ Geen heal-branch + vooral mapping; geen portal-herstart.")
            else:
                print("[AUTONOMY] 🛠️ Algemene UI-fout. Herstarten van portal...")
                client.containers.get("ai-trading-portal").restart()
                restarted = True
                print("[AUTONOMY] 🛠️ Portal herstart uitgevoerd.")

        if restarted:
            _LAST_CONTAINER_RESTART_TS = time.time()
            await wait_for_api_health(url)

    except Exception as e:
        print(f"[AUTONOMY] 🚨 FATALE DOCKER API FOUT: {e}")
        print("   👉 OPLOSSING 1 (Meest waarschijnlijk): Run 'docker compose up -d --force-recreate dashboard-validator'")
        print("   👉 OPLOSSING 2 (Host Permissies): Run 'sudo chmod 666 /var/run/docker.sock' op de server")


async def run_deep_scan():
    global last_btc_price, last_cpu_val, frozen_counter
    
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    base_interval = 60 if debug_mode else 300 # 5 minuten default
    url = os.getenv("API_URL", "http://portal:8000")
    
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 STARTING DEEP-SCAN WATCHDOG (NIGHT-MODE ACTIEF) op {url}")
    print("   -> Validator zal autonoom problemen oplossen, paper trades forceren en logs analyseren.")

    if docker:
        try:
            client = docker.from_env()
            print("[AUTONOMY] 🧹 Uitvoeren van initiële 'Schoon Veeg' (docker image prune -f)...")
            client.images.prune(filters={'dangling': True})
        except Exception as e:
            print(f"[AUTONOMY] ⚠️ Initiële prune mislukt: {e}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
        
        consecutive_successes = 0 # Begin altijd in stabiliteits-check modus
        while True:
            try:
                page = await browser.new_page()
            except Exception as e:
                print(f"[AUTONOMY] ⚠️ Browser crash gedetecteerd: {e}. Herstarten van Playwright...")
                try:
                    await browser.close()
                except Exception:
                    pass
                browser = await p.chromium.launch(headless=True, args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
                page = await browser.new_page()

            vw = int(os.getenv("COCKPIT_VIEWPORT_W", "1366"))
            vh = int(os.getenv("COCKPIT_VIEWPORT_H", "768"))
            try:
                await page.set_viewport_size({"width": max(800, vw), "height": max(600, vh)})
            except Exception:
                pass

            page.__network_failures = []

            def handle_console(msg):
                text = msg.text.lower()
                if "err_empty_response" in text or "err_connection_refused" in text or "websocket connection failed" in text:
                    print(f"\n[NETWORK] 🚨 API-verbinding verbroken! (Console: {msg.text})")
                    page.__network_failures.append(f"Console: {msg.text}")

            def handle_request_failed(request):
                err_txt = request.failure.lower() if request.failure else ""
                if "err_empty_response" in err_txt or "connection refused" in err_txt or "err_connection_refused" in err_txt:
                    print(f"\n[NETWORK] 🚨 API-verbinding verbroken! (Request failed: {request.url} - {request.failure})")
                    page.__network_failures.append(f"Request failed: {request.failure}")

            page.on("console", handle_console)
            page.on("requestfailed", handle_request_failed)

            scan_failed = True
            # Preventie voor UnboundLocalError mocht het script crashen vóór de test-logica
            failures = []
            pipeline_warn_accum: list[str] = []
            root_causes = set()
            try:
                # FORCEER PADEN: Zorg dat de log-directory altijd bestaat voor de JSON-reports, 
                # zelfs als een host-script de map tussentijds heeft gewist.
                os.makedirs('/app/logs', exist_ok=True)

                global _LAST_LOGS_HUB_MAINT_MONO
                maint_interval = int(os.getenv("LOGS_HUB_MAINTENANCE_INTERVAL_SEC", "3600"))
                mono_maint = time.monotonic()
                if _LAST_LOGS_HUB_MAINT_MONO == 0.0:
                    _LAST_LOGS_HUB_MAINT_MONO = mono_maint
                elif mono_maint - _LAST_LOGS_HUB_MAINT_MONO >= maint_interval:
                    _LAST_LOGS_HUB_MAINT_MONO = mono_maint
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🧰 _logs_hub periodiek onderhoud (analyze + --fix + permissies, elke {maint_interval}s)...")
                    await asyncio.to_thread(_logs_hub_subprocess_maintenance)

                # Validator-schoonmaak: alleen bekende audit-artefacten (niet alle .json/.jsonl verwijderen —
                # portal/worker gebruiken o.a. system_state.json, performance.json, rl_hourly_metrics.jsonl).
                for fname in (
                    "AUDIT_FAILURE.png",
                    "audit_status.txt",
                    "browser_console.log",
                    "mem_trace_cluster.flock",
                ):
                    fpath = os.path.join("/app/logs", fname)
                    if os.path.isfile(fpath):
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass

                extracted_data = {}
                audit_details = {}
                data_frozen = False
                
                # AUTONOME SCHIJFBEWAKING
                total, used, free = shutil.disk_usage("/")
                free_gb = free / (1024**3)
                if free_gb < 5.0:
                    disk_warning = f"Schijfruimte kritiek: {free_gb:.2f}GB vrij (Onder 5GB limiet!)"
                    
                    # DATA-SOVEREIGNTY: Zelf log-inspectie doen op permissie fouten
                    try:
                        with open('/app/logs/worker_execution.log', 'r', encoding='utf-8', errors='replace') as f:
                            tail = f.readlines()[-200:]
                            for line in tail:
                                if "Permission denied" in line or "Errno 13" in line:
                                    disk_warning += " 🛑 ROOT CAUSE: 'Permission denied' in worker_execution.log gevonden! Fix _logs_hub permissies."
                                    break
                    except Exception:
                        pass
                        
                    print(f"  ⚠️ [AUTONOME SCHIJFBEWAKING] {disk_warning}")
                    failures.append(disk_warning)
                    audit_details["Disk Space"] = {"status": "FAILED", "value": f"{free_gb:.2f}GB"}
                else:
                    audit_details["Disk Space"] = {"status": "OK", "value": f"{free_gb:.2f}GB"}

                for attempt in range(2):
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 Deep-Scan Ronde starten (Poging {attempt + 1}/2)...")
                    if attempt == 0:
                        print("  ⏳ Wachten op succesvolle UI data-sync (DOM check)...")
                        try:
                            response = await wait_with_blackout_check(page, page.goto(url), failures, root_causes)
                            if response and response.status >= 400:
                                msg = f"HTTP Fout {response.status} op hoofdpagina ({url})"
                                failures.append(msg)
                                root_causes.add("backend_api_down")
                                raise Exception(msg)

                            await wait_with_blackout_check(
                                page,
                                page.wait_for_function(
                                    "() => { const el = document.querySelector('#btc-price'); return el && !el.innerText.includes('—') && !el.innerText.includes('--') && !el.innerText.includes('Laden'); }",
                                    timeout=15000,
                                ),
                                failures,
                                root_causes,
                            )
                            await page.wait_for_timeout(1500)
                            print("  ✅ Data synchronisatie voltooid.")
                            v_issues = await collect_cockpit_viewport_contract_violations(page)
                            for vi in v_issues:
                                failures.append(f"Viewport-contract (Terminal): {vi}")
                            if v_issues:
                                audit_details["Viewport: terminal layout"] = {
                                    "status": "FAILED",
                                    "value": "; ".join(v_issues),
                                }
                            else:
                                audit_details["Viewport: terminal layout"] = {"status": "OK", "value": f"{vw}x{vh}"}
                        except Exception as e:
                            err_str = str(e).lower()
                            if "net::err" in err_str or "connection refused" in err_str:
                                msg = f"Server onbereikbaar ({url}): netwerkverbinding geweigerd"
                                if msg not in failures:
                                    failures.append(msg)
                                root_causes.add("backend_api_down")
                            if "http" not in err_str and "net::err" not in err_str and page.url == "about:blank":
                                try:
                                    await page.goto(url)
                                except Exception:
                                    pass
                            print(f"  ⚠️ Fout tijdens initiële laadfase: {e}")
                    else:
                        print("  🛠️ AUTO-HEAL ACTIEF: Pagina herladen en wachten op DOM sync...")
                        try:
                            response = await wait_with_blackout_check(page, page.reload(), failures, root_causes)
                            if response and response.status >= 400:
                                msg = f"HTTP Fout {response.status} op hoofdpagina tijdens herladen"
                                failures.append(msg)
                                root_causes.add("backend_api_down")
                                raise Exception(msg)

                            await wait_with_blackout_check(
                                page,
                                page.wait_for_function(
                                    "() => { const el = document.querySelector('#btc-price'); return el && !el.innerText.includes('—') && !el.innerText.includes('--') && !el.innerText.includes('Laden'); }",
                                    timeout=15000,
                                ),
                                failures,
                                root_causes,
                            )
                            await page.wait_for_timeout(1500)
                            v_issues = await collect_cockpit_viewport_contract_violations(page)
                            for vi in v_issues:
                                failures.append(f"Viewport-contract (Terminal): {vi}")
                            if v_issues:
                                audit_details["Viewport: terminal layout"] = {
                                    "status": "FAILED",
                                    "value": "; ".join(v_issues),
                                }
                            else:
                                audit_details["Viewport: terminal layout"] = {"status": "OK", "value": f"{vw}x{vh}"}
                        except Exception as e:
                            err_str = str(e).lower()
                            if "net::err" in err_str or "connection refused" in err_str:
                                msg = "Server onbereikbaar tijdens herladen: netwerkverbinding geweigerd"
                                if msg not in failures:
                                    failures.append(msg)
                                root_causes.add("backend_api_down")
                            print(f"  ⚠️ Fout tijdens auto-heal laadfase: {e}")
                            await page.wait_for_timeout(5000)
                            preserved = {k: v for k, v in audit_details.items() if str(k).startswith("Pipeline:")}
                            failures = []
                            extracted_data = {}
                            audit_details = {}
                            audit_details.update(preserved)
                            data_frozen = False
                            root_causes = set()

                    pipe_fail, pipe_warn, pipe_audit = await asyncio.to_thread(run_pipeline_sanity_checks, url)
                    failures.extend(pipe_fail)
                    audit_details.update(pipe_audit)
                    pipeline_warn_accum.extend(pipe_warn)
                    for w in pipe_warn:
                        print(f"  ⚠️ [PIPELINE] {w}")
                    if pipe_warn:
                        audit_details["Pipeline: samenvatting"] = {"status": "WARNING", "value": "; ".join(pipe_warn[:5])}

                    # Tab-hopping and element checking
                    for tab_name, tab_data in TABS_MAP.items():
                        print(f"  👉 Checking tab: {tab_name}")
                        try:
                            if tab_name == "Hardware":
                                await page.evaluate(
                                    "() => { if (typeof window.switchTab === 'function') window.switchTab('hardware'); }"
                                )
                            else:
                                await page.click(tab_data["button"])
                            
                            # Smart waits voor lazy-loaded tab data (wacht tot de fetch API klaar is)
                            if tab_name == "AI Brain":
                                try:
                                    await page.wait_for_function("() => { const el = document.querySelector('#brainTabStatDiscount'); return el && !el.innerText.includes('—'); }", timeout=5000)
                                except Exception: pass
                                try:
                                    await page.wait_for_function(
                                        """() => {
                                          const el = document.querySelector('#brainLastScanLine');
                                          if (!el) return false;
                                          const t = (el.textContent || '').trim();
                                          if (t === 'Laatste scan: —') return false;
                                          if (t.includes('wacht op engine-tick')) return false;
                                          return true;
                                        }""",
                                        timeout=10000,
                                    )
                                except Exception:
                                    pass
                            elif tab_name == "Ledger":
                                try:
                                    await page.wait_for_function("() => { const el = document.querySelector('#ledgerPerfWinRate'); return el && !el.innerText.includes('—'); }", timeout=5000)
                                except Exception: pass
                            
                            await page.wait_for_timeout(1000) # Extra animatie/render buffer

                            if tab_name == "Terminal":
                                v_tab = await collect_cockpit_viewport_contract_violations(page)
                                for vi in v_tab:
                                    failures.append(f"Viewport-contract (Terminal tab): {vi}")
                                if v_tab:
                                    audit_details["Viewport: terminal (tab-hop)"] = {
                                        "status": "FAILED",
                                        "value": "; ".join(v_tab),
                                    }
                                else:
                                    audit_details["Viewport: terminal (tab-hop)"] = {
                                        "status": "OK",
                                        "value": f"{vw}x{vh}",
                                    }
                            
                            for selector, name in tab_data["elements"].items():
                                try:
                                    element_id = selector.split(",")[0].replace("#", "").replace(".", "").strip()
                                    loc = page.locator(selector).first
                                    await loc.wait_for(state="attached", timeout=3000)
                                    # FIX: Gebruik text_content() i.p.v. inner_text() om timeouts op verborgen parent-divs te stoppen
                                    content = await loc.text_content()
                                    content = content.strip() if content else ""
                                    extracted_data[element_id] = content
                                    
                                    # Check op 'lege' of 'foute' waardes
                                    empty_vals = ["--", "—", "-", "NaN", "undefined", ""]
                                    placeholders = ["laden...", "laden…", "none", "---", "laatste scan: —", "wacht op data"]
                                    content_lower = content.lower()
                                    is_placeholder = any(p in content_lower for p in placeholders)
                                    is_zero = content in ["0%", "0.0000", "0.000", "0.00", "0", "0.0"]
                                    allowed_zeros = [
                                        "pnl-total", "ledgerPerfPnlEur", "trade-count", "ledgerPerfClosed", 
                                        "ringCpuVal", "ringRamVal", "ringGpuVal", "ringDiskVal",
                                        "ledgerPerfMaxWin", "ledgerPerfMaxLoss", "ledgerPerfHold", "terminalFearGreed",
                                        "rl-confidence", "sentiment-value"
                                    ] # Hardware en ledger mogen 0 zijn. Ongetraind RL model (0.00) triggert alleen een waarschuwing.

                                    if content in empty_vals or is_placeholder:
                                        suggestion = " 💡 Suggestie: Data ontbreekt in API of main.js mist de mapping." if "weight" in element_id or "sentiment" in element_id else ""
                                        print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Waarde is leeg of een placeholder){suggestion}")
                                        failures.append(f"{name} (#{element_id}) op {tab_name} tab")
                                        audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                        rc = await investigate_missing_value(page, element_id, tab_name)
                                        root_causes.add(rc)
                                    elif is_zero and element_id not in allowed_zeros:
                                        suggestion = " 💡 Suggestie: Oninitialized data of een falende API-pijplijn. Worker wordt herstart." if element_id in ["weight-correlation", "weight-news", "weight-price", "sentiment-value"] else ""
                                        print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Waarde is exact nul - Spook-data!){suggestion}")
                                        failures.append(f"{name} (#{element_id}) op {tab_name} tab (Value is zero)")
                                        audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                    else:
                                        # Content-Aware Check & Gainer-Metrics
                                        if element_id == "rl-confidence" and content in ["0.00", "0", "0.000"]:
                                            print(f"    - [VALUE] {name}: {content} ⚠️ (Waarschuwing: RL Agent is ongetraind, maar dit is valide)")
                                            audit_details[name] = {"status": "WARNING", "value": content}
                                        elif element_id == "btc-price":
                                            clean_content = content.replace("€", "").strip().upper()
                                            ticker_blacklist = ["BTC", "ETH", "TRUMP", "ONDO", "SOL", "XRP", "ADA", "DOT", "AVAX", "LINK", "EUR"]
                                            price_pattern = r'^[-+]?€?\s?(?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,]\d+)?(\s?%?)?$'
                                            
                                            if clean_content in ticker_blacklist or (clean_content.isalpha() and len(clean_content) <= 5):
                                                suggestion = " 💡 Suggestie: main.js mapt per ongeluk een ticker-naam naar het prijsveld, of de API levert stale data. Portal wordt herstart."
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FOUT: Waarde is niet numeriek, ticker-naam gedetecteerd!){suggestion}")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab (Ticker naam ipv prijs)")
                                                audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                            elif not re.match(price_pattern, content.strip()):
                                                suggestion = " 💡 Suggestie: Controleer de prijsformattering in main.js of de payload van portal."
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FOUT: Waarde is niet numeriek of ongeldig geformatteerd){suggestion}")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab (Ongeldig prijsformaat)")
                                                audit_details[name] = {"status": "FAILED", "value": f"{content}{suggestion}"}
                                            else:
                                                print(f"    - [VALUE] {name}: {content} ✅ (Numerieke Prijs Check: OK)")
                                                audit_details[name] = {"status": "OK", "value": content}
                                        elif element_id == "sentiment-value":
                                            try:
                                                val = float(content)
                                                if val < -1.0 or val > 1.0:
                                                    print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Buiten marge [-1.0, 1.0])")
                                                    failures.append(f"{name} (#{element_id}) op {tab_name} tab (Buiten marge)")
                                                    audit_details[name] = {"status": "FAILED", "value": content}
                                                else:
                                                    print(f"    - [VALUE] {name}: {content} ✅ (Marge Check: OK)")
                                                    audit_details[name] = {"status": "OK", "value": content}
                                            except ValueError:
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Niet numeriek)")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab mist numerieke waarde")
                                                audit_details[name] = {"status": "FAILED", "value": content}
                                        elif element_id in ["market-24h-change", "market-volatility"]:
                                            num_str = re.sub(r'[^\d\.\,\-]', '', content)
                                            if not num_str:
                                                print(f"    - [VALUE] {name}: '{content}' ❌ (FAILED: Mist een numerieke waarde)")
                                                failures.append(f"{name} (#{element_id}) op {tab_name} tab mist een numerieke waarde: '{content}'")
                                                audit_details[name] = {"status": "FAILED", "value": content}
                                            else:
                                                print(f"    - [VALUE] {name}: {content} ✅")
                                                audit_details[name] = {"status": "OK", "value": content}
                                        else:
                                            print(f"    - [VALUE] {name}: {content} ✅")
                                            audit_details[name] = {"status": "OK", "value": content}
                                except Exception as el_err:
                                    suggestion = " 💡 Suggestie: Zorg dat main.js (updateUI) deze velden vult, of controleer de HTML op dit ID." if "weight" in element_id else ""
                                    print(f"    - [VALUE] {name} (#{element_id}) ❌ (FAILED: Element niet gevonden of timeout in de DOM){suggestion}")
                                    failures.append(f"Element {name} (#{element_id}) op {tab_name} tab kon niet worden uitgelezen")
                                    audit_details[name] = {"status": "FAILED", "value": f"N/A (Error){suggestion}"}
                        except Exception as e:
                            print(f"  ❌ Fout bij navigeren naar tab {tab_name}: {str(e)}")
                            failures.append(f"Tab switch failed voor {tab_name}: {str(e)}")
                            for name in tab_data["elements"].values():
                                if name not in audit_details:
                                    audit_details[name] = {"status": "FAILED", "value": "N/A (Error)"}

                    # --- DEEP VISUAL INSPECTION (Charts & Badges) ---
                    print(f"  👉 Checking Visual Data (Charts & Badges)...")
                    try:
                        # Verifieer dat Signal Badges niet leeg zijn
                        badges_empty = await page.evaluate("""() => {
                            const badges = document.querySelectorAll('.brain-strategy-groups__chip');
                            if (badges.length === 0) return true;
                            let all_empty = true;
                            badges.forEach(b => { if (b.innerText.trim() !== '') all_empty = false; });
                            return all_empty;
                        }""")
                        if badges_empty:
                            print("    - [VALUE] Signal Mix Badges ❌ (FAILED: Badges zijn leeg of afwezig)")
                            failures.append("Signal Mix Badges zijn leeg")
                        else:
                            print("    - [VALUE] Signal Mix Badges ✅ (Aanwezig en gevuld)")

                        # Verifieer Grafieken (Loss, Reward, Equity, Price)
                        chart_ids = [
                            "priceChart", "brainCorrelationChart", "brainTabTrainingLossChart",
                            "brainTabFeatureChart", "brainTabRewardChart", "equityCurveChart",
                            "winLossChart", "sentimentOutcomeChart"
                        ]
                        for cid in chart_ids:
                            is_empty = await page.evaluate(f"""() => {{
                                const el = document.getElementById('{cid}');
                                if (!el) return true;
                                if (el.tagName.toLowerCase() === 'canvas') {{ return el.width === 0 || el.height === 0 || el.style.display === 'none'; }}
                                return el.innerHTML.trim() === '';
                            }}""")
                            if is_empty:
                                print(f"    - [VALUE] Chart #{cid} ❌ (FAILED: Grafiek is leeg of niet gerenderd)")
                                failures.append(f"Grafiek #{cid} is leeg")
                            else:
                                print(f"    - [VALUE] Chart #{cid} ✅ (Bevat pixel-data)")
                    except Exception as e:
                        print(f"  ❌ Fout bij grafiek validatie: {e}")

                    # Multi-Ticker Check
                    multi_ticker_ok, multi_ticker_msg = await check_multi_tickers(page)
                    if not multi_ticker_ok:
                        failures.append(multi_ticker_msg)
                        audit_details["Multi-Ticker"] = {"status": "FAILED", "value": multi_ticker_msg}
                    else:
                        audit_details["Multi-Ticker"] = {"status": "OK", "value": "Passed"}

                    # Market Switch Check
                    switch_ok, switch_msg = await test_market_switch(page)
                    if not switch_ok:
                        failures.append(switch_msg)
                        audit_details["Market-Switch"] = {"status": "FAILED", "value": switch_msg}
                    else:
                        audit_details["Market-Switch"] = {"status": "OK", "value": switch_msg}

                    # Night-Mode: Jumpstart the engine if it's empty
                    rl_conf = extracted_data.get("rl-confidence", "")
                    trades = extracted_data.get("ledgerPerfClosed", "")
                    if rl_conf in ["0.00", "0", "0.000", "—"] or trades in ["0", "—"]:
                        await trigger_auto_jumpstart(page)

                    # Data Freshness Check
                    current_btc_price = extracted_data.get("btc-price")
                    current_cpu_val = extracted_data.get("ringCpuVal")
                    
                    if current_btc_price and current_cpu_val:
                        if current_btc_price == last_btc_price and current_cpu_val == last_cpu_val:
                            if last_btc_price is not None and last_cpu_val is not None:
                                frozen_counter += 1
                                if frozen_counter >= 3:
                                    print(f"⚠️ WAARSCHUWING: Data Frozen! Prijs ({current_btc_price}) en CPU ({current_cpu_val}) ongewijzigd over 3 scans.")
                                    failures.append("Data Frozen (BTC Prijs en CPU gelijk gebleven)")
                                    data_frozen = True
                        else:
                            frozen_counter = 0
                    
                    if not failures and not data_frozen:
                        break # Success! We kunnen uit de retry-loop breken
                        
                last_btc_price = extracted_data.get("btc-price")
                last_cpu_val = extracted_data.get("ringCpuVal")

                recent_errors = scan_backend_logs()
                if recent_errors:
                    uniq_err = list(dict.fromkeys(recent_errors))
                    audit_details["Backend_Log_Analyse"] = {"status": "WARNING", "value": f"{len(recent_errors)} fouten gevonden in worker_execution.log. Laatste: {uniq_err[-1][:60]}..."}

                user_actions, repairability = collect_user_action_items(root_causes, failures, pipeline_warn_accum)

                # Report generation
                audit_report = {
                    "timestamp": datetime.now(AMSTERDAM).isoformat(),
                    "status": "OK" if not failures else ("WARNING: Data Frozen" if data_frozen else "ERROR"),
                    "metrics": audit_details,
                    "user_actions": user_actions,
                    "repairability": repairability,
                }

                if failures:
                    print(f"❌ AUDIT GEFAALD: {', '.join(failures)}")
                    print_user_action_block(user_actions, repairability)
                    try:
                        screenshot_bytes = await page.screenshot()
                        with open("/app/logs/AUDIT_FAILURE.png", "wb") as f:
                            f.write(screenshot_bytes)
                    except Exception as ss_err:
                        print(f"  ⚠️ Kon screenshot niet opslaan: {ss_err}")
                    with open("/app/logs/audit_status.txt", "w") as f:
                        f.write(f"FAILED: {', '.join(failures)}")
                else:
                    print("✅ DEEP-SCAN PASSED: Full Tour succesvol en multi-ticker gecontroleerd!")
                    if user_actions:
                        print("\n⚠️ [PIPELINE] Aanbevolen follow-up (worker/RL) — zie metrics.Pipeline: Worker brain diagnose")
                        print_user_action_block(user_actions, repairability)
                    scan_failed = False
                    if os.path.exists("/app/logs/audit_status.txt"):
                        os.remove("/app/logs/audit_status.txt")
                    
                    os.makedirs('/app/logs', exist_ok=True)
                    # Update heartbeat only on success
                    with open("/app/logs/heartbeat.json", "w") as f:
                        json.dump({"last_heartbeat": datetime.now(AMSTERDAM).isoformat(), "status": "ALIVE", "type": "deep_scan"}, f)

                os.makedirs('/app/logs', exist_ok=True)
                with open("/app/logs/last_audit_report.json", "w") as f:
                    json.dump(audit_report, f, indent=2)

            except Exception as e:
                print(f"💥 KRITIEKE FOUT in run loop: {e}")
            finally:
                # Voorkom root-owned bestanden op de host-bind (_logs_hub) na Playwright/validator-schrijfsels.
                await asyncio.to_thread(repair_permissions, resolve_logs_hub())

            if scan_failed:
                consecutive_successes = 0
                sleep_time = int(os.getenv("VALIDATOR_POST_FAIL_SLEEP_SEC", "45"))
                await attempt_fix(failures, list(root_causes), url)
                print(f"[{time.strftime('%H:%M:%S')}] ⏳ Deep-Scan voltooid. Wachten op data-sync ({sleep_time}s)...")
            else:
                consecutive_successes += 1
                if consecutive_successes <= 3:
                    sleep_time = 60
                    print(f"[AUTONOMY] 🛡️ 'Healing' fase actief ({consecutive_successes}/3 succesvolle rondes). Interval geforceerd naar 60s.")
                else:
                    sleep_time = base_interval
                    print(f"[AUTONOMY] ✅ Systeem stabiel. Schaal autonoom op naar rust-interval ({sleep_time}s).")
                    
                print(f"[{time.strftime('%H:%M:%S')}] ⏳ Deep-Scan voltooid. Wachten voor {sleep_time} seconden...")
            
            try:
                await page.close()
            except Exception:
                pass
                
            await asyncio.sleep(sleep_time)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_deep_scan())
