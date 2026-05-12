#!/usr/bin/env python3
"""
Quick diagnostics for _logs_hub + trading/RL activity.

Usage:
  python3 scripts/analyze_logs_hub.py
  python3 scripts/analyze_logs_hub.py --fix   # veilige ontbrekende stubs (geen overwrite)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.diagnostics.logs_hub_maintenance import repair_permissions, resolve_logs_hub

LOGS_HUB = resolve_logs_hub()
_td = str(os.getenv("TRADE_HISTORY_DB_PATH", "") or "").strip()
if _td:
    _tp = Path(_td)
    TRADE_DB = _tp if _tp.is_absolute() else (ROOT / _tp)
else:
    TRADE_DB = ROOT / "data" / "database.db"
RL_DB_HUB = LOGS_HUB / "rl_training_metrics.sqlite"
RL_DB_STORAGE = ROOT / "storage" / "rl_training_metrics.sqlite"
WORKER_LOG = LOGS_HUB / "worker_execution.log"


def _rl_metrics_db_candidates() -> list[tuple[str, Path]]:
    """Zelfde default als worker: env RL_METRICS_DB_PATH, anders storage, dan legacy _logs_hub."""
    out: list[tuple[str, Path]] = []
    raw = str(os.getenv("RL_METRICS_DB_PATH", "") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        out.append(("RL_METRICS_DB_PATH", p))
    out.append(("storage", RL_DB_STORAGE))
    out.append(("_logs_hub", RL_DB_HUB))
    return out


def _print_header(title: str) -> None:
    print(f"\n== {title} ==")


def _owner_name(path: Path) -> str:
    try:
        import pwd

        st = path.stat()
        return pwd.getpwuid(st.st_uid).pw_name
    except Exception:
        try:
            return str(path.stat().st_uid)
        except Exception:
            return "?"


def analyze_permissions() -> None:
    _print_header("Permissions / ownership")
    if not LOGS_HUB.is_dir():
        print(f"Missing directory: {LOGS_HUB}")
        return
    me = os.geteuid()
    bad: list[str] = []
    for p in sorted(LOGS_HUB.iterdir()):
        if not p.is_file():
            continue
        try:
            st = p.stat()
            if st.st_uid != me and me != 0:
                bad.append(f"{p.name} (owner={_owner_name(p)}, uid={st.st_uid})")
        except OSError as exc:
            bad.append(f"{p.name} (<stat: {exc}>)")
    if not bad:
        print(f"- All listed files owned by uid {me} (or running as root).")
        return
    print("- Files not owned by current euid (can break portal writes / throttles):")
    for line in bad:
        print(f"  - {line}")
    print(f"- Fix:  ./scripts/fix_logs_hub.sh   or   sudo chown -R $(id -un):$(id -gn) {LOGS_HUB}")


def analyze_logs_hub_files() -> None:
    _print_header("_logs_hub status")
    if not LOGS_HUB.exists():
        print(f"Missing: {LOGS_HUB}")
        return
    print(f"Path: {LOGS_HUB}")
    for p in sorted(LOGS_HUB.iterdir()):
        if not p.is_file():
            continue
        try:
            st = p.stat()
            own = _owner_name(p)
            print(f"- {p.name:30} {st.st_size:10d} bytes  [{own}]")
        except Exception:
            print(f"- {p.name:30} <stat failed>")


def analyze_worker_log() -> None:
    _print_header("worker_execution.log diagnostics")
    lines = _tail_lines(WORKER_LOG, max_lines=6000)
    if not lines:
        print("No worker log lines found.")
        return

    tracebacks = sum(1 for ln in lines if "Traceback (most recent call last)" in ln)
    startup_nameerror = sum(1 for ln in lines if "build_text_audit_report" in ln and "NameError" in ln)
    engine_sweeps = sum(1 for ln in lines if "[ENGINE] Elite sweep" in ln)
    training_starts = sum(1 for ln in lines if "Training loop started" in ln)
    paper_cmds = sum(1 for ln in lines if "Commando ontvangen: Paper run" in ln)

    print(f"- Tracebacks (tail): {tracebacks}")
    print(f"- Startup audit NameError hits: {startup_nameerror}")
    print(f"- Engine sweeps (tail): {engine_sweeps}")
    print(f"- RL training loop starts (tail): {training_starts}")
    print(f"- Paper run commands (tail): {paper_cmds}")

    interesting = [
        ln
        for ln in lines
        if (
            "Harde crash bij startup audit email" in ln
            or "Traceback (most recent call last)" in ln
            or "[ENGINE] Cycle failed" in ln
            or "mislukt" in ln.lower()
        )
    ]
    if interesting:
        print("\nRecent notable log lines:")
        for ln in interesting[-10:]:
            print(f"  {ln}")


def _tail_lines(path: Path, max_lines: int = 2000) -> list[str]:
    if not path.exists():
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    return data[-max_lines:]


def analyze_trade_db() -> None:
    _print_header(f"Trade activity ({TRADE_DB})")
    if not TRADE_DB.exists():
        print(f"Missing: {TRADE_DB}")
        return

    with sqlite3.connect(str(TRADE_DB)) as con:
        cur = con.cursor()
        try:
            total_events = cur.execute("SELECT COUNT(*) FROM trade_events").fetchone()[0]
            print(f"- trade_events total: {total_events}")
        except Exception as exc:
            print(f"- trade_events unavailable: {exc}")
            return

        rows = cur.execute(
            """
            SELECT ts_utc, market, action, signal, status, reason
            FROM trade_events
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()
        if not rows:
            print("- No trade events yet.")
            return
        print("- Last 10 trade events:")
        for row in rows:
            ts, market, action, signal, status, reason = row
            print(f"  {ts} | {market} | {action}/{signal} | {status} | reason={reason or '-'}")


def analyze_rl_db() -> None:
    _print_header("RL learning activity")
    jsonl = LOGS_HUB / "rl_hourly_metrics.jsonl"
    if jsonl.exists():
        try:
            jl_n = sum(1 for _ in jsonl.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            jl_n = -1
        print(f"- rl_hourly_metrics.jsonl lines: {jl_n}")
    else:
        print("- rl_hourly_metrics.jsonl: missing (worker schrijft na eerste uur-checkpoint)")

    found = False
    for label, dbp in _rl_metrics_db_candidates():
        if not dbp.exists():
            print(f"- {label}: not found ({dbp})")
            continue
        found = True
        print(f"- Using {label}: {dbp}")
        with sqlite3.connect(str(dbp)) as con:
            cur = con.cursor()
            try:
                n = cur.execute("SELECT COUNT(*) FROM rl_training_chunks").fetchone()[0]
                print(f"  rl_training_chunks: {n}")
                row = cur.execute(
                    "SELECT ts_utc, pair, global_step FROM rl_training_chunks ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    print(f"  last chunk: ts={row[0]} pair={row[1]} global_step={row[2]}")
            except Exception as exc:
                print(f"  RL metrics query failed: {exc}")
        break
    if not found:
        print("No RL metrics SQLite yet ( eerste train-sessie maakt het bestand ).")
        print("Tip: docker-compose.env → RL_METRICS_DB_PATH=/app/storage/rl_training_metrics.sqlite + volume ./storage")


def analyze_json_health() -> None:
    _print_header("Health snapshots")
    for fn in ("heartbeat.json", "performance.json", "system_state.json", "last_audit_report.json"):
        p = LOGS_HUB / fn
        if not p.exists():
            print(f"- {fn}: missing")
            continue
        try:
            raw = p.read_text(encoding="utf-8").strip()
            if not raw:
                print(f"- {fn}: empty file")
                continue
            obj = json.loads(raw)
            keys = list(obj.keys())[:12] if isinstance(obj, dict) else []
            print(f"- {fn}: ok ({p.stat().st_size} bytes), keys={keys}")
        except Exception as exc:
            print(f"- {fn}: invalid json ({exc})")


def _repair_json_if_corrupt(path: Path, factory: dict[str, object] | None, empty_fallback: str) -> bool:
    """Als bestand bestaat maar geen geldige JSON is: .bak wegschuiven en vervangen."""
    if not path.exists():
        return False
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError("empty")
        json.loads(raw)
        return False
    except Exception:
        pass
    try:
        bak = path.with_suffix(path.suffix + ".bak")
        path.replace(bak)
    except OSError:
        try:
            path.unlink()
        except OSError:
            return False
    try:
        if factory is not None:
            path.write_text(json.dumps(factory, indent=2), encoding="utf-8")
        else:
            path.write_text(empty_fallback, encoding="utf-8")
        print(f"+ repaired corrupt {path.name} (backup .bak)")
        return True
    except OSError as exc:
        print(f"SKIP repair {path.name}: {exc}")
        return False


def repair_logs_hub() -> int:
    """Create directory + missing stubs; repareer corrupte JSON; RL-nachtrun klaarzetten."""
    _print_header("Repair (--fix)")
    try:
        LOGS_HUB.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"FAIL: cannot mkdir {LOGS_HUB}: {exc}")
        return 1

    fixed = 0
    for corrupt in (LOGS_HUB / "heartbeat.json", LOGS_HUB / "performance.json", LOGS_HUB / "system_state.json"):
        fac: dict[str, object] | None
        if corrupt.name == "heartbeat.json":
            fac = {
                "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                "status": "placeholder",
                "type": "analyze_logs_hub",
                "note": "Repaired stub; vervangen door worker resource_watchdog zodra die draait.",
            }
        elif corrupt.name == "performance.json":
            fac = {"cpu_max": 0.0, "ram_max": 0.0, "gpu_util_max": 0.0, "gpu_temp_max": 0.0, "last_reset": 0.0}
        else:
            fac = {}
        if corrupt.name == "system_state.json":
            if _repair_json_if_corrupt(corrupt, {}, "{}"):
                fixed += 1
        else:
            if _repair_json_if_corrupt(corrupt, fac, ""):
                fixed += 1

    hb = LOGS_HUB / "heartbeat.json"
    if not hb.exists():
        try:
            hb.write_text(
                json.dumps(
                    {
                        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                        "status": "placeholder",
                        "type": "analyze_logs_hub",
                        "note": "Replace with live heartbeat from dashboard-validator or runtime.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"+ wrote {hb.name}")
            fixed += 1
        except OSError as exc:
            print(f"SKIP heartbeat.json: {exc}")

    ss = LOGS_HUB / "system_state.json"
    if not ss.exists():
        try:
            ss.write_text("{}", encoding="utf-8")
            print(f"+ wrote empty {ss.name} (throttle / dedupe state)")
            fixed += 1
        except OSError as exc:
            print(f"SKIP system_state.json: {exc}")

    lar = LOGS_HUB / "last_audit_report.json"
    if not lar.exists():
        try:
            lar.write_text(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "NO_AUDIT_YET",
                        "metrics": {},
                        "user_actions": [],
                        "repairability": (
                            "Nog geen dashboard-validator deep-scan geschreven. "
                            "Na PLAYwright audit wordt dit overschreven met echte metrics."
                        ),
                        "note": "Stub aangemaakt door analyze_logs_hub.py --fix",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"+ wrote {lar.name} (stub tot eerste validator-run)")
            fixed += 1
        except OSError as exc:
            print(f"SKIP last_audit_report.json: {exc}")

    jsonl = LOGS_HUB / "rl_hourly_metrics.jsonl"
    if not jsonl.exists():
        try:
            jsonl.touch()
            print(f"+ touched {jsonl.name} (append-only uurlijkse RL-samenvatting)")
            fixed += 1
        except OSError as exc:
            print(f"SKIP rl_hourly_metrics.jsonl: {exc}")

    try:
        from app.services.rl_metrics_store import init_rl_metrics_db

        candidates = _rl_metrics_db_candidates()
        existing = next(((lab, dbp) for lab, dbp in candidates if dbp.exists()), None)
        if existing:
            print(f"- RL metrics DB OK ({existing[0]}): {existing[1]}")
        else:
            label, dbp = candidates[0]
            try:
                dbp.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(f"SKIP RL DB mkdir: {exc}")
            else:
                prev = os.environ.get("RL_METRICS_DB_PATH")
                os.environ["RL_METRICS_DB_PATH"] = str(dbp)
                try:
                    init_rl_metrics_db()
                    print(f"+ initialized RL metrics schema ({label}): {dbp}")
                    fixed += 1
                except Exception as exc:
                    print(f"SKIP RL init {dbp}: {exc}")
                finally:
                    if prev is None:
                        os.environ.pop("RL_METRICS_DB_PATH", None)
                    else:
                        os.environ["RL_METRICS_DB_PATH"] = prev
    except Exception as exc:
        print(f"SKIP RL metrics bootstrap: {exc}")

    if fixed == 0:
        print("Nothing to create or repair.")
    print("If ownership warnings remain: --fix-permissions (validator) of ./scripts/fix_logs_hub.sh")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose _logs_hub")
    ap.add_argument(
        "--fix",
        action="store_true",
        help="Ontbrekende stubs, corrupte JSON (.bak), rl_hourly_metrics.jsonl, RL-metrics-DB schema indien nodig",
    )
    ap.add_argument(
        "--fix-permissions",
        action="store_true",
        help="Als root in Docker: chown volume naar LOGS_HUB_HOST_UID:GID (portal/worker schrijfbaar)",
    )
    args = ap.parse_args()

    if args.fix:
        rc = repair_logs_hub()
        if rc != 0:
            return rc

    if args.fix_permissions:
        rc = repair_permissions(LOGS_HUB)
        if rc != 0:
            return rc

    analyze_logs_hub_files()
    analyze_permissions()
    analyze_worker_log()
    analyze_trade_db()
    analyze_rl_db()
    analyze_json_health()
    return 0


if __name__ == "__main__":
    sys.exit(main())
