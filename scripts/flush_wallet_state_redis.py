#!/usr/bin/env python3
"""
Flush gigantische wallet_state uit Redis en prunet predict_rl_feature_snapshots in SQLite.

Wat dit doet:
  - DEL ai_trading_snapshot  (string met volledige STATE JSON)
  - DEL worker_snapshot      (hash met per-key STATE velden incl. paper_portfolio)
  - Prunet predict_rl_feature_snapshots naar laatste N rijen per market (default 200)
  - Schrijft een minimale wallet_state snapshot terug naar SQLite (optioneel)

Wat dit NIET doet:
  - trade_history aanraken
  - trade_events aanraken
  - Actieve posities wissen
  - Redis FLUSHALL

Gebruik:
  python scripts/flush_wallet_state_redis.py [--db data/database.db] [--keep-snapshots 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _redis_url() -> str:
    host = os.getenv("REDIS_HOST", "redis").strip()
    port = os.getenv("REDIS_PORT", "6379").strip()
    return os.getenv("REDIS_URL", f"redis://{host}:{port}/0").strip()


def flush_redis(dry_run: bool) -> None:
    try:
        import redis as redis_lib
    except ImportError:
        print("[SKIP] redis-py niet beschikbaar in dit environment.")
        return

    url = _redis_url()
    r = redis_lib.Redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)
    try:
        pong = r.ping()
        print(f"[REDIS] Verbonden ({url}), ping={pong}")
    except Exception as exc:
        print(f"[REDIS] Kan niet verbinden: {exc}")
        return

    keys_to_del = ["ai_trading_snapshot", "worker_snapshot"]
    for key in keys_to_del:
        try:
            exists = r.exists(key)
            if not exists:
                print(f"[REDIS] {key!r} — niet aanwezig, skip.")
                continue
            if key == "worker_snapshot":
                size = r.hlen(key)
                print(f"[REDIS] {key!r} (hash, {size} velden) → {'DRY-RUN skip' if dry_run else 'DELETE'}")
            else:
                raw = r.get(key)
                n = len(raw) if raw else 0
                print(f"[REDIS] {key!r} ({n:,} bytes) → {'DRY-RUN skip' if dry_run else 'DELETE'}")
            if not dry_run:
                r.delete(key)
                print(f"[REDIS] {key!r} verwijderd.")
        except Exception as exc:
            print(f"[REDIS] Fout bij {key!r}: {exc}")
    r.close()


def prune_sqlite(db_path: Path, keep: int, dry_run: bool) -> None:
    if not db_path.exists():
        print(f"[SQLITE] DB niet gevonden: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        if "predict_rl_feature_snapshots" in tables:
            total = conn.execute("SELECT COUNT(*) FROM predict_rl_feature_snapshots").fetchone()[0]
            markets = [r[0] for r in conn.execute("SELECT DISTINCT market FROM predict_rl_feature_snapshots")]
            print(f"[SQLITE] predict_rl_feature_snapshots: {total} rijen over {len(markets)} markets")
            deleted = 0
            for market in markets:
                mcount = conn.execute(
                    "SELECT COUNT(*) FROM predict_rl_feature_snapshots WHERE market = ?", (market,)
                ).fetchone()[0]
                to_delete = max(0, mcount - keep)
                print(f"         {market}: {mcount} rijen → bewaar laatste {keep}, verwijder {to_delete}")
                if not dry_run and to_delete > 0:
                    conn.execute(
                        """
                        DELETE FROM predict_rl_feature_snapshots
                        WHERE market = ? AND id NOT IN (
                            SELECT id FROM predict_rl_feature_snapshots
                            WHERE market = ?
                            ORDER BY id DESC LIMIT ?
                        )
                        """,
                        (market, market, keep),
                    )
                    deleted += to_delete
            if not dry_run:
                conn.commit()
                remaining = conn.execute("SELECT COUNT(*) FROM predict_rl_feature_snapshots").fetchone()[0]
                print(f"[SQLITE] predict_rl_feature_snapshots: {deleted} verwijderd, {remaining} over.")
        else:
            print("[SQLITE] predict_rl_feature_snapshots tabel niet gevonden.")

        if "wallet_state" in tables:
            rows = conn.execute(
                "SELECT tenant_id, length(snapshot_json) as sz, updated_ts_utc FROM wallet_state ORDER BY updated_ts_utc DESC"
            ).fetchall()
            for row in rows:
                print(f"[SQLITE] wallet_state: tenant={row['tenant_id']} size={row['sz']:,} bytes ts={row['updated_ts_utc']}")
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Flush gigantische wallet_state uit Redis + SQLite pruning")
    ap.add_argument("--db", default=os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db"))
    ap.add_argument("--keep-snapshots", type=int, default=200, metavar="N",
                    help="Bewaar laatste N predict_rl_feature_snapshots per market (default: 200)")
    ap.add_argument("--dry-run", action="store_true", help="Toon wat er zou gebeuren zonder iets te wijzigen")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"\n=== flush_wallet_state_redis.py [{mode}] ===")
    print(f"DB: {db_path}")
    print(f"Snapshot keep: {args.keep_snapshots}\n")

    flush_redis(dry_run=args.dry_run)
    print()
    prune_sqlite(db_path=db_path, keep=args.keep_snapshots, dry_run=args.dry_run)
    print("\nKlaar. Herstart de worker om een verse snapshot naar Redis te pushen.")


if __name__ == "__main__":
    main()
