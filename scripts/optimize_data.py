"""
Bestand: scripts/optimize_data.py
Relatief pad: ./scripts/optimize_data.py
Functie: Onderhoud voor storage- en modelbeheer (downsample, pruning, model cleanup, csv->parquet).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

try:
    from datetime import UTC
except ImportError:  # Python < 3.11
    UTC = timezone.utc
from pathlib import Path
from typing import Any

import pandas as pd

STORAGE_ROOT = Path.home() / "AI_Trading" / "storage"
MODEL_ROOT = Path(os.getenv("RL_MODEL_DIR", "artifacts/rl")).expanduser()
TOP_MODEL_COUNT = 5
STATS_PATH = STORAGE_ROOT / "stats.json"
LOG_PATH = STORAGE_ROOT / "logs" / "bot_execution.log"
MAX_LOG_SIZE_BYTES = 50 * 1024 * 1024


def _time_col(df: pd.DataFrame) -> str | None:
    for candidate in ("timestamp", "published_at", "ts", "time"):
        if candidate in df.columns:
            return candidate
    return None


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _is_second_resolution(df: pd.DataFrame, ts_col: str) -> bool:
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 10:
        return False
    diffs = ts.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return False
    median_delta = float(diffs.median())
    return median_delta < 60.0


def _downsample_old_rows(path: Path, cutoff: datetime) -> bool:
    df = _read_parquet(path)
    if df is None or df.empty:
        return False
    ts_col = _time_col(df)
    if not ts_col:
        return False
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    older = df.loc[ts < cutoff].copy()
    newer = df.loc[ts >= cutoff].copy()
    if older.empty:
        return False
    older[ts_col] = pd.to_datetime(older[ts_col], utc=True, errors="coerce")
    older = older.dropna(subset=[ts_col]).sort_values(ts_col)
    if older.empty or not _is_second_resolution(older, ts_col):
        return False

    older["bucket"] = older[ts_col].dt.floor("min")
    if {"open", "high", "low", "close", "volume"}.issubset(older.columns):
        agg: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for col in older.columns:
            if col in {"bucket", ts_col, *agg.keys()}:
                continue
            agg[col] = "mean"
        older = older.groupby("bucket", as_index=False).agg(agg).rename(columns={"bucket": ts_col})
    else:
        older = older.drop(columns=["bucket"]).drop_duplicates(subset=[ts_col])

    out = pd.concat([older, newer], ignore_index=True)
    out = out.drop_duplicates().sort_values(ts_col).reset_index(drop=True)
    out.to_parquet(path, index=False)
    return True


def _prune_raw_news(path: Path, cutoff: datetime) -> bool:
    if "historical_news" not in str(path):
        return False
    df = _read_parquet(path)
    if df is None or df.empty:
        return False
    ts_col = _time_col(df)
    if not ts_col:
        return False
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    older_mask = ts < cutoff
    if not older_mask.any():
        return False

    raw_cols = [c for c in ("title", "summary", "url", "source", "description", "headline", "text") if c in df.columns]
    if not raw_cols:
        return False
    older = df.loc[older_mask].copy().drop(columns=raw_cols, errors="ignore")
    newer = df.loc[~older_mask].copy()
    out = pd.concat([older, newer], ignore_index=True).drop_duplicates().reset_index(drop=True)
    out.to_parquet(path, index=False)
    return True


def _convert_csv_to_parquet() -> tuple[int, int]:
    converted = 0
    failed = 0
    if not STORAGE_ROOT.exists():
        return converted, failed
    for csv_path in STORAGE_ROOT.rglob("*.csv"):
        parquet_path = csv_path.with_suffix(".parquet")
        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, index=False)
            csv_path.unlink()
            converted += 1
        except Exception:
            failed += 1
    return converted, failed


def _gather_model_candidates() -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for zip_path in MODEL_ROOT.glob("*.zip"):
        candidates[str(zip_path)] = {
            "path": zip_path,
            "reward": float("-inf"),
            "mtime": zip_path.stat().st_mtime,
        }
    for meta_path in MODEL_ROOT.glob("*_models.json"):
        try:
            rows = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_path = Path(str(row.get("model_path", "")))
            if not model_path.is_absolute():
                model_path = (MODEL_ROOT / model_path.name).resolve()
            key = str(model_path)
            if key not in candidates:
                continue
            candidates[key]["reward"] = float(row.get("reward_score", float("-inf")))
    return list(candidates.values())


def _cleanup_models() -> tuple[int, int]:
    if not MODEL_ROOT.exists():
        return 0, 0
    rows = _gather_model_candidates()
    if len(rows) <= TOP_MODEL_COUNT:
        return len(rows), 0
    ranked = sorted(rows, key=lambda r: (float(r.get("reward", float("-inf"))), float(r.get("mtime", 0))), reverse=True)
    keep = ranked[:TOP_MODEL_COUNT]
    drop = ranked[TOP_MODEL_COUNT:]
    removed = 0
    keep_names = {Path(str(x["path"])).name for x in keep}
    for item in drop:
        path = Path(str(item["path"]))
        if path.exists():
            path.unlink()
            removed += 1

    for meta_path in MODEL_ROOT.glob("*_models.json"):
        try:
            rows_json = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows_json, list):
            continue
        kept_rows = []
        for row in rows_json:
            if not isinstance(row, dict):
                continue
            model_path = Path(str(row.get("model_path", "")))
            if model_path.name in keep_names:
                kept_rows.append(row)
        meta_path.write_text(json.dumps(kept_rows[:TOP_MODEL_COUNT], ensure_ascii=True, indent=2), encoding="utf-8")

    return len(keep), removed


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except Exception:
                continue
    return total


def _prune_and_rotate_logs() -> dict[str, int]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch(exist_ok=True)
        return {"log_lines_before": 0, "log_lines_after": 0, "log_bytes_before": 0, "log_bytes_after": 0}

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    raw = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    kept: list[str] = []
    for line in lines:
        if line.startswith("[") and "]" in line:
            stamp = line[1 : line.index("]")]
            try:
                ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    kept.append(line)
                continue
            except Exception:
                pass
        kept.append(line)

    text = "\n".join(kept)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_LOG_SIZE_BYTES:
        encoded = encoded[-MAX_LOG_SIZE_BYTES:]
        text = encoded.decode("utf-8", errors="ignore")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    LOG_PATH.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return {
        "log_lines_before": len(lines),
        "log_lines_after": len(text.splitlines()),
        "log_bytes_before": len(raw.encode("utf-8")),
        "log_bytes_after": len((text + ("\n" if text else "")).encode("utf-8")),
    }


def _write_stats(
    size_before: int,
    size_after: int,
    downsampled: int,
    pruned_news: int,
    converted_csv: int,
    removed_models: int,
    log_metrics: dict[str, int],
) -> None:
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "size_before": int(size_before),
        "size_after": int(size_after),
        "saved_bytes": int(max(0, size_before - size_after)),
        "downsampled_files": int(downsampled),
        "pruned_news_files": int(pruned_news),
        "converted_csv_files": int(converted_csv),
        "removed_models": int(removed_models),
        "history_days": 400,
        "resolution": "Mixed (1s/1m)",
        "logs": log_metrics,
    }
    STATS_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    size_before = _dir_size_bytes(STORAGE_ROOT)
    downsample_cutoff = datetime.now(UTC) - timedelta(days=7)
    news_prune_cutoff = datetime.now(UTC) - timedelta(days=30)

    scanned = 0
    downsampled = 0
    pruned_news = 0
    if STORAGE_ROOT.exists():
        for parquet_path in STORAGE_ROOT.rglob("*.parquet"):
            scanned += 1
            try:
                if _downsample_old_rows(parquet_path, downsample_cutoff):
                    downsampled += 1
            except Exception:
                pass
            try:
                if _prune_raw_news(parquet_path, news_prune_cutoff):
                    pruned_news += 1
            except Exception:
                pass

    converted_csv, failed_csv = _convert_csv_to_parquet()
    kept_models, removed_models = _cleanup_models()
    log_metrics = _prune_and_rotate_logs()
    size_after = _dir_size_bytes(STORAGE_ROOT)
    _write_stats(
        size_before=size_before,
        size_after=size_after,
        downsampled=downsampled,
        pruned_news=pruned_news,
        converted_csv=converted_csv,
        removed_models=removed_models,
        log_metrics=log_metrics,
    )

    print(
        "Optimize data complete:"
        f" size_before={size_before}, size_after={size_after}, saved_bytes={max(0, size_before - size_after)},"
        f" scanned_parquet={scanned}, downsampled={downsampled}, pruned_news={pruned_news},"
        f" csv_to_parquet={converted_csv}, csv_failed={failed_csv},"
        f" models_kept={kept_models}, models_removed={removed_models},"
        f" log_lines_before={log_metrics.get('log_lines_before', 0)}, log_lines_after={log_metrics.get('log_lines_after', 0)}"
    )


if __name__ == "__main__":
    main()
