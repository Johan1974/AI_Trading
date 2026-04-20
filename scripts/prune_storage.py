"""
Bestand: scripts/prune_storage.py
Relatief pad: ./scripts/prune_storage.py
Functie: Downsamplet oude Parquet data en verwijdert RL-irrelevante ruis.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from datetime import UTC
except ImportError:  # Python < 3.11
    UTC = timezone.utc
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE_ROOT = Path.home() / "AI_Trading" / "storage"
SUBDIRS = ("historical_prices", "historical_news", "rl_features")


def _parse_ts(df: pd.DataFrame) -> pd.Series:
    for candidate in ("timestamp", "published_at", "ts"):
        if candidate in df.columns:
            return pd.to_datetime(df[candidate], utc=True, errors="coerce")
    raise ValueError("Geen timestamp-kolom gevonden voor pruning.")


def _downsample_ohlcv(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    bucketed = df.copy()
    bucketed[ts_col] = pd.to_datetime(bucketed[ts_col], utc=True, errors="coerce")
    bucketed = bucketed.dropna(subset=[ts_col]).sort_values(ts_col)
    bucketed["bucket"] = bucketed[ts_col].dt.floor("min")
    if {"open", "high", "low", "close", "volume"}.issubset(bucketed.columns):
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        others = [c for c in bucketed.columns if c not in {"bucket", ts_col, *agg.keys()}]
        for col in others:
            agg[col] = "mean"
        out = bucketed.groupby("bucket", as_index=False).agg(agg)
        out = out.rename(columns={"bucket": ts_col})
        return out
    return bucketed.drop(columns=["bucket"]).drop_duplicates(subset=[ts_col])


def _remove_rl_noise(df: pd.DataFrame) -> pd.DataFrame:
    if not {"price_action", "sentiment_score", "news_confidence"}.issubset(df.columns):
        return df
    keep = ~(
        (df["price_action"].abs() < 1e-8)
        & (df["sentiment_score"].abs() < 1e-6)
        & (df["news_confidence"].abs() < 1e-6)
    )
    pruned = df.loc[keep].copy()
    return pruned if not pruned.empty else df


def prune_file(path: Path, cutoff_dt: datetime) -> bool:
    df = pd.read_parquet(path)
    if df.empty:
        return False
    ts_series = _parse_ts(df)
    ts_col = ts_series.name
    if ts_col is None:
        return False
    older_mask = ts_series < cutoff_dt
    if not older_mask.any():
        return False
    older = df.loc[older_mask].copy()
    newer = df.loc[~older_mask].copy()
    older = _downsample_ohlcv(older, ts_col=ts_col)
    older = _remove_rl_noise(older)
    merged = pd.concat([older, newer], axis=0, ignore_index=True)
    merged = merged.drop_duplicates().sort_values(ts_col).reset_index(drop=True)
    merged.to_parquet(path, index=False)
    return True


def main() -> None:
    cutoff_dt = datetime.now(UTC) - timedelta(days=30)
    changed = 0
    scanned = 0
    for sub in SUBDIRS:
        directory = STORAGE_ROOT / sub
        if not directory.exists():
            continue
        for path in directory.glob("*.parquet"):
            scanned += 1
            try:
                if prune_file(path, cutoff_dt=cutoff_dt):
                    changed += 1
            except Exception:
                continue
    print(f"Prune complete. scanned={scanned}, changed={changed}, storage={STORAGE_ROOT}")


if __name__ == "__main__":
    main()
