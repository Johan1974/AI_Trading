"""Lees Overall voortgang uit ROADMAP.md (robust tegen markdown ** en spaties)."""

from __future__ import annotations

import re
from pathlib import Path


def parse_roadmap_progress_text(md: str) -> tuple[int, int, int]:
    """
    Returns (percent, done_count, total_count) uit een regel als:
    Overall voortgang: 91% (182/200 taken)
    """
    if not md:
        return 0, 0, 0
    clean = re.sub(r"[*_`]+", "", md)
    m = re.search(
        r"Overall\s+voortgang:\s*(\d+)\s*%\s*\(\s*(\d+)\s*/\s*(\d+)",
        clean,
        flags=re.IGNORECASE,
    )
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_roadmap_progress_file(path: str | Path) -> tuple[int, int, int]:
    p = Path(path)
    if not p.exists():
        return 0, 0, 0
    return parse_roadmap_progress_text(p.read_text(encoding="utf-8", errors="replace"))
