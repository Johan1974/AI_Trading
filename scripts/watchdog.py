#!/usr/bin/env python3
"""Dunne wrapper: de productie-watchdog staat in ``watchdog.sh`` (bash + docker)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    sh = Path(__file__).resolve().parent / "watchdog.sh"
    if not sh.is_file():
        print(f"[watchdog] ontbreekt: {sh}", file=sys.stderr)
        return 127
    return subprocess.call(["/bin/bash", str(sh), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
