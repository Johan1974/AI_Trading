"""_logs_hub / Docker volume `/app/logs`: pad-resolutie en permissieherstel na root-owned validator-schrijfsels."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_logs_hub() -> Path:
    if Path("/.dockerenv").exists():
        p = Path("/app/logs")
        if p.is_dir() or p.exists():
            return p
    root = Path(__file__).resolve().parents[2]
    return root / "_logs_hub"


def ensure_repo_on_path() -> Path:
    """Zorg dat `import app.*` werkt vanaf scripts op de host of onder /app."""
    root = Path(__file__).resolve().parents[2]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def repair_permissions(hub: Path | None = None) -> int:
    """
    chown -R hub naar LOGS_HUB_HOST_UID:LOGS_HUB_HOST_GID wanneer euid==0 (validator-container).
    Zonder uid/gid in env: geen-op (veilig op de host als niet-root).
    """
    hub = hub or resolve_logs_hub()
    if os.geteuid() != 0:
        return 0
    u_s = (os.environ.get("LOGS_HUB_HOST_UID") or "").strip()
    g_s = (os.environ.get("LOGS_HUB_HOST_GID") or "").strip()
    if not u_s or not g_s:
        return 0
    try:
        uid, gid = int(u_s), int(g_s)
    except ValueError:
        print("fix-permissions: ongeldige LOGS_HUB_HOST_UID of LOGS_HUB_HOST_GID")
        return 1
    if not hub.exists():
        return 0
    if not hub.is_dir():
        try:
            os.chown(hub, uid, gid, follow_symlinks=False)
        except OSError as exc:
            print(f"fix-permissions: chown {hub}: {exc}")
            return 1
        print(f"fix-permissions: {hub} -> {uid}:{gid}")
        return 0
    n = 0
    for dirpath, dirnames, filenames in os.walk(hub, topdown=False):
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            try:
                os.chown(p, uid, gid, follow_symlinks=False)
                n += 1
            except OSError as exc:
                print(f"fix-permissions: chown {p}: {exc}")
        for name in dirnames:
            p = base / name
            try:
                os.chown(p, uid, gid, follow_symlinks=False)
                n += 1
            except OSError as exc:
                print(f"fix-permissions: chown {p}: {exc}")
    try:
        os.chown(hub, uid, gid, follow_symlinks=False)
        n += 1
    except OSError as exc:
        print(f"fix-permissions: chown {hub}: {exc}")
    print(f"fix-permissions: {hub} -> uid:gid {uid}:{gid} ({n} entries)")
    return 0
