"""
Bestand: app/services/system_stats.py
Functie: CPU/RAM/disk via psutil; GPU-load primair via nvidia-smi utilization.gpu; VRAM via nvidia-smi.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import time
from typing import Any


def _to_float_loose(raw: str) -> float:
    s = str(raw).strip().replace("%", "").replace("MiB", "").strip()
    s = s.replace(",", ".")
    if s.upper() in {"", "N/A", "[N/A]"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_index_name_line(line: str) -> tuple[int, str] | None:
    try:
        row = next(csv.reader(io.StringIO(line.strip())))
    except StopIteration:
        return None
    if len(row) < 2:
        return None
    try:
        return int(str(row[0]).strip()), str(row[1]).strip()
    except ValueError:
        return None


def _nvidia_gpu_query_row(gpu_index: int) -> tuple[float, float, float, float]:
    """Eén `nvidia-smi` call: utilization.gpu, utilization.memory, memory.used (MiB), memory.total (MiB)."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(int(gpu_index)),
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0.0, 0.0, 0.0, 0.0
        uline = proc.stdout.strip().splitlines()[-1]
        row = next(csv.reader(io.StringIO(uline.strip())))
        if len(row) < 4:
            return 0.0, 0.0, 0.0, 0.0
        sm = max(0.0, min(100.0, _to_float_loose(row[0])))
        memu = max(0.0, min(100.0, _to_float_loose(row[1])))
        used = max(0.0, _to_float_loose(row[2]))
        tot = max(0.0, _to_float_loose(row[3]))
        return sm, memu, used, tot
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError, StopIteration):
        return 0.0, 0.0, 0.0, 0.0


def _nvidia_smi_stats() -> dict[str, Any]:
    """Kiest GPU (voorkeur GTX 1080); SM-util, memory-controller-util, VRAM; `gpu_util_effective` = max(SM, mem)."""
    out: dict[str, Any] = {
        "gpu_util_pct": 0.0,
        "gpu_mem_util_pct": 0.0,
        "gpu_util_effective": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "gpu_ok": False,
        "gpu_name": "",
        "gpu_index": -1,
    }
    if not shutil.which("nvidia-smi"):
        return out
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return out
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        parsed: list[tuple[int, str]] = []
        for line in lines:
            row = _parse_index_name_line(line)
            if row:
                parsed.append(row)
        if not parsed:
            return out
        chosen_idx, chosen_name = parsed[0][0], parsed[0][1]
        for idx, name in parsed:
            if "1080" in name.lower():
                chosen_idx, chosen_name = idx, name
                break

        sm_u, mem_u, used, total = _nvidia_gpu_query_row(chosen_idx)
        eff = max(0.0, min(100.0, max(sm_u, mem_u)))
        out["gpu_util_pct"] = sm_u
        out["gpu_mem_util_pct"] = mem_u
        out["gpu_util_effective"] = eff
        out["vram_used_mb"] = used
        out["vram_total_mb"] = total
        out["gpu_name"] = chosen_name.strip()
        out["gpu_index"] = int(chosen_idx)
        out["gpu_ok"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return out


def collect_system_stats() -> dict[str, Any]:
    """Eén snapshot voor WebSocket/API: CPU%, RAM%, disk% root, GPU%, VRAM (MB)."""
    cpu_pct = 0.0
    ram_pct = 0.0
    disk_pct = 0.0
    try:
        import psutil

        # Eerste cpu_percent-call is vaak 0.0; warm-up + interval geeft een echte sample.
        _ = psutil.cpu_percent(interval=0.05)
        cpu_pct = float(psutil.cpu_percent(interval=0.22))
        ram_pct = float(psutil.virtual_memory().percent)
        disk_pct = float(psutil.disk_usage("/").percent)
    except Exception:
        pass

    if cpu_pct < 0.1:
        try:
            load1, _, _ = os.getloadavg()
            n = max(1, (os.cpu_count() or 1))
            cpu_pct = min(100.0, (float(load1) / float(n)) * 100.0)
        except Exception:
            pass

    gpu = _nvidia_smi_stats()
    try:
        gpu_idx = int(gpu.get("gpu_index", -1))
    except (TypeError, ValueError):
        gpu_idx = -1
    sm = round(float(gpu.get("gpu_util_pct", 0.0) or 0.0), 1)
    memu = round(float(gpu.get("gpu_mem_util_pct", 0.0) or 0.0), 1)
    eff = round(float(gpu.get("gpu_util_effective", 0.0) or 0.0), 1)
    # GTX 1080 idle meldt vaak 0% terwijl CUDA actief kan zijn; cockpit blijft "levend".
    if bool(gpu.get("gpu_ok")) and eff < 1.0:
        eff = 1.0
    return {
        "topic": "system_stats",
        "cpu_pct": round(cpu_pct, 1),
        "ram_pct": round(ram_pct, 1),
        "disk_pct": round(disk_pct, 1),
        "gpu_util_pct": sm,
        "gpu_mem_util_pct": memu,
        "gpu_util_effective": eff,
        "vram_used_mb": round(float(gpu.get("vram_used_mb", 0.0) or 0.0), 1),
        "vram_total_mb": round(float(gpu.get("vram_total_mb", 0.0) or 0.0), 1),
        "gpu_ok": bool(gpu.get("gpu_ok")),
        "gpu_name": str(gpu.get("gpu_name") or ""),
        "gpu_index": gpu_idx,
    }


def get_system_stats() -> dict[str, Any]:
    """Cached snapshot to reduce frequent nvidia-smi subprocess overhead."""
    ttl = max(0.5, float(os.getenv("SYSTEM_STATS_CACHE_SEC", "2.0") or 2.0))
    now = time.time()
    cache = getattr(get_system_stats, "_cache", None)
    if isinstance(cache, dict):
        ts = float(cache.get("ts", 0.0) or 0.0)
        if (now - ts) < ttl and isinstance(cache.get("data"), dict):
            return dict(cache["data"])
    data = collect_system_stats()
    setattr(get_system_stats, "_cache", {"ts": now, "data": dict(data)})
    return data
