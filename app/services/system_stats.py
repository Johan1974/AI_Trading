"""
Bestand: app/services/system_stats.py
Functie: CPU/RAM/disk via psutil; GPU via één achtergrond-poll van ``nvidia-smi`` (cache, geen subprocess per API-call).
"""

from __future__ import annotations

import csv
import io
import os
import psutil
import shutil
import subprocess
import threading
import time
from typing import Any

_stats_cache_lock = threading.Lock()
_stats_bg_thread: threading.Thread | None = None

# Warmup: initialiseert de interne psutil-teller zodat interval=None niet 0.0 teruggeeft.
try:
    psutil.cpu_percent(interval=None)
except Exception:
    pass


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


_gpu_stats_throttle_lock = threading.Lock()
_gpu_stats_last_ts: float = 0.0
_gpu_stats_cached: dict[str, Any] | None = None
_gpu_poller_thread: threading.Thread | None = None
_gpu_poller_started_lock = threading.Lock()


def _gpu_poll_interval_sec() -> float:
    """Centrale GPU-poll: voorkeur ``SYSTEM_STATS_GPU_POLL_INTERVAL_SEC``, anders legacy ``SYSTEM_STATS_GPU_MIN_INTERVAL_SEC``."""
    try:
        raw = str(os.getenv("SYSTEM_STATS_GPU_POLL_INTERVAL_SEC", "") or "").strip()
        if raw:
            v = float(raw)
            return max(5.0, min(3600.0, v))
    except (TypeError, ValueError):
        pass
    try:
        leg = float(os.getenv("SYSTEM_STATS_GPU_MIN_INTERVAL_SEC", "60") or 60.0)
    except (TypeError, ValueError):
        leg = 60.0
    return max(5.0, min(3600.0, leg))


def _gpu_stats_empty() -> dict[str, Any]:
    return {
        "gpu_util_pct": 0.0,
        "gpu_mem_util_pct": 0.0,
        "gpu_util_effective": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "gpu_ok": False,
        "gpu_name": "",
        "gpu_index": -1,
    }


def _gpu_poller_loop() -> None:
    """Eén achtergrondlus: periodiek ``nvidia-smi`` (via ``_nvidia_smi_stats``), geen subprocess vanuit API-hot-path."""
    global _gpu_stats_last_ts, _gpu_stats_cached
    interval = _gpu_poll_interval_sec()
    while True:
        try:
            fresh = _nvidia_smi_stats()
            with _gpu_stats_throttle_lock:
                _gpu_stats_last_ts = time.time()
                _gpu_stats_cached = dict(fresh)
        except Exception:
            pass
        time.sleep(interval)


def _ensure_gpu_poller_started() -> None:
    global _gpu_poller_thread
    if not shutil.which("nvidia-smi"):
        return
    with _gpu_poller_started_lock:
        if _gpu_poller_thread is not None and _gpu_poller_thread.is_alive():
            return
        th = threading.Thread(target=_gpu_poller_loop, daemon=True, name="nvidia-smi-poller")
        _gpu_poller_thread = th
        th.start()


def _nvidia_smi_stats_from_poller_cache() -> dict[str, Any]:
    """Leest alleen cache gevuld door GPU-poller; start poller bij eerste gebruik."""
    _ensure_gpu_poller_started()
    with _gpu_stats_throttle_lock:
        if isinstance(_gpu_stats_cached, dict):
            return dict(_gpu_stats_cached)
    return _gpu_stats_empty()


def _disk_usage_percent() -> float:
    """Schijf%: standaard container-root; zet SYSTEM_STATS_DISK_PATH=/hostfs + bind-mount host `/` voor echte host-schijf."""


    path = str(os.getenv("SYSTEM_STATS_DISK_PATH", "/") or "/").strip() or "/"
    for candidate in (path, "/"):
        try:
            return float(psutil.disk_usage(candidate).percent)
        except Exception:
            continue
    return 0.0


def _collect_psutil_cpu_ram_disk() -> tuple[float, float, float]:
    """CPU/RAM/disk zonder nvidia-smi (geschikt voor hot paths zoals /predict)."""
    cpu_pct = 0.0
    ram_pct = 0.0
    disk_pct = 0.0
    try:
        cpu_pct = float(psutil.cpu_percent(interval=None))
        ram_pct = float(psutil.virtual_memory().percent)
        disk_pct = _disk_usage_percent()
    except Exception:
        pass
    if cpu_pct < 0.1:
        try:
            load1, _, _ = os.getloadavg()
            n = max(1, (os.cpu_count() or 1))
            cpu_pct = min(100.0, (float(load1) / float(n)) * 100.0)
        except Exception:
            pass
    return cpu_pct, ram_pct, disk_pct


def collect_system_stats_light_no_gpu() -> dict[str, Any]:
    """Zelfde payload-vorm als ``collect_system_stats`` maar zonder nvidia-smi (blokkeert niet)."""
    cpu_pct, ram_pct, disk_pct = _collect_psutil_cpu_ram_disk()
    return {
        "topic": "system_stats",
        "cpu_pct": round(cpu_pct, 1),
        "ram_pct": round(ram_pct, 1),
        "disk_pct": round(disk_pct, 1),
        "gpu_util_pct": 0.0,
        "gpu_mem_util_pct": 0.0,
        "gpu_util_effective": 0.0,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "gpu_ok": False,
        "gpu_name": "",
        "gpu_index": -1,
    }


def collect_system_stats() -> dict[str, Any]:
    """Eén snapshot voor WebSocket/API: CPU%, RAM%, disk% root, GPU%, VRAM (MB)."""
    cpu_pct, ram_pct, disk_pct = _collect_psutil_cpu_ram_disk()

    gpu = _nvidia_smi_stats_from_poller_cache()
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


def _bg_refresh_system_stats_cache() -> None:
    """Volledige snapshot inclusief nvidia-smi; draait op achtergrondthread."""
    try:
        data = collect_system_stats()
        with _stats_cache_lock:
            setattr(get_system_stats, "_cache", {"ts": time.time(), "data": dict(data)})
    except Exception:
        pass


def _spawn_stats_refresh_if_idle() -> None:
    global _stats_bg_thread
    with _stats_cache_lock:
        if _stats_bg_thread is not None and _stats_bg_thread.is_alive():
            return
        th = threading.Thread(target=_bg_refresh_system_stats_cache, daemon=True, name="system-stats-nvidia-refresh")
        _stats_bg_thread = th
    th.start()


def get_system_stats() -> dict[str, Any]:
    """
    Cached snapshot. Bij cache-miss of TTL-verloop: retourneer laatste bekende data of een lichte
    CPU/RAM/disk-snapshot **zonder** nvidia-smi op de aanroepende thread; GPU-waarden worden
    op de achtergrond bijgewerkt (P1: /predict en API niet laten blokkeren op subprocess).
    """
    ttl = max(0.5, float(os.getenv("SYSTEM_STATS_CACHE_SEC", "2.0") or 2.0))
    now = time.time()
    with _stats_cache_lock:
        cache = getattr(get_system_stats, "_cache", None)
    if isinstance(cache, dict):
        ts = float(cache.get("ts", 0.0) or 0.0)
        data = cache.get("data")
        if (now - ts) < ttl and isinstance(data, dict):
            return dict(data)
        if isinstance(data, dict):
            _spawn_stats_refresh_if_idle()
            return dict(data)
    light = collect_system_stats_light_no_gpu()
    with _stats_cache_lock:
        setattr(get_system_stats, "_cache", {"ts": now, "data": dict(light)})
    _spawn_stats_refresh_if_idle()
    return dict(light)
