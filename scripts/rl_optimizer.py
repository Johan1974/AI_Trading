#!/usr/bin/env python3
"""
RL Optimizer — monitort trainer/worker logs, analyseert via Ollama (qwen2.5-coder:7b),
schrijft hyperparameter-overrides naar data/optimizer_overrides.json en start de
validator automatisch als VRAM-headroom voldoende is.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import docker
import redis

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",        "http://ollama:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",       "qwen2.5-coder:7b")
REDIS_URL         = os.getenv("REDIS_URL",          "redis://redis:6379/0")
OVERRIDES_FILE    = Path(os.getenv("OVERRIDES_FILE","/app/data/optimizer_overrides.json"))
REPORT_FILE       = Path(os.getenv("REPORT_FILE",   "/app/data/optimization_report.txt"))
VRAM_TOTAL        = int(os.getenv("VRAM_TOTAL_BYTES",  str(8  * 1024**3)))  # GTX 1080 = 8 GiB
VRAM_HEADROOM     = int(os.getenv("VRAM_HEADROOM_BYTES",str(1 * 1024**3)))  # 1 GiB minimum
POLL_SEC          = int(os.getenv("OPTIMIZER_POLL_SEC", "300"))             # 5 min
LOG_TAIL          = int(os.getenv("OPTIMIZER_LOG_TAIL", "400"))

TRAINER   = "ai-trading-trainer"
WORKER    = "ai-trading-worker"
VALIDATOR = "ai-trading-validator"

MIN_LR, MAX_LR, DEFAULT_LR = 5e-6, 5e-4, 1e-4


# ── Ollama helpers ──────────────────────────────────────────────────────────────
def _ollama_generate(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "")


def _vram_used() -> int:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/ps")
        with urllib.request.urlopen(req, timeout=8) as r:
            return sum(int(m.get("size_vram") or 0) for m in (json.loads(r.read()).get("models") or []))
    except Exception:
        return 0


def _vram_free() -> int:
    return max(0, VRAM_TOTAL - _vram_used())


# ── Docker log extraction ───────────────────────────────────────────────────────
def _logs(client: docker.DockerClient, name: str) -> str:
    try:
        return client.containers.get(name).logs(tail=LOG_TAIL, timestamps=False).decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[UNAVAILABLE: {exc}]"


# ── Log parsing ─────────────────────────────────────────────────────────────────
def _parse_trainer(logs: str) -> dict:
    cycles  = logs.count("PPO chunk OK")
    errors  = logs.count("Training mislukt")
    times   = [float(x) for x in re.findall(r"PPO chunk OK.*?(\d+\.\d+)s", logs)]
    saves   = logs.count("Canonical weights opgeslagen")
    return {
        "cycles":       cycles,
        "errors":       errors,
        "saves":        saves,
        "avg_cycle_s":  round(sum(times) / len(times), 1) if times else 0.0,
    }


def _trend(series: list[float]) -> str:
    if len(series) < 4:
        return "insufficient_data"
    mid  = len(series) // 2
    a    = sum(series[:mid]) / mid
    b    = sum(series[mid:]) / (len(series) - mid)
    if abs(b - a) < 1e-4:
        return "flat"
    return "improving" if b > a else "declining"


def _parse_rewards(logs: str) -> dict:
    rewards   = [float(x) for x in re.findall(r"reward[=:\s]+(-?[\d.]+)", logs, re.I)]
    drawdowns = [float(x) for x in re.findall(r"drawdown[=:\s]+([\d.]+)", logs, re.I)]
    losses    = [float(x) for x in re.findall(r"(?:train[/_]?loss|value_loss)[=:\s]+(-?[\d.]+)", logs, re.I)]
    return {
        "count":         len(rewards),
        "mean":          round(sum(rewards) / len(rewards), 5) if rewards else None,
        "last5":         rewards[-5:] if rewards else [],
        "reward_trend":  _trend(rewards),
        "dd_mean":       round(sum(drawdowns) / len(drawdowns), 4) if drawdowns else None,
        "loss_trend":    _trend(losses),
    }


def _redis_stats(r: redis.Redis) -> dict:
    try:
        raw = r.get("ai_trading_snapshot")
        if not raw:
            return {}
        tenant = json.loads(raw).get("tenant", {})
        rl = tenant.get("last_training_stats", tenant.get("rl_stats", {}))
        return {
            "learning_rate":       rl.get("learning_rate"),
            "global_step_count":   rl.get("global_step_count"),
            "exploration_rate_pct":rl.get("exploration_rate_pct"),
        }
    except Exception:
        return {}


# ── Ollama analyse ──────────────────────────────────────────────────────────────
def _analyse(t: dict, rw: dict, rs: dict) -> dict:
    lr    = float(rs.get("learning_rate") or DEFAULT_LR)
    steps = rs.get("global_step_count") or 0

    prompt = f"""You are a reinforcement learning expert analyzing a PPO trading agent.

Current state:
- Training cycles OK: {t['cycles']}, errors: {t['errors']}, model saves: {t['saves']}
- Global steps: {steps}
- Current learning_rate: {lr:.2e}
- Reward trend: {rw['reward_trend']} (mean={rw['mean']}, last5={rw['last5']})
- Drawdown mean: {rw['dd_mean']}
- Loss trend: {rw['loss_trend']}

Reply ONLY with valid JSON, no markdown, no explanation outside the JSON:
{{
  "verdict": "CONVERGING" or "STAGNATING",
  "reason": "<one sentence>",
  "action": "none" or "decrease_lr" or "increase_lr" or "adjust_reward_scale",
  "new_learning_rate": <float between {MIN_LR} and {MAX_LR}>,
  "new_pnl_scale": <float between 10.0 and 40.0, default 20.0>
}}

Decision rules:
- STAGNATING = reward_trend is flat or declining for multiple cycles
- If STAGNATING and lr > 1e-4: recommend decrease_lr by 40%
- If STAGNATING and lr <= 1e-5: recommend increase_lr by 100%
- If reward_trend = improving: verdict=CONVERGING, action=none
- Only recommend adjust_reward_scale if both reward AND loss trend are flat"""

    raw = _ollama_generate(prompt)
    match = re.search(r"\{[\s\S]*?\}", raw)
    if not match:
        return {"verdict": "UNKNOWN", "reason": "parse_error", "action": "none",
                "new_learning_rate": lr, "new_pnl_scale": 20.0, "raw": raw[:300]}
    try:
        out = json.loads(match.group())
        out.setdefault("verdict", "UNKNOWN")
        out.setdefault("action", "none")
        out["new_learning_rate"] = float(max(MIN_LR, min(MAX_LR, float(out.get("new_learning_rate") or lr))))
        out["new_pnl_scale"]     = float(max(10.0,  min(40.0,  float(out.get("new_pnl_scale") or 20.0))))
        return out
    except Exception as exc:
        return {"verdict": "UNKNOWN", "reason": str(exc), "action": "none",
                "new_learning_rate": lr, "new_pnl_scale": 20.0}


# ── Override schrijven ──────────────────────────────────────────────────────────
def _apply(analysis: dict) -> None:
    action = analysis.get("action", "none")
    if action == "none":
        return

    overrides: dict = {}
    try:
        if OVERRIDES_FILE.exists():
            overrides = json.loads(OVERRIDES_FILE.read_text())
    except Exception:
        pass

    ts = datetime.now(timezone.utc).isoformat()

    if action in ("decrease_lr", "increase_lr"):
        overrides["learning_rate"]  = analysis["new_learning_rate"]
        overrides["lr_reason"]      = analysis.get("reason", "")
        overrides["lr_applied_at"]  = ts

    if action == "adjust_reward_scale":
        overrides["pnl_scale"]           = analysis["new_pnl_scale"]
        overrides["pnl_scale_reason"]    = analysis.get("reason", "")
        overrides["pnl_scale_applied_at"]= ts

    OVERRIDES_FILE.write_text(json.dumps(overrides, indent=2))
    print(f"[optimizer] override geschreven: {action} → {OVERRIDES_FILE}", flush=True)


# ── Validator activeren ─────────────────────────────────────────────────────────
def _start_validator(client: docker.DockerClient, free: int) -> bool:
    if free < VRAM_HEADROOM:
        return False
    try:
        c = client.containers.get(VALIDATOR)
        if c.status in ("exited", "created"):
            c.start()
            print(f"[optimizer] validator gestart (VRAM vrij: {free // 1024**2}MB)", flush=True)
            return True
        return False
    except docker.errors.NotFound:
        return False
    except Exception as exc:
        print(f"[optimizer] validator start mislukt: {exc}", flush=True)
        return False


# ── Report ──────────────────────────────────────────────────────────────────────
def _report(t: dict, rw: dict, rs: dict, analysis: dict, free: int, validator: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = "\n".join([
        f"=== {ts} ===",
        f"VRAM vrij    : {free // 1024**2} MB / {VRAM_TOTAL // 1024**2} MB",
        f"Trainer      : {t['cycles']} cycles OK | {t['errors']} errors | {t['saves']} saves | {t['avg_cycle_s']}s/cycle",
        f"Reward       : trend={rw['reward_trend']} mean={rw['mean']} last5={rw['last5']}",
        f"Drawdown     : {rw['dd_mean']}",
        f"Loss         : trend={rw['loss_trend']}",
        f"LR           : {rs.get('learning_rate')} | steps={rs.get('global_step_count')}",
        f"Verdict      : {analysis.get('verdict')} — {analysis.get('reason')}",
        f"Actie        : {analysis.get('action')}",
        f"Nieuwe LR    : {analysis.get('new_learning_rate')}",
        f"PnL scale    : {analysis.get('new_pnl_scale')}",
        f"Validator    : {'gestart' if validator else 'niet gestart'}",
        "",
    ])
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "a") as f:
        f.write(block + "\n")
    print(f"[optimizer] rapport bijgewerkt: {REPORT_FILE}", flush=True)


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[optimizer] start | poll={POLL_SEC}s | VRAM headroom={VRAM_HEADROOM // 1024**2}MB", flush=True)
    dclient = docker.from_env()
    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    while True:
        try:
            free = _vram_free()
            print(f"[optimizer] VRAM vrij: {free // 1024**2}MB", flush=True)

            trainer_logs  = _logs(dclient, TRAINER)
            worker_logs   = _logs(dclient, WORKER)

            t_stats  = _parse_trainer(trainer_logs)
            rw_stats = _parse_rewards(trainer_logs + "\n" + worker_logs)
            rs_stats = _redis_stats(r)

            print(f"[optimizer] trainer={t_stats} reward_trend={rw_stats['reward_trend']}", flush=True)

            if free >= VRAM_HEADROOM:
                print("[optimizer] Ollama analyse starten...", flush=True)
                analysis = _analyse(t_stats, rw_stats, rs_stats)
                print(f"[optimizer] verdict={analysis['verdict']} actie={analysis['action']}", flush=True)
                _apply(analysis)
                validator_started = _start_validator(dclient, free)
            else:
                analysis = {"verdict": "SKIPPED", "reason": "te weinig VRAM", "action": "none",
                            "new_learning_rate": DEFAULT_LR, "new_pnl_scale": 20.0}
                validator_started = False
                print(f"[optimizer] analyse overgeslagen: VRAM < {VRAM_HEADROOM // 1024**2}MB", flush=True)

            _report(t_stats, rw_stats, rs_stats, analysis, free, validator_started)

        except Exception as exc:
            print(f"[optimizer] fout: {exc}", flush=True)

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
