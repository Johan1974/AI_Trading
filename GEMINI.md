# Project: AI Trading Bot - Core Architecture (Updated April 2026)

## 🖥️ System Environment
- **Hardware:** 62GB RAM | NVIDIA GPU (CUDA enabled).
- **Orchestration:** Docker Compose (Microservices).
- **Disk Status:** Cleaned (reclaimed 260GB). High-speed build caching active via `run_bot.sh`.

## 🏗️ Architecture & Services
1. **Portal (The Waiter):** - **Role:** FastAPI Web UI & API Gateway.
   - **Constraint:** Lightweight. Handles user interaction and monitoring.
2. **Worker (The Chef):** - **Role:** Heavy-duty AI/Trading Engine.
   - **Entry Point:** `app/worker_entry.py`.
   - **Tech Stack:** PyTorch, CUDA, Trading Logic.
   - **STRICT CONSTRAINT:** No FastAPI or Starlette dependencies. Must remain "headless".
3. **Redis (The Intercom):** - **Role:** Pub/Sub & State management between Portal and Worker.

## 🛠️ Development Rules & Coding Standards
- **Configuration:** Always use `pydantic-settings` for `BaseSettings`. 
- **Imports:** Never import `fastapi` or `HTTPException` inside Worker-related services (`app/services/`). Use standard Python `ValueError` or `Exception`.
- **Memory:** Optimize for 62GB RAM. Prefer in-memory caching for market data where possible.
- **Builds:** Do not modify the hashing logic in `run_bot.sh` without checking the impact on build speed.

## 🎯 Current Sprint: The Great Decoupling
1. **Pydantic Fix:** Migrate all `BaseSettings` imports from `pydantic` to `pydantic-settings`.
2. **Worker Purge:** Remove all remaining web-framework traces from the Worker's execution path.
3. **Bash Cleanup:** Fix syntax error in `run_bot.sh` (Line 60: remove "tenzij 'rebuild'").
4. **GPU Optimization:** Ensure PyTorch correctly detects the NVIDIA GPU within the Docker container.