# 🧠 MEMORY - AI Trading Bot Technical State & Protocol

**Last updated:** 2026-05-10
**Project Root:** `/home/johan/AI_Trading` (Referenced as `PROJECT_ROOT` or `.`)

---

## ⚡ AI OPERATIONAL PROTOCOL (Token Diet)
*Volg deze regels strikt om API-kosten te minimaliseren en snelheid te maximaliseren.*

### Output & Communication
- **Extreme Conciseness:** Geen beleefdheidsvormen, geen introducties ("Ik ga nu..."), geen samenvattingen achteraf. Direct technisch antwoord.
- **Silent Success:** Als een actie slaagt zonder dat er een vraag is, antwoord dan enkel met "Done.".
- **No Repetition:** Herhaal nooit bestaande code. Toon enkel de gewijzigde regels (diff-formaat of snippets).
- **No Images:** Geen verwijzingen naar screenshots of visuele data in de tekst. Alleen tekst en code.

### File & Path Handling
- **Relative Paths:** Gebruik altijd paden relatief aan `PROJECT_ROOT`. Nooit absolute paden uitschrijven.
- **Targeted Reading:** Bij bestanden >200 regels: gebruik line-ranges (`grep` eerst voor locatie). Nooit het hele bestand inladen als het niet nodig is.
- **Single Source of Truth:** Gebruik `.clinerules` (gesymlinkt naar `CLAUDE.md`) voor alle AI-instructies.

### Memory Management
- **Chat Reset:** Als de chatgeschiedenis >20 berichten bevat, herinner de gebruiker eraan om de chat te wissen na het updaten van dit `MEMORY.md` bestand.

---

## 🚀 CURRENT TECHNICAL STATE (Helsinki Node)

### Core Engine & Intelligence
- **Regime Logic:** `decision_threshold_regime_boost` (0.05) actief bij **ATR₁₄ > 24-bar gem.** (1h).
- **RL Observation:** 22 features (17 state + 5 account). Nieuwe features: `bid_ask_spread` & `orderbook_imbalance`.
- **Inference:** Worker draait multi-inference (`RL_MULTI_INFER_CONCURRENCY=1`). snapshots naar `predict_rl_feature_snapshots`.
- **Compute:** Voorkeur `cuda:0` (NVIDIA runtime). Automatische fallback naar `cpu` bij falen zonder crash.

### Database & Persistence
- **Trade History:** SQLite `trade_history.db`. Dubbele `opened` posities worden in de UI samengevouwen.
- **State Capture:** `brain_state_json` automatisch gevuld via `brain_state_capture.py`.
- **Storage Policy:** Prijzen, nieuws en features uitsluitend als **Parquet**. Automatische pruning na 30 dagen (`scripts/prune_storage.py`).

### Portal & UI (Genesis Noir Quant)
- **Layout:** Bento-grid (Monitor / Signals / Visual / Action). 
- **Tech:** FastAPI + WebSocket (`/ws/system-stats`, `/ws/trading-updates`).
- **Hardware:** GPU monitoring via `nvidia-smi` (gecached via `SYSTEM_STATS_CACHE_SEC`).

---

## 🛠 ARCHITECTURE & GOVERNANCE
- **Portability:** Geen hardcoded paden. `PROJECT_ROOT` wordt ingeladen via `.env`.
- **Safety:** `RiskManager` dwingt max 3% budget per trade en SL/TP checks af buiten de RL-omgeving.
- **Watchdog:** Engine heartbeat monitor (`WATCHDOG_STALL_LIMIT_SEC=300`) herstart container bij stalls.
- **Audit:** Elk uur zelfreflectie-loop; past thresholds aan op basis van Profit Factor en Win-Rate.

---

## 🔴 ACTIVE ISSUES & TASKS
1. **BUG:** `ERR_EMPTY_RESPONSE` op route `:8000/api/v1/predictions?symbol=TON-EUR:1`.
   - *Status:* Onderzoek naar data-flow tussen `worker.py` en `portal.py` loopt.
2. **Refactoring:** Alle resterende absolute paden in de Python-code vervangen door `PROJECT_ROOT` configuratie.
3. **Optimimalisatie:** Verifiëren of `.clineignore` en `.claudeignore` (gesymlinkt) alle data-ruis correct filteren.

---

## 📈 NEXT STEPS
- [ ] Debuggen van `read_worker_portal_snapshot` functie in `portal.py`.
- [ ] Verifiëren of `TON-EUR` snapshot correct wordt weggeschreven door de worker.
- [ ] Testen van de nieuwe "Silent Success" modus tijdens kleine code-fixes.

---
> **AI Note:** Lees bij elke nieuwe sessie dit bestand en de `.clinerules` om de context te herstellen. Doe geen aannames over paden buiten de `PROJECT_ROOT`.