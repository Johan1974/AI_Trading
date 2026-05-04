# 🛸 AI Trading Bot - Core Architecture & AI Mandate (2026)

## 🖥️ System Environment
- **Hardware:** 62GB RAM | NVIDIA GPU (CUDA enabled).
- **Orchestration:** Docker Compose (Microservices).
- **Timezone:** `Europe/Amsterdam` (Strikt verplicht voor alle logs en logica).
- **Disk Status:** High-speed build caching via `run_bot.sh`.
- **Log Hub:** `./_logs_hub/` (Centrale opslag voor alle service-logs).

---

## 🏗️ Architecture & Services
1. **Portal (The Waiter):** FastAPI Web UI & API Gateway.
   - **UI Standard:** Flexbox-based layout. No `position: absolute` voor sidebar labels.
2. **Worker (The Chef):** Headless AI/Trading Engine (`app/worker_entry.py`).
   - **Constraint:** Geen FastAPI dependencies. Gebruik `psutil` voor hardware telemetry.
3. **Redis (The Intercom):** Pub/Sub & State Management. 
   - **Safety:** Altijd `if data is not None` checks uitvoeren op Redis-responses.
4. **Dashboard Validator (The Executioner):** Playwright Watchdog (`app/tests/run_ui_tests.py`).
   - **Role:** Voert de Deep-Scan uit en bewaakt de integriteit van de hele keten.

---

## 📜 Coding Standards & File Header Mandate

**MANDAAT:** Elk bestand dat door Gemini wordt aangemaakt of gewijzigd, **MOET** beginnen met dit commentaarblok:

```python
"""
BESTANDSNAAM: [Volledig pad/naar/bestand]
FUNCTIE: [Heldere beschrijving van de taak en de rol binnen de trading-architectuur]
"""
```

### UI & Logic Rules:
- **Configuration:** Uitsluitend `pydantic-settings` voor `BaseSettings`.
- **Error Prevention:** Imports (`os`, `psutil`, `time`) altijd bovenaan. Geen `NameError` of `AttributeError` toegestaan.
- **CSS Standards:** `display: flex` met `flex-direction: column`. Gebruik `height: auto !important` voor dynamische panel-noir kaders.

---

## 🧠 Lessons Learned & Harde Mandaten
1. **Timezone (Amsterdam):** Amsterdam is te allen tijde de default. Gebruik `from app.datetime_util import UTC` (compatibiliteit Python 3.10).
2. **Asynchrone UI:** Test-scripts moeten expliciet wachten (`page.wait_for_function`) op dynamische elementen zoals `#marketSelect`.
3. **Frozen Data:** Als prijzen of CPU-load gedurende een uur exact gelijk blijven, wordt dit gemarkeerd als een systeem-crash (Stall).

---

## ⚖️ HET BINDEND MANDAAT: OMNI-TEST & AUDIT-FIRST

**Gemini mag NOOIT aannemen dat code werkt. Alleen de `dashboard-validator` container is de bron van waarheid. We testen tot op de punten en de komma's.**

### 1. Acceptatie-criterium
Een taak is pas **'VOLTOOID'** als `_logs_hub/last_audit_report.json` de status **`SUCCESS`** of **`OK`** heeft.

### 2. Het Omni-Test Protocol
- **Deep-Scan Tour:** De validator klikt door Terminal, AI Brain, Ledger en Hardware tabs.
- **Precisie-Eis:** - Prijzen moeten numeriek, actueel en correct geformatteerd zijn.
    - Tabs moeten binnen 2 seconden volledig geladen zijn met data (geen `--` of `0.00`).
- **Multi-Ticker & Profitability:** Verifieer dat niet alleen BTC, maar ook de door de scanner geselecteerde 'meest winstgevende' munten (ETH, SOL, etc.) zichtbaar zijn in de dropdown en UI.
- **Regressie-Preventie:** Nieuwe features mogen oude functionaliteit nooit verbreken.

### 3. Diagnose-keten bij Fout (FAILED)
Als een audit faalt, volgt Gemini direct deze hiërarchie:
- **STAP A (Worker):** Check `worker_execution.log` (Berekent de AI de data wel?).
- **STAP B (Redis):** Check `redis.log` (Is de cache gevuld met de juiste keys?).
- **STAP C (Portal):** Check `portal_api.log` (Geeft de API de data door?).
- **STAP D (UI):** Check `main.js` (Matcht de ID met de HTML?).

---

## 🎯 Current Sprint: Stability & Profitability
1. **Dynamic Ticker Fix:** De 'Multi-Ticker Failure' oplossen door de `core/scanner.py` winst-logica te koppelen aan de UI.
2. **Hardware Sync:** Volledige synchronisatie tussen header-stats en hardware-tab ringen.
3. **Self-Evolving Audit:** De validator automatisch uitbreiden met checks voor elke nieuwe 'business rule' die we bespreken.

---
*Gemini: Ik ben de validator niet, de container is dat. Kom pas bij de gebruiker terug met 'Klaar' als de validator een groene vlag geeft.*
