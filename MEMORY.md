# MEMORY - AI Trading Bot Technical State

Last updated: 2026-04-21

## Governance Mode
- Full Agency active via `.cursorrules`:
  - challenge risky/suboptimal requests,
  - propose safer alternatives,
  - proactively improve portal UX/performance,
  - future-proof decisions for RL phase,
  - flag logic gaps and edge cases immediately.

## Portal layout & GPU monitoring
- **Terminal `/activity` vs Redis WS:** `/ws/trading-updates` kan lang open blijven zonder pub/sub-berichten. `terminal.js` gebruikt **`lastTradingWsActivityAtMs`**: alleen als er ~20s geen bruikbare payload was, draait de gestaggerde **`refreshActivity()`** poll weer (voorheen werd die uitgezet zodra de socket opende). **`refreshMarkets()`** triggert activity nu non-blocking (`void refreshActivity()`).
- **Hardware (GPU/schijf) in split compose:** de **portal**-image heeft geen `nvidia-smi`; `/ws/system-stats` op de portal gebruikt **`STATE['_system_stats_ws_payload']`** gevuld door Redis-kanaal **`system_stats`** waar de **worker** elke ~2s `collect_system_stats()` publiceert (echte GPU + load). Schijf% gebruikt **`SYSTEM_STATS_DISK_PATH`** (default **`/hostfs`**) met compose-bind **`/:/hostfs:ro`** zodat de meter de **host**-rootvulling toont, niet alleen de dunne container-laag.
- **Genesis Noir Quant:** tabs **Terminal**, **AI Brain**, **Ledger** (trade history + equity/PnL charts), **Hardware** (vier ring-meters + logs). CSS-grid **`repeat(4, minmax(0,1fr))`**; **Inter** als UI-font, **JetBrains Mono** voor data/tabulair; labels min. **14px**; minimalistische **ghost/outline** header-knoppen; hint-bubbles via **`#brainHintPortal`** (**z-index 99999**). Terminal toont **3** top-nieuwsregels; boven de chart een **Elite-8 AI status bar** (`#elite8AiStatusBar`, kleuren uit `/activity` → `elite_ai_signals` i.c.m. `rl_multi_decisions` + whale panic/danger). Intelligence ticker gebruikt **`/api/v1/news/ticker?elite_mix=1`** (round-robin alle Elite-munten) + **`scanner_intel_feed`** (scanner replace-berichten). Markt-switch via dropdown/scanner/pill: **`switchEliteMarket`** → POST **`/markets/select`** + chart + Brain Lab + ticker refresh.
- **AI Brain Lab:** kolom 1 = Monitor + RL stats; kolom 2 = reasoning + state (o.a. MACD); kolom 3 = reward/feature weights/benchmark; kolom 4 = balance/risk/paper. Hints via **`initHintPortals()`** op alle tab-roots → **`#brainHintPortal`** (fixed, z-index ~2e9).
- Dashboard cards hebben nu geforceerde contrast-layout: `border: 2px solid #333`, `border-radius: 8px`, `margin: 10px`, `background: #0b0e11`; titelbalken met eigen donkere headerstrip per panel/chart-box (`.chart-box-titlebar` voor ledger charts).
- **Card-headers** met achtergrond `#111` (Electric Quant) voor sectiescheiding; hoofdprijschart achtergrond **#000000**, tekst **#FFFFFF**; neon-lijn **lineWidth 4**.
- **Dockerfile:** **`nvidia/cuda:12.4.1-runtime-ubuntu22.04`**, **`ENTRYPOINT []`** (NVIDIA-default entrypoint verstopte torch &lt;2.4 en brak `transformers`). Daarna **`torch==2.6.0+cu124`** (`--index-url https://download.pytorch.org/whl/cu124`) — nodig voor **transformers 5.x** (CVE/torch.load-check) en **stable-baselines3**. **docker-compose:** `gpus: all`, **`GENESIS_REQUIRE_GPU`**, **`FINBERT_USE_CUDA`**. Logs: **`[DEVICE] Using device: cuda:0 (...)`**, **`[DASHBOARD] Dashboard live op poort 8000`**, **`[ENGINE] Tick | torch_device=cuda:0`**. GPU-%: `nvidia-smi` in `system_stats.py`.
- Ubuntu image levert **Python 3.10**; `datetime.UTC` is pas 3.11+. Code gebruikt `app/datetime_util.UTC` (`timezone.utc`) + scripts met try/except zodat de API niet meer crasht op import.
- `system_stats._nvidia_smi_stats`: lossere parsing van %/komma’s; extra `utilization.gpu`-query per GPU-index.
- `app/services/system_stats.py` gebruikt nu TTL-cache op `get_system_stats()` (env `SYSTEM_STATS_CACHE_SEC`, default 2s) om `nvidia-smi` subprocess-load te verlagen.

## Risk engine (position sizing & SL/TP)
- `core/risk_manager.py`: `RiskManager` leest o.a. `RISK_SIZING_MODE` (`fixed_eur` | `equity_pct`), `RISK_BASE_TRADE_EUR`, `RISK_MAX_TRADE_EQUITY_PCT`, `RISK_MAX_POSITION_EQUITY_PCT`, `RISK_STOP_LOSS_PCT`, `RISK_TAKE_PROFIT_PCT`.
- `calculate_trade_size()` → fractie/notional t.o.v. equity/cash; `check_safety()` blokkeert BUY bij te hoge exposure of te weinig cash; `hard_exit_for_sl_tp()` dwingt SELL buiten RL op basis van gewogen gemiddelde entry uit `open_lots`.
- Paper cycle (`/paper/run`): RL levert richting; `CORE_RISK` zet omvang; spread/sentiment blijft via `app.services.risk.RiskManager`; `/activity` bevat `risk_profile` voor de UI (kolom 4 „Risk profile”).
- `compute_risk_controls()` delegeert naar `core.risk_manager.risk_controls_for_close` (zelfde SL/TP-% als harde exits).

## Current Runtime Modes
- `LIVE_MODE` from vault, default fallback is paper mode when missing.
- `DRY_RUN` supported; when true, order calls are logged/simulated.
- **Operationeel:** vault via **`scripts/lib_compose_root.sh`** (of per-service **`./scripts/run_*.sh`**); stack met **`docker compose up -d`** vanuit repo-root na vault laden.
- Eén gedeelde **FinBERT**-instantie voor `SignalEngine` en `NewsMappingService`; `FINBERT_USE_CUDA=0` in compose om PyTorch/CUDA-driver-warnings te vermijden tenzij je driver expliciet matcht.
- **Tijdzone:** `ai-trading-bot` mount `/etc/localtime` en `/etc/timezone` read-only zodat container-CET/CEST gelijk loopt met de Linux-host; `TZ=Europe/Amsterdam` blijft in compose. Telegram start/stop gebruikt `pytz.timezone('Europe/Amsterdam')` voor tijdstempels.
- Compose / scripts:
  - Stack: `source ./scripts/lib_compose_root.sh` dan `docker compose up -d`; portal op `http://localhost:8000`
  - Per image: `./scripts/run_{redis,worker,portal}.sh [build|rebuild]`
  - `docker-compose.yml`: **`restart: always`** op services — na crash/reboot start Docker ze opnieuw (tenzij verwijderd).
  - Geen `.env` in de repo; secrets via vault vóór `docker compose`.

## Secret Management
- Secrets sourced from: `$HOME/.trading_vault`
- Expected key groups:
  - Read-only: `BITVAVO_KEY_READ`, `BITVAVO_SECRET_READ`
  - Trading: `BITVAVO_KEY_TRADE`, `BITVAVO_SECRET_TRADE`
  - Telegram lifecycle alerts (optioneel): `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` — in de shell vóór `docker compose` (via vault + `lib_compose_root.sh`) of via `docker-compose.yml` environment.
- Bij succesvolle FastAPI lifecycle: `TELEGRAM.send_start()` na engine-start; `TELEGRAM.send_stop()` bij shutdown (o.a. `docker compose stop`).

## Exchange Integrations
- Bitvavo REST client implemented:
  - signed requests
  - rate-limit aware behavior
  - retry/backoff on 429
  - optional `clientOrderId` for idempotent order flow
- Bitvavo rate-limit manager implemented (`app/exchanges/bitvavo_manager.py`):
  - parses:
    - `bitvavo-ratelimit-limit`
    - `bitvavo-ratelimit-remaining`
    - `bitvavo-ratelimit-resetat`
  - auto-throttle at 80% usage
  - hourly API usage logging (`requests`, `weight`, `errors`, `last_status`)
  - circuit breaker on repeated failures:
    - threshold-based fail counting inside rolling time window
    - temporary open state that blocks new requests until cooldown
- Global shared manager active across Bitvavo client instances for centralized tracking.
- Bitvavo auth connectivity validated for both READ and TRADE key sets.
- Concrete class fix:
  - `BitvavoClient` now correctly implements abstract methods inside class scope:
    - `get_balance()`
    - `place_market_order()`
  - resolves runtime instantiation error for `check_pair_balance_from_vault`.
- Paper-mode execution manager in `app/exchanges/bitvavo_manager.py`:
  - `BitvavoPaperManager.get_balance()`
  - `BitvavoPaperManager.place_market_order()`
  - persists paper balances/orders in SQLite (`data/paper_bitvavo.db` by default)
  - `BitvavoClient` delegates to this manager when `LIVE_MODE` is disabled.

## Whale pressure & whale radar (CryptoCompare)
- `app/services/whale_watcher.py` gebruikt **geen** Whale Alert API (deprecated).
- Whale pressure (0..1) komt uit **CryptoCompare `/data/v2/news/`** headlines: regex op o.a. `whale`, `large transfer`, `billion`, `million btc/eth`.
- **`core/social_engine.py` (mijlpaal Whale Radar):** zelfde news-feed, gefilterd op Elite-8 market bases; schatting USD-notional uit headline; whale move ≥ **1M USD**; richting **inflow** (naar exchange) vs **outflow** (naar cold wallet/withdrawal) uit keyword-heuristiek; `refresh_whale_radar_state` → `STATE["whale_radar_moves"]`, `STATE["whale_flow_by_market"]`; `GET /api/v1/whale/radar` voor de terminal-widget (max 3 items); `refreshWhaleRadar()` wordt mee aangeroepen vanuit `refreshActivity()`.
- Paper ledger: kolom **Social/Whale Context** = `trade_history.ledger_context` (ingevuld bij signal via `format_ledger_social_whale_context` / `build_trade_decision_context`); FIFO-close merged entry+exit context.
- RL: vóór `decide` wordt **whale_pressure** attention-slot geblend met default gewicht **`WHALE_ATTENTION_WEIGHT` = 0.25**; bij conflict **inflow** + sterke bullish price-action + **BUY** wordt confidence vermenigvuldigd (`WHALE_TECH_CONFLICT_CONF_MULT`); bij **outflow** + BUY lichte boost (`WHALE_OUTFLOW_BUY_CONF_BOOST`).
- **Whale Panic Mode (`core/risk_management.py`):** bij elke `refresh_whale_radar_state` worden grote **exchange-inflow** moves (≥ **`WHALE_PANIC_INFLOW_MIN_USD`**, default **5M USD**) in `STATE["whale_inflow_panic_log"]` gezet (dedup). Binnen **`WHALE_PANIC_WINDOW_SEC`** (default **600s**) tellen we per `MARKET`; bij ≥ **`WHALE_PANIC_MIN_INFLOWS`** (default **4** = “meer dan 3”) en **open positie** op die markt voert `_paper_whale_panic_intervention` vóór de RL-stap een **volledige MARKET SELL** uit (`process_signal`), zet **`whale_panic_cooldown_until`** (**60 min** koopblokkade), `record_whale_panic_sell_fired` (re-arm debounce **`WHALE_PANIC_REARM_SEC`**), Telegram **`send_whale_panic_mode`**, en event `whale_panic_sell`. `/api/v1/history` bevat **`whale_danger_zone`** voor de rode **Whale Danger Zone** op de chart (CSS `#priceChart.whale-danger-zone` + optionele price line). `/activity` bevat **`whale_panic_cooldowns`** (ISO tot per markt).
- `CRYPTOCOMPARE_KEY` wordt gebruikt (zelfde key als nieuws); optioneel ook zonder key als de endpoint reageert.
- Bij startup wordt `bot_execution.log` gevuld met `[WHALE-SYNC] Data nu via CryptoCompare feed.`
- `/api/v1/brain/state-overview` ververst `whale_watch` en zet `state.whale_pressure` live voor het AI Brain dashboard.

## Data Integrations
- Price history:
  - yfinance-based historical market fetch
  - Bitvavo candle ingestion for crypto datasets
- Sentiment/macro:
  - CryptoCompare + RSS (Cointelegraph/CoinDesk/Decrypt) nieuwsingestie
  - Crypto Fear & Greed index ingestion
- Data unification:
  - `DataAggregator` builds normalized multi-source DataFrame with volatility/news interaction features
- Storage policy:
  - historical koersdata, nieuwsdata en RL-featureframes worden uitsluitend als Parquet opgeslagen
  - opslaglocatie: `$HOME/AI_Trading/storage` met submappen:
    - `historical_prices`
    - `historical_news`
    - `rl_features`
  - onderhoudsscript: `scripts/prune_storage.py`
    - downsamplet data ouder dan 30 dagen naar compactere minutenbuckets
    - verwijdert RL-irrelevante ruis (near-zero price/sentiment/confidence bewegingen)
- Dynamic market scanner:
  - `app/services/market_scanner.py` pulls active Bitvavo markets + 24h stats
  - filters by `MIN_24H_VOLUME_EUR` (default 500000)
  - `core/scanner.py` **DynamicVolatilityScanner** (Elite-8): **Pillars** = BTC-EUR, ETH-EUR, SOL-EUR + één roterende EUR-pair uit voorkeurslijst die in de **top-10 op 24h-volume** zit; **Movers** = resterende slots uit **top-50 volume**, `SCANNER_MOVER_MIN_VOLUME_EUR` (default 1M), sorteer op **4h high/low swing %**, met **market cap ≥ `SCANNER_MIN_MARKET_CAP_EUR`** (default 500M) via **CoinGecko** `/coins/markets?vs_currency=eur`; zonder CoinGecko-data: striktere **fallback allowlist** (`FALLBACK_LARGE_CAP_BASES`) voor Movers.
  - `_refresh_active_markets_cache` in `app/main.py` zet `active_markets` in **scanner-volgorde** en vult `is_pillar` / `pillar_kind` / optioneel `move_pct_4h`; pairs die onder `MIN_24H_VOLUME_EUR` vallen maar wél in de Elite-8 zitten krijgen een **synthetische rij** zodat ze niet verdwijnen uit de dropdown.
  - Historical quality gate (mijlpaal refinement): Next-5 movers moeten **alle drie** slagen: `close > SMA200` (dag), `momentum_30d_pct > 0`, `listing_days >= 365`; veldset `quality_score` (0..3), `passes_quality`, `is_long_term_downtrend`, `selection_reason`. Stablecoin-bases (o.a. USDT/USDC/DAI) worden uitgesloten als mover.
  - Scanner feedback: `/markets/active` rows bevatten nu scanner-rationale (`selection_reason`) plus quality metadata; notifier Executive Summary toont voortaan per selected pair de reden.
- **Social momentum (`core/news_engine.py`):** periodieke refresh (`SOCIAL_REFRESH_SEC`, default 300s) via `refresh_social_momentum_state` → `STATE["social_momentum_by_market"]` + `STATE["social_buzz_summary"]`. CryptoCompare `data/social/coin/latest` (met `api_key`) parsed voor o.a. Reddit posts/h & active users + Twitter follower delta; **velocity** = max(CC-composite, news-mention 60m) t.o.v. baseline ≥ `SOCIAL_VELOCITY_MIN_AGE_SEC` (~1h). **RL-overlay** alleen voor markets met `is_pillar` of `passes_quality` (`apply_social_overlay_to_rl_row`). REST: `GET /api/v1/social/buzz`; Brain WS/REST bevat `social_buzz`.
- News mapping:
  - `app/services/news_mapping.py` maps headlines to active Bitvavo top-volume coins
  - alias rules include direct BTC mapping (`Bitcoin`, `BTC`, `Halving`)
  - extended keyword matrix for ETH/SOL/XRP/ADA/etc.
  - impact filter only forwards:
    - strong sentiment (`> 0.4` or `< -0.4`), or
    - direct mention of top-20 active volume coins
  - generic `MKT` fallback replaced with concrete high-liquidity coin fallback.
  - mapped records now include modal metadata:
    - `title`, `summary`, `url`, `keywords`, `ai_reasoning`.

## AI Stack
- Technical analyzer:
  - Scikit-learn based trend model (`SklearnTechnicalAnalyzer`)
- Sentiment analyzer:
  - HuggingFace Transformers FinBERT (`FinBertSentimentAnalyzer`)
- Judge:
  - weighted fusion of technical + sentiment (`WeightedJudge`)
- Orchestration:
  - `SignalEngine` coordinates analyzers + judge
- RL autonomy service:
  - `app/rl/agent_rl.py` with `RLAgentService`
  - PPO pre-train/load per pair
  - decision output: `HOLD/BUY/SELL` with confidence + expected reward estimate
  - explainability: per-decision feature weights and natural-language reasoning
  - training monitor now tracks:
    - cumulative reward,
    - episode length,
    - policy entropy,
    - approx_kl,
    - value_loss,
    - global step count.

## RL Stack
- Background PPO-updates (`_rl_background_training_loop`): voor **BTC-EUR** en **ETH-EUR** wordt `RL_TRAIN_CHUNK_STEPS` vermenigvuldigd met **`RL_PRIORITY_PAIR_TRAIN_MULT`** (default 1.65); zelfde factor bij AUTO-OPT micro-finetune na ordening (BTC/ETH eerst in de boost-batch).
- Stable Baselines3 PPO training loop added.
- Custom gymnasium env:
  - reward: EUR pnl delta
  - penalty: drawdown
  - max trades simulation (10,000 target)
- Historical training frame includes major event flags (incl. Bitcoin Halving 2024).
- RL state now includes:
  - price-action,
  - volume change,
  - sentiment score,
  - news confidence,
  - orderbook imbalance (historical proxy),
  - MACD,
  - RSI(14),
  - EMA gap.
- RL model versioning:
  - modellen worden met UTC timestamp opgeslagen
  - per pair wordt een registry bijgehouden op reward-score
  - alleen top-5 best presterende modellen blijven bewaard; overige versies worden automatisch verwijderd.
- Pre-train rule:
  - before first paper cycle per pair, agent pre-trains on ~400 days lookback (vault-driven `LOOKBACK_DAYS`).

## Risk & Execution
- `RiskManager` enforced rules:
  - max 3% budget per trade
  - volatility/spread filter
  - emergency exit on strong negative sentiment
- Dry-run decorator logs trade attempts to `trades_log.csv`
- Daily fictive PnL endpoint includes 0.15% fee assumptions.
- Paper execution is managed by `PaperTradeManager` (`app/services/paper_engine.py`):
  - virtual wallet with configurable start balance (`PAPER_START_BALANCE_EUR`)
  - execution fees fixed at 0.15% per trade leg
  - FIFO lot closing for realized PnL accuracy
  - live Bitvavo reference price used for `XXX-EUR` markets
  - trade history now stores `ai_thought` text for explainable outcomes.
  - ownership guard blocks SELL without position ownership; event is persisted as `CRITICAL_BLOCKED` with `status=critical_blocked`.
  - round-trip ledger API view is available via `GET /api/v1/trades?view=roundtrip` (open time, market, entry/exit, net pnl eur/%, **`ledger_context`** = Social/Whale Context).

## Autonomous AuditEngine
- Hourly self-reflection loop in `app/main.py` (`_audit_engine_loop`) evaluates last 24h outcomes per Elite-8 market from SQLite trade history.
- Metrics per market: `profit_factor`, `win_rate`, `wins`, `losses` via `PaperTradeManager.elite8_audit_metrics(...)`.
- Auto-tuning applies bounded step updates (`+/- 0.05`) to:
  - `STATE["decision_threshold"]`
  - `STATE["stop_loss_pct"]`
- Audit state exported in memory:
  - `AUDIT_LAST_RUN`
  - `AUDIT_LAST_TUNING`
  - `AUDIT_REFLECTIONS` (rolling history)
- Restart/scheduled notifier now includes `AI Zelfreflectie` section with threshold/stop-loss deltas and rationale.

## Autonomous Maintenance Mode
- Watchdog loop in `app/main.py` forces auto-recovery (`os._exit(1)`) when:
  - engine heartbeat is stale beyond `WATCHDOG_STALL_LIMIT_SEC` (default 60s),
  - or websocket heartbeat is stale while websocket clients are connected.
- `docker-compose.yml` already enforces `restart: always`; watchdog exit triggers container self-restart.
- `PaperTradeManager` now persists/loads wallet snapshot in SQLite table `wallet_state` for state restore after restart.
- Email policy updated:
  - periodic mail: daily executive summary at 08:00 Europe/Amsterdam (`daily_restart_report_loop` schedule),
  - urgent-only alerts via `send_urgent_alert()` for API failure streak, insufficient balance trade rejection, and 3% stop-loss threshold.
- UI headless mode added:
  - toggle button `Headless` in terminal chart controls,
  - hides chart/ticker/time controls client-side while trading engine continues server-side on GPU.

## RL Feature Optimization
- New module `core/preprocessor.py` centralizes RL preprocessing:
  - robust min-max scaling to `[0,1]`,
  - tanh(z-score) scaling to `[-1,1]`,
  - dead-signal forward fill (`forward_fill_dead_signal`),
  - lightweight feature attention gate (`attention_gate_weights`).
- `app/rl/data.py` now applies:
  - signal fallback cache per market for `sentiment_score` and `whale_pressure`,
  - forward-fill replacement when API-origin values collapse to `0`,
  - final normalization for all RL observation features before training/inference.
- `build_rl_training_frame(..., metadata_out=...)` now exports `signal_integrity` counters for non-zero sentiment/whale coverage.
- `app/main.py` emits explicit warnings when sentiment/whale channels are effectively empty after preprocessing.
- `app/rl/env.py` and `app/rl/agent_rl.py` apply the same attention-gating transformation to observation features.
- Strategy feature bars in `app/static/js/terminal.js` use a zoom transform (`log10(1 + value * 1000)`) with tooltips showing both raw and zoomed values.

## Autonomous Self-Improvement
- `app/main.py` includes `_autonomous_improvement_loop()`:
  - cadence via `AUTO_OPT_INTERVAL_SEC` (default 3600s),
  - reads 24h Elite-8 metrics (`profit_factor`, `win_rate`) from SQLite rollup,
  - self-adjusts runtime knobs within safety bounds:
    - `RL_EXPLORATION_FINAL_EPS`,
    - `RISK_MAX_PER_ASSET_TRADE_PCT`,
    - `RL_TRAIN_CHUNK_STEPS`.
- Underperformance branch triggers opportunistic micro-finetune (`RL_AGENT.online_update`) in parallel for top active markets.
- Optimizer state exposed in `/api/v1/system/report-status` under `autonomous_optimizer` with last tuning rationale and applied values.
- Optimizer persistence:
  - SQLite table `optimizer_state` stores rolling snapshots (`settings_json`) from autonomous optimizer loop.
  - Startup restores best-known optimizer settings and score history, then reapplies env/runtime knobs.
- Rollback guardrail:
  - if optimizer score degrades for 2 consecutive cycles, runtime settings are reverted to best-known stable settings.
  - score uses combined 24h edge metric (`profit_factor * 100 + win_rate`).

## Monitoring/API
- FastAPI endpoints for health, predict, activity, paper run, and daily dry-run PnL.
- Hot routes `/predict` en `/paper/run` lopen nu via async worker-queues (`PREDICT_QUEUE`, `PAPER_RUN_QUEUE`) zodat blocking IO niet op de request-thread blijft hangen.
- Tenant isolation hardening:
  - `app/services/state.py` gebruikt tenant-scoped runtime state via `ContextVar` + middleware (`x-tenant-id`/`tenant_id`),
  - paper SQLite records zijn tenant-tagged (`tenant_id`) en tenant-filtered voor analytics/ledger/recent trades/optimizer-state.
- Execution realism:
  - orderbook-based frictie actief in paper cycle: spread + slippage uit Bitvavo book (`/v2/book`) beïnvloeden execution price.
- Daily auto-calibration:
  - extra 24h-loop (`_daily_auto_calibration_loop`) stuurt `decision_threshold` en `stop_loss_pct` bij met max stap ±0.05.
- News ticker endpoint:
  - `GET /api/v1/news/ticker` returns mapped high-impact news tuples (`text`, `coin`, `sentiment` + metadata).
- History endpoint:
  - `GET /api/v1/history?pair=...` returns chart labels/prices/markers for selected market pair.
  - crypto pairs (`XXX-EUR`) now use Bitvavo candle history fallback; payload includes `tv_symbol` for chart sync context.
- Trades endpoint:
  - `GET /api/v1/trades?limit=...` returns latest records from `trade_history` for table polling.
- System storage/logging endpoints:
  - `GET /api/v1/system/storage`:
    - leest `storage/stats.json`,
    - voegt host disk usage toe via `shutil.disk_usage`.
  - `GET /api/v1/system/logs`:
    - snelle tail-read van laatste regels uit `storage/logs/bot_execution.log`.
  - `WS /ws/logs`:
    - live streaming van nieuwe logregels voor portal console-tab.
  - `WS /ws/system-stats`:
    - elke ~5s een JSON-snapshot met `topic: "system_stats"` (`cpu_pct`, `ram_pct`, `gpu_util_pct`, `vram_used_mb`, `vram_total_mb`, `gpu_ok`);
    - implementatie in `app/services/system_stats.py` (`psutil` + `nvidia-smi` subprocess; geen GPU/driver ⇒ nullen / `gpu_ok: false`).
- AI Brain endpoints:
  - `GET /api/v1/brain/reasoning`
  - `GET /api/v1/brain/feature-importance`
  - `GET /api/v1/brain/training-monitor`
  - recovery floors active:
    - learning rate floor blijft `>= 1e-5` (geen 0.00e+0 freeze-state meer in UI/stats),
    - exploration floor blijft `>= 5%` (`RL_EXPLORATION_FINAL_EPS` runtime clamp op `0.05`).
  - strategy bars zijn nu altijd relatief genormaliseerd (som ~1.0) via `core/analytics.py::normalize_feature_weights` (softmax/minmax pad).
  - `training-monitor` now returns:
    - learning stats cards (`learning_rate`, `global_step_count`, `exploration_rate_pct`)
    - benchmark block (`rl_pnl_pct`, `buy_hold_pnl_pct`, `alpha_pct`)
    - SB3 health logs (`approx_kl`, `value_loss` series).
- Performance analytics endpoint:
  - `GET /api/v1/performance/analytics` returns:
    - equity curve,
    - recent closed trades,
    - win/loss ratio,
    - sentiment vs outcome stats,
    - weight-adjustment suggestions.
- Web portal includes prediction and paper PnL visualization.
- Portal now also includes:
  - full-screen bento terminal layout
  - readability-updated high-contrast text palette and stronger badge backgrounds
  - **Electric Quant** thema (`electric-quant.css`): achtergrond `#000`, tekst `#fff`, kaarten **transparant** met rand `rgba(255,255,255,0.15)`, hoeken **4px**; gauges **#00D1FF**; Pause/Panic subtiele neon-glow; tooltips (portal + inline waar zichtbaar) **glassmorphism** (`backdrop-filter: blur(5px)`).
  - **Uniforme 4-koloms grid** (Monitor · Signals · Visual · Action) op **Terminal** en **AI Brain Lab**: Monitor = CPU/RAM/GPU + ticker (+ Brain: LR/steps/explore); Signals = Terminal: sentiment + nieuws; Brain: reasoning + state + features + correlatie + nieuws; Visual = chart resp. training charts; Action = trades (verticale glow-lijst) + balance + risk-knoppen (Brain heeft eigen Pause/Resume/Panic ids).
  - WebSocket **`/ws/system-stats`** actief op **Terminal én Brain**; GPU via `nvidia-smi` (voorkeur GTX 1080 in naam).
  - animated ticker tape with colored sentiment headlines
  - Lightweight Charts price visualization with AI prediction markers
  - Performance Analytics dashboard:
    - equity curve line chart,
    - win/loss doughnut chart,
    - sentiment-vs-outcome bar chart.
  - live trade history table (tijd, pair, type, entry, sentiment, pnl).
  - interactive news modal with:
    - full title/publication date,
    - summary,
    - original source link,
    - FinBERT reasoning + decisive keywords,
    - affected ticker list.
  - chart-dropdown synchronization:
    - `updateChart(newPair)` clears series, fetches new pair history, redraws chart, and runs alongside websocket pair switch.
    - explicit loading/error text is shown in the chart area while pair data is fetched.
  - trade history table is now fed by `/api/v1/trades` polling every 5 seconds.
  - tab system:
    - `Terminal` tab (cockpit only)
    - `AI Brain Lab` tab (reasoning + feature weights + reward/loss monitor)
    - `System Logs` tab:
      - live console met monospaced font op zwarte achtergrond,
      - kleurcodes: info wit, success groen, warning oranje, error rood, `[RL-BRAIN]` blauw/paars,
      - auto-refresh toggle (5s) + interactieve controls:
        - `Pause Stream` (buffered while paused),
        - `Clear Console` (DOM reset),
      - websocket reconnect logica,
      - DOM cap van 500 regels voor frontend performance.
      - intelligente auto-scroll: alleen mee-scrollen als gebruiker al onderaan zat.
  - AI Brain data is lazily loaded/polled only when that tab is active.
  - AI Brain learning dashboard includes:
    - Cumulative Reward curve,
    - Episode Length curve,
    - Policy Entropy curve,
    - Value Loss curve,
    - RL vs Buy & Hold comparison table/chart.
  - Chart.js + Lightweight Charts in `app/static/js/terminal.js`: `highContrastChartOptions()` + JetBrains defaults; witte ticks/assen (12px), grid `#222`; tooltips hoog-contrast; `initHintPortals()` bindt `(i)` hints op Terminal/Brain/System naar **portal** `#brainHintPortal` (`z-index` ~2e9, blur). Brain-tab: cumulatieve reward alleen; extra train-grafieken alleen op Terminal.
  - `WS /ws/trades` voedt Brain-tab live trades wanneer actief.
  - emergency high-contrast theme overrides (`app/static/css/style.css`):
    - global text forced to white for readability,
    - brighter panel contrast (`#161616` + `#333` border),
    - vivid action button palette + hover glow,
    - impact badges with threshold colors:
      - positive > 0.2 => bright green,
      - negative < -0.2 => bright red,
      - neutral => gray.
  - Vault/Bitvavo status line now renders explicit:
    - `Connected` in green when balance check succeeds,
    - `Disconnected` in red on failures.
  - dynamic market dropdown for live pair selection
  - Bitvavo live price feed via WebSocket (`ticker24h`)
  - current sentiment score display (`/sentiment/current`)
  - auto-refresh polling for sentiment/status/activity
  - visible "last updated" timestamp in live monitor
  - vault balance-check status per selected market
  - bot status controls:
    - `POST /bot/pause`
    - `POST /bot/resume`
    - `POST /bot/panic`
  - status endpoint: `GET /bot/status`
  - market endpoints:
    - `GET /markets/active`
    - `POST /markets/select`
    - `GET /markets/selected`
    - `GET /vault/balance-check`
- Portal stability note:
  - fixed `TemplateResponse` signature compatibility for newer Starlette.
  - fixed technical analyzer input shaping (`close_prices` flattened to 1D) to prevent `/paper/run` 400 errors.

## Logging & Maintenance
- Runtime stdout/stderr wordt ge-teed naar `storage/logs/bot_execution.log`.
- `scripts/optimize_data.py` onderhoudt logs:
  - verwijdert regels ouder dan 1 uur (timestamp-based),
  - houdt logbestand onder 50MB (truncate/rotate gedrag).

## Next Logical Technical Step
- Add IP-whitelisting runbook in docs and wire active orders panel to real fill/status reconciliation.

## Trade History Database Schema
- Database path: `TRADE_HISTORY_DB_PATH` (default `data/trade_history.db`)
- Table: `trade_history`
  - `market`, `coin`
  - `entry_ts_utc`, `exit_ts_utc`
  - `entry_price`, `exit_price`, `qty`
  - `sentiment_score`
  - `headlines_json` (top 3 headlines captured at entry)
  - `fees_eur`, `pnl_eur`, `pnl_pct`, `outcome`
