# MEMORY - AI Trading Bot Technical State

Last updated: 2026-04-20

## Governance Mode
- Full Agency active via `.cursorrules`:
  - challenge risky/suboptimal requests,
  - propose safer alternatives,
  - proactively improve portal UX/performance,
  - future-proof decisions for RL phase,
  - flag logic gaps and edge cases immediately.

## Portal layout & GPU monitoring
- **Genesis Noir Quant:** tabs **Terminal**, **AI Brain**, **Ledger** (trade history + equity/PnL charts), **Hardware** (vier ring-meters + logs). CSS-grid **`repeat(4, minmax(0,1fr))`**; **Inter** als UI-font, **JetBrains Mono** voor data/tabulair; labels min. **14px**; minimalistische **ghost/outline** header-knoppen; hint-bubbles via **`#brainHintPortal`** (**z-index 99999**). Terminal toont **3** top-nieuwsregels.
- **AI Brain Lab:** kolom 1 = Monitor + RL stats; kolom 2 = reasoning + state (o.a. MACD); kolom 3 = reward/feature weights/benchmark; kolom 4 = balance/risk/paper. Hints via **`initHintPortals()`** op alle tab-roots → **`#brainHintPortal`** (fixed, z-index ~2e9).
- **Card-headers** met achtergrond `#111` (Electric Quant) voor sectiescheiding; hoofdprijschart achtergrond **#000000**, tekst **#FFFFFF**; neon-lijn **lineWidth 4**.
- **Dockerfile:** **`nvidia/cuda:12.4.1-runtime-ubuntu22.04`**, **`ENTRYPOINT []`** (NVIDIA-default entrypoint verstopte torch &lt;2.4 en brak `transformers`). Daarna **`torch==2.6.0+cu124`** (`--index-url https://download.pytorch.org/whl/cu124`) — nodig voor **transformers 5.x** (CVE/torch.load-check) en **stable-baselines3**. **docker-compose:** `gpus: all`, **`GENESIS_REQUIRE_GPU`**, **`FINBERT_USE_CUDA`**. Logs: **`[DEVICE] Using device: cuda:0 (...)`**, **`[DASHBOARD] Dashboard live op poort 8000`**, **`[ENGINE] Tick | torch_device=cuda:0`**. GPU-%: `nvidia-smi` in `system_stats.py`.
- Ubuntu image levert **Python 3.10**; `datetime.UTC` is pas 3.11+. Code gebruikt `app/datetime_util.UTC` (`timezone.utc`) + scripts met try/except zodat de API niet meer crasht op import.
- `system_stats._nvidia_smi_stats`: lossere parsing van %/komma’s; extra `utilization.gpu`-query per GPU-index.

## Risk engine (position sizing & SL/TP)
- `core/risk_manager.py`: `RiskManager` leest o.a. `RISK_SIZING_MODE` (`fixed_eur` | `equity_pct`), `RISK_BASE_TRADE_EUR`, `RISK_MAX_TRADE_EQUITY_PCT`, `RISK_MAX_POSITION_EQUITY_PCT`, `RISK_STOP_LOSS_PCT`, `RISK_TAKE_PROFIT_PCT`.
- `calculate_trade_size()` → fractie/notional t.o.v. equity/cash; `check_safety()` blokkeert BUY bij te hoge exposure of te weinig cash; `hard_exit_for_sl_tp()` dwingt SELL buiten RL op basis van gewogen gemiddelde entry uit `open_lots`.
- Paper cycle (`/paper/run`): RL levert richting; `CORE_RISK` zet omvang; spread/sentiment blijft via `app.services.risk.RiskManager`; `/activity` bevat `risk_profile` voor de UI (kolom 4 „Risk profile”).
- `compute_risk_controls()` delegeert naar `core.risk_manager.risk_controls_for_close` (zelfde SL/TP-% als harde exits).

## Current Runtime Modes
- `LIVE_MODE` from vault, default fallback is paper mode when missing.
- `DRY_RUN` supported; when true, order calls are logged/simulated.
- **Operationeel:** container start/stop voor dagelijks gebruik via **`./run_bot.sh`**; script sourcet vault, health-checks, `docker compose -f <repo>/docker-compose.yml`. Flags: `--skip-optimize`, `--no-cache`, `--clean`, **`--heal`** (bounded rebuild/restart + log-pattern grep op o.a. CUDA OOM / driver mismatch / API errors), `--follow` voor live compose logs.
- Standaard **geen** `docker compose down --volumes --rmi local` meer bij elke start (te traag); `--clean` voor oude gedrag. Eén gedeelde **FinBERT**-instantie voor `SignalEngine` en `NewsMappingService`; `FINBERT_USE_CUDA=0` in compose om PyTorch/CUDA-driver-warnings te vermijden tenzij je driver expliciet matcht.
- **Tijdzone:** `ai-trading-bot` mount `/etc/localtime` en `/etc/timezone` read-only zodat container-CET/CEST gelijk loopt met de Linux-host; `TZ=Europe/Amsterdam` blijft in compose. Telegram start/stop gebruikt `pytz.timezone('Europe/Amsterdam')` voor tijdstempels.
- `run_bot.sh` supports:
  - **default:** background/detached (`docker compose up -d`); portal op `http://localhost:8000`
  - foreground: `-f` / `--foreground` / `--interactive` (logs in terminal)
  - expliciet achtergrond: `-b` / `--background` (zelfde als default)
  - `docker-compose.yml`: **`restart: always`** op `ai-trading-bot` — bij procescrash of na reboot van Docker start de container opnieuw (tenzij expliciet verwijderd).
  - Docker Compose only runtime (no local venv dependency)
  - environment loading from project `.env` when present (`--env-file`)

## Secret Management
- Secrets sourced from: `$HOME/.trading_vault`
- Expected key groups:
  - Read-only: `BITVAVO_KEY_READ`, `BITVAVO_SECRET_READ`
  - Trading: `BITVAVO_KEY_TRADE`, `BITVAVO_SECRET_TRADE`
  - Telegram lifecycle alerts (optioneel): `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` — moeten in de shell staan vóór `docker compose` of via `docker-compose.yml` environment; `run_bot.sh` sourcet de vault zodat deze variabelen de container ingaan.
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

## Whale pressure (CryptoCompare)
- `app/services/whale_watcher.py` gebruikt **geen** Whale Alert API (deprecated).
- Whale pressure (0..1) komt uit **CryptoCompare `/data/v2/news/`** headlines: regex op o.a. `whale`, `large transfer`, `billion`, `million btc/eth`.
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

## Monitoring/API
- FastAPI endpoints for health, predict, activity, paper run, and daily dry-run PnL.
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
