# AI Trading Bot Roadmap

Single Source of Truth voor planning, voortgang en architecturale keuzes.

## Voortgang

**Overall voortgang: 83% (124/150 taken)**

`[████████████████████░░] 83%`

---

## Fase 1 - Data Infra

Doel: betrouwbare, genormaliseerde datafundering voor prijs, nieuws en markt-sentiment.

- [x] Maak CryptoCompare + RSS nieuws-koppeling
- [x] Maak marktdata koppeling (historische prijsdata)
- [x] Voeg Bitvavo candles ingestie toe
- [x] Voeg Fear & Greed API ingestie toe
- [x] Bouw `DataAggregator` voor multi-source normalisatie naar 1 DataFrame
- [x] Voeg volatility features toe (`returns`, rolling std)
- [x] Voeg nieuws-intensiteit features toe (`news_count`, `news_peak_zscore`)
- [x] Voeg event impact flags toe (incl. Bitcoin Halving 2024)
- [x] Bouw `BitvavoRateLimitManager` die rate-limit headers leest en auto-throttle toepast
- [x] Gebruik uitsluitend Parquet-opslag voor historische koers-, nieuws- en feature-data (`$HOME/AI_Trading/storage`)
- [x] Voeg pruning/downsampling script toe (ouder dan 30 dagen naar compactere resolutie)
- [ ] Voeg duplicate/rumor filtering toe op nieuwsbron
- [ ] Voeg data quality scores toe (staleness, missingness, source reliability)
- [x] Voeg persistent feature store toe (Parquet partitioned per dag/symbol)
- [ ] Voeg data drift monitor toe (feature-distributies over tijd)

## Fase 1.1 - Bitvavo Safety & Compliance

Doel: API-veiligheid, ban-preventie en operationele robuustheid rond exchange-verkeer.

- [x] Implementatie van de Rate Limit Controller (1000 punten/minuut regel).
- [x] Logging van API-verbruik per uur.
- [ ] IP-Whitelisting instructie (veiligheid voor je API keys).
- [x] 'Circuit Breaker' bij herhaalde API-fouten om een ban te voorkomen.

## Fase 0 - Governance & Memory

Doel: projectbesturing, kennisborging en consistente documentatie-cyclus.

- [x] Maak `.cursorrules` met Bitvavo rate-limit + security programmeerregels.
- [x] Maak `MEMORY.md` met actuele technische staat van API's en modellen.
- [x] Leg dynamische updateflow vast voor `ROADMAP.md` en `MEMORY.md`.
- [x] Voeg executie-startscript toe met Docker-only interactive/background mode.
- [x] Activeer "Full Agency" proactieve architectuurregels in `.cursorrules`.

## Fase 2 - AI Brain

Doel: signalering op basis van technische en tekstuele signalen met duidelijke model-contracten.

- [x] Definieer BaseClasses voor `TechnicalAnalyzer`, `SentimentAnalyzer`, `Judge`
- [x] Implementeer Scikit-learn technical analyzer
- [x] Implementeer FinBERT sentiment analyzer (HuggingFace Transformers)
- [x] Implementeer weighted `Judge` voor signaalfusie
- [x] Bouw `SignalEngine` orchestration laag
- [x] Expose model scores in activity output (`last_scores`)
- [x] Bouw RL omgeving met reward/penalty constructie
- [x] Voeg Stable-Baselines3 PPO train-loop toe
- [x] Simuleer max 10.000 trades in RL env
- [ ] Maak judge-gewichten configureerbaar via vault vars
- [x] Voeg model registry/versionering toe (model + feature schema hash)
- [x] Behoud enkel top-5 RL modelversies op reward-score (auto cleanup)
- [ ] Voeg walk-forward evaluatie pipeline toe (rolling train/test)
- [ ] Voeg confidence gating toe (trade alleen bij minimale confidence)

## Fase 2.1 - Dynamic Market Selection

Doel: automatisch liquiditeitsgestuurde marketselectie en live pair switching in portal.

- [x] Bouw startup market scanner die actieve Bitvavo pairs ophaalt.
- [x] Filter pairs op 24u volume drempel (default > EUR 500.000).
- [x] Voeg portal dropdown toe voor live pair selectie en WebSocket follow.
- [x] Voeg vault-gedreven balanscheck per geselecteerd pair toe.

## Fase 3 - Bitvavo Integratie

Doel: veilige exchange-koppeling met duidelijke scheiding tussen read, paper en live execution.

- [x] Bouw Bitvavo exchange client met signed requests
- [x] Valideer READ key connectie
- [x] Valideer TRADE key connectie
- [x] Richt vault-gedreven secret management in (`$HOME/.trading_vault`)
- [x] Voeg `LIVE_MODE` configuratie toe (default fallback naar paper)
- [x] Voeg fail-fast checks toe voor ontbrekende live credentials
- [x] Voeg dry-run decorator toe die live calls blokkeert bij `DRY_RUN=true`
- [x] Log alle trade-attempts naar `trades_log.csv`
- [x] Bouw aparte live execution adapter (idempotent order placement + retry/backoff)
- [x] Toon Bitvavo live prijsfeed in portal via WebSocket (`ticker24h`)
- [ ] Voeg order status reconciliatie toe (fills/partials/cancels)
- [ ] Voeg websocket prijsfeed integratie toe i.p.v. polling-only
- [ ] Voeg circuit breaker toe bij API timeout/rate-limit spikes

## Fase 3.1 - Advanced UI & News Mapping

Doel: professionele terminal-ervaring met high-impact nieuwsrouting naar actieve markten.

- [x] Bouw `NewsMappingService` met alias mapping (o.a. Bitcoin/BTC/Halving -> BTC).
- [x] Voeg impactfilter toe (|sentiment| > 0.4 of top-20 volume coin mention).
- [x] Expose endpoint `GET /api/v1/news/ticker` met `{text, coin, sentiment}` records.
- [x] Bouw ticker tape + bento terminal layout + Lightweight Charts met AI markers.

## Fase 4 - Risk Management

Doel: drawdown beperken, executionrisico beheersen en kapitaal beschermen.

- [x] Implementeer `RiskManager` module
- [x] Maximaal 3% budget per trade afdwingen (legacy; paper sizing via env `RISK_MAX_TRADE_EQUITY_PCT`, default 10%)
- [x] `core/risk_manager.py`: vaste EUR vs % equity sizing, max-position guard, harde SL/TP naast RL
- [x] Terminal kolom 4: Risk profile panel (neon oranje) + `/activity.risk_profile`
- [x] Volatility filter op spread-proxy afdwingen
- [x] Emergency exit op sentiment shock (`<= -0.8`)
- [x] Voeg stop-loss/take-profit controls toe
- [x] Voeg daily fictieve PnL inclusief 0.15% fees toe
- [ ] Gebruik echte bid/ask spread uit orderbook (vervang candle-range proxy)
- [ ] Voeg daily max drawdown kill switch toe
- [ ] Voeg max consecutive losses guard toe
- [ ] Voeg max exposure per asset toe
- [ ] Voeg portfolio-level VaR limiet toe
- [ ] Maak risk parameters per regime dynamisch

## Fase 4.1 - Paper Trading & Learning

Doel: paper trading niet alleen simuleren, maar outcome-gedreven leren op trade- en sentimentniveau.

- [x] Bouw `PaperTradeManager` met virtuele wallet en 0.15% fee-model.
- [x] Gebruik live Bitvavo prijs als execution reference voor crypto pairs.
- [x] Sla trade history op in SQLite (`trade_history`) met entry/exit, sentiment en top-3 headlines.
- [x] Voeg analytics toe: sentiment bij top-10 verliezen versus top-10 winsten.
- [x] Voeg `adjust_weights()` suggestie-engine toe voor coin-specifieke judge tuning.
- [x] Voeg Performance Analytics panel toe (equity curve, win/loss ratio, sentiment vs outcome).

## Fase 5 - Portal, Monitoring & Ops

Doel: transparante operatie, snelle troubleshooting en veilige uitrol.

- [x] Bouw webportal met prediction en activity overzicht
- [x] Voeg paper trading PnL visualisatie toe
- [x] Voeg daily dry-run PnL endpoint toe
- [x] Documenteer architectuur en policies in README
- [x] Toon actuele sentiment score in portal
- [x] Voeg botstatus controls toe (Running/Paused/Panic Button)
- [x] Telegram-melding bij API/engine start en bij shutdown (`TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` in omgeving / vault)
- [x] Tijdzonesync: `docker-compose` mount `/etc/localtime` + `/etc/timezone` (ro); Telegram-tijdstempels via `pytz` `Europe/Amsterdam`
- [x] Genesis Modern Noir UI: strict `1.2fr 1fr 1fr 1fr` grid alle tabs, JetBrains Mono + witte typografie, unified header (ticker/markt/balans + outline-knoppen), portal-hints overal, Brain Lab alleen cumulative reward + MACD in state, System Logs 4-koloms console-layout
- [x] Uniform Genesis high-contrast thema (zwart/wit, rand `#222`, neon accenten) over alle tabs + tab-nav witte actieve rand.
- [x] Terminal herbouwd naar 4 kolommen (system / news / chart / execution+trades); Fear&Greed via `/activity`; performance-UI verborgen maar polling-compatibel.
- [x] GPU stats: `nvidia-smi` CSV-parse + voorkeurs-GTX 1080; WebSocket payload met `gpu_name` / `gpu_index`.
- [x] Electric Quant theme engine (`electric-quant.css`): transparante kaarten, witte rand 15% opacity, 4px radius; Terminal + Brain dezelfde 4-koloms structuur; glass tooltips; trade-lijsten met neon buy/sell dots.
- [x] Symmetrisch dashboard: CSS grid `repeat(4,1fr)` + `grid-auto-rows: 1fr`, kaarten vullen kolomhoogte; Terminal kolommen Monitor / Signals (reasoning·state·news) / Learning (training·features) / Action (trades|balance); Brain-tab in dezelfde indeling.
- [x] GPU in container: Dockerfile op `nvidia/cuda` runtime + `docker-compose` `gpus: all`; `nvidia-smi` parsing robuuster (util-fallback per GPU-index).
- [x] Genesis **4-tab** portal (Noir Quant): **Terminal** / **AI Brain** / **Ledger** / **Hardware**; ledger-tab tot 200 trades + performance-charts; Terminal **3** top-nieuws; uniform **4×1fr** grid, **Inter** + mono data, ghost-knoppen, hint-portal **z-index 99999**.
- [x] Dockerfile: **CUDA 12.4.1-runtime** + **`torch==2.2.0+cu121`**; compose **`GENESIS_REQUIRE_GPU`** + **`FINBERT_USE_CUDA`**; startup **`[DEVICE] Using device: cuda:0 (...)`**; `run_bot.sh --heal` wacht op device-log + scant Traceback/CUDA/driver-fouten.
- [ ] Toon trades log live in portal
- [ ] Voeg model metrics dashboard toe (hit-rate, pnl/trade, max DD)
- [ ] Voeg alerting toe (Telegram/Slack) bij risk breaches
- [ ] Voeg health checks per component toe (ingestion/model/exchange/risk)
- [ ] Voeg docker healthcheck + restart policy toe
- [ ] Voeg CI checks toe (lint/tests/type checks)

## Fase 5.1 - UI Readability & Trade Transparency

Doel: hogere leesbaarheid, expliciete coin-mapping en live trade-overzicht in terminal.

- [x] Verhoog UI-contrast voor hoofdtekst en koppen in terminal thema.
- [x] Voeg duidelijke impact/ticker badges toe met positieve/negatieve achtergrondkleuren.
- [x] Breid news keyword-matrix uit en vervang generieke `MKT` fallback met concrete coin-tag.
- [x] Voeg live `Trade History` tabel toe (tijd, pair, type, entry, sentiment, pnl).

## Fase 5.2 - Interactive Terminal Finalization

Doel: terminal interactiever maken met klikbare nieuwsdetails en robuuste paper/live exchange fallback.

- [x] Maak Live News Feed items klikbaar met detail modal (titel, datum, summary, bronlink).
- [x] Toon AI-analyse in modal incl. FinBERT reden en beslissende keywords.
- [x] Verhoog contrast naar minimaal `#cccccc` feed tekst en `#ffffff` titels.
- [x] Voeg coin-specifieke badge kleuren toe (o.a. BTC oranje, SOL diepblauw).
- [x] Voeg paper-mode balance/order storage toe in `bitvavo_manager.py` en koppel `BitvavoClient` hieraan.
- [x] Voeg `WS /ws/system-stats` toe: elke 5s JSON met `topic: system_stats` (CPU/RAM via `psutil`, GPU/VRAM via `nvidia-smi` wanneer beschikbaar).
- [x] Terminal header: CPU/RAM/GPU resource-balkjes naast ticker; strakkere `#333` panelranden, witte meta-tekst, neon equity/cash, dieprode panic-glow, witte randen op inputs/knoppen.
- [x] Nieuwsfeed: felgroen/rood impact + sentiment-tags; dikke witte headlines.

## Fase 5.3 - Chart Sync & News Drilldown

Doel: chart synchronisatie op market-switch en diepere nieuwsinteractie in terminal.

- [x] Voeg `/api/v1/history?pair=...` endpoint toe voor pair-gedreven chart data.
- [x] Implementeer `updateChart(newPair)` flow bij dropdown change incl. chart reset + websocket pair switch.

## Fase 5.4 - Trade Visibility Hardening

Doel: gegarandeerde zichtbaarheid van trades en universele chart-load over Bitvavo pairs.

- [x] Voeg endpoint `GET /api/v1/trades` toe (trade_history records uit SQLite).
- [x] Voeg frontend polling toe (5s) voor live trade table refresh.
- [x] Breid `/api/v1/history` uit met Bitvavo-candles fallback voor `XXX-EUR` pairs.

## Fase 5.5 - Autonomous RL & AI Brain Lab

Doel: RL-agent autonoom laten redeneren en deze "brain state" live visualiseren in een aparte portal-tab.

- [x] Voeg `agent_rl.py` service toe met PPO pre-train, decision API en feature-importance/redenering.
- [x] Breid RL state features uit met `sentiment_score`, `rsi_14`, `ema_gap_pct`.
- [x] Voeg `orderbook_imbalance` + `macd` toe als RL state features en sla op in Parquet feature store.
- [x] Dwing pre-train af op 400 dagen (vault lookback) vóór eerste paper cycle per pair.
- [x] Koppel RL action-output direct aan `PaperTradeManager` execution flow.
- [x] Voeg `AI Brain Lab` tab toe met reasoning box + feature chart + reward/loss monitor.
- [x] Sla AI thought op in `trade_history` (`ai_thought`) voor uitlegbare trades.

## Fase 5.6 - RL Progress Monitoring

Doel: aantoonbaar maken dat de PPO-agent leert en beter presteert dan baseline.

- [x] Voeg leercurves toe in AI Brain: cumulative reward, episode length, policy entropy.
- [x] Voeg training-stat cards toe: learning rate, global step count, exploration rate.
- [x] Voeg RL vs Buy & Hold benchmark visualisatie + tabel toe.
- [x] Expose SB3 network logs (`approx_kl`, `value_loss`) via brain monitor endpoint.
- [x] Houd AI Brain rendering lazy (alleen laden wanneer AI Brain tab actief is).
- [x] Hard reset chart-as leesbaarheid: Chart.js ticks/grid/tooltip/legenda wit; AI Brain tab globale tekst #fff (uitz. groen/rood).
- [x] Migreer whale-sensing van Whale Alert naar CryptoCompare nieuws-headlines; `WHALE_ALERT_API_KEY` verwijderd.

## Fase 6 - System Maintenance & Health Monitoring

Doel: data-huishouding, opslag-efficiency en operationele zichtbaarheid structureel borgen.

- [x] Voeg `scripts/optimize_data.py` toe voor onderhoud (downsample + pruning + model cleanup + csv->parquet).
- [x] Log storage metrics voor/na optimalisatie naar `storage/stats.json`.
- [x] Expose storage + host disk usage via `GET /api/v1/system/storage`.
- [x] Voeg Storage Health card toe in portal met progress bar en besparingsstatistiek.
- [x] Voeg `System Logs` tab toe met high-contrast consoleweergave.
- [x] Expose `GET /api/v1/system/logs` met snelle tail van laatste regels.
- [x] Voeg WebSocket logstream endpoint `GET /ws/logs` toe voor live console.
- [x] Voeg reconnect + max 500 DOM logregels toe voor frontend performance.
- [x] Breid optimize script uit met log pruning (>1h) en bestandslimiet (50MB).
- [x] Voeg console-toolbar controls toe: Pause Stream + Clear Console.
- [x] Voeg pause-buffering toe zodat stream doorloopt zonder DOM updates tijdens pauze.
- [x] Breid logkleurcodering uit met `[RL-BRAIN]` accentkleur naast info/success/warning/error.
- [x] Verbeter intelligente auto-scroll (alleen volgen als gebruiker onderaan is).
- [ ] Voeg gecentraliseerde structured logging toe (JSON schema + trace IDs per request).

## Fase 6.1 - Validatie & Go-Live

Doel: alleen gecontroleerd live gaan met aantoonbare robuustheid.

- [ ] Maak backtest protocol met kosten/slippage model
- [ ] Definieer live go/no-go criteria
- [ ] Draai minimaal 2 weken paper-forward test
- [ ] Vergelijk paper vs verwachte slippage/fees
- [ ] Voer staged rollout uit (tiny size -> partial size -> target size)
- [ ] Definieer incident runbook (API down, latency spike, flash move)
- [ ] Plan periodieke model hertraining + recalibratie

---

## Architecturale Beslissingen

### Waarom Scikit-learn voor Technical Analysis

- Snel en stabiel voor baseline-modellen met lage operationele complexiteit.
- Goed uitlegbaar (lineaire trend, feature-bijdrage) voor snelle iteratie.
- CPU-efficiënt, dus ideaal naast zwaardere NLP-componenten.

### Waarom FinBERT (Transformers) voor Sentiment

- Domeinspecifiek voor finance/marktteksten (beter dan generieke sentimentmodellen).
- Werkt goed op headlines en korte nieuwsteksten.
- Combineerbaar met confidence-score voor risk gating.

### Waarom een Judge-laag

- Voorkomt dat technische of nieuwsmodellen in isolatie beslissen.
- Maakt signaalfusie expliciet en controleerbaar via gewichten/thresholds.
- Eenvoudig aanpasbaar per regime zonder complete herbouw van de pipeline.

### Waarom Reinforcement Learning (PPO)

- Geschikt voor sequentiële besluitvorming met transactiekosten en drawdown-effect.
- Reward/penalty structuur sluit direct aan op tradingdoelen (EUR PnL vs risico).
- Praktisch te trainen op historische trajecten met event-annotaties.

### Waarom Vault-first secret management

- Geheimen blijven buiten git/projectfolder.
- Duidelijke scheiding tussen code en credentials.
- Verkleint kans op token leaks en vereenvoudigt portable deployment.

---

## Updateprotocol (verplicht)

Bij elke afgeronde taak:

1. Zet taak op `[x]` in dit bestand.
2. Herbereken voortgang bovenaan (`voltooid/totaal` + progressbar).
3. Evalueer of de eerstvolgende taak nog logisch is op basis van nieuwe bevindingen.
4. Werk taakomschrijving bij als scope is veranderd.
5. Synchroniseer waar nodig met `README.md` zodat ontwerp en uitvoering consistent blijven.

**Beslisregel:** als een taak door nieuwe inzichten risicoverhogend of inefficiënt is geworden, eerst roadmap herprioriteren voordat implementatie doorgaat.

## Laatste Update

- Toegevoegd: governance-bestanden `.cursorrules` en `MEMORY.md`.
- Bevestigd: rate-limit en security regels uit Bitvavo docs vastgelegd als ontwikkelstandaard.
- Toegevoegd: executiefase met nieuw `run_bot.sh` (Docker-only + .env + background/interactief) en portal-controls.
- Stabilisatie: portal runtime issues opgelost (`TemplateResponse` compat + `/paper/run` shape-fix in technical analyzer).
- Toegevoegd: Fase 2.1 dynamic market selection (scanner + volumefilter + dropdown + vault balance-check).
- Verbeterd: portal live polling voor sentiment/status/activity zodat updates zonder handmatige actie zichtbaar zijn.
- Verbeterd: live monitor toont nu expliciet "laatste update" timestamp bij polling-data.
- Governance: "Full Agency" gedrag vastgelegd in `.cursorrules` (challenge + UX/perf + future-proofing + edge-case alerts).
- Uitgevoerd: hourly API-usage logging + circuit breaker in Bitvavo manager, incl. status endpoint.
- Uitgevoerd: professional terminal upgrade met ticker tape, bento grid, news-impact mapping en `/api/v1/news/ticker`.
- Uitgevoerd: Fase 4.1 Paper Trading & Learning met database trade history en sentiment correlation analytics.
- Hotfix: abstract class fout opgelost door concrete `BitvavoClient` methoden correct in class scope te plaatsen (`get_balance`, `place_market_order`).
- Verbeterd: news mapping en terminal readability met badge-contrast + trade history tabel.
- Uitgevoerd: interactieve news modal + coin-specifieke ticker badges + paper-mode Bitvavo manager fallback.
- Hotfix: chart-dropdown sync nu pair-gedreven via `/api/v1/history` en `updateChart(newPair)`.
- Uitgevoerd: trade visibility hardening met `/api/v1/trades` en 5s polling + universele history fallback.
- Uitgevoerd: RL autonomy upgrade met AI Brain tab, pre-train gating en explainable decision flow.
- Uitgevoerd: RL progress monitoring dashboard met leercurves, benchmark en SB3-health logs.
- Uitgevoerd: Parquet-only storagelaag onder `$HOME/AI_Trading/storage` voor candles, nieuws en RL-features.
- Uitgevoerd: automatische storage-pruning script voor data ouder dan 30 dagen (downsample + noise filtering).
- Uitgevoerd: RL modelversionering met timestamp + top-5 reward retentie.
- Uitgevoerd: data-optimalisatie + storage monitoring milestone met `storage/stats.json` en system storage endpoint.
- Uitgevoerd: System Logs milestone met live WebSocket console, tail endpoint en log pruning/rotation.
- Uitgevoerd: Interactive Live Console controls met pause/resume buffer en clear-actie.
- Volgende logische stap: IP-whitelisting runbook toevoegen + order status reconciliatie (fills/partials) + risk guards (daily max drawdown / consecutive losses).
- Whale Alert vervangen door CryptoCompare headline-scan voor `whale_pressure`; System Logs tonen `[WHALE-SYNC]` in hoog contrast.
