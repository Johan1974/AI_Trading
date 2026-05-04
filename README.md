# AI Trading Bot (OHLCV + News Sentiment)

Deze repository bevat een modulaire AI trading bot met een FastAPI webportal.  
De bot combineert prijsdata (OHLCV) met nieuws-sentiment en past basis risk controls toe.

## Projectdoel

- Historische marktdata combineren met nieuwscontext voor voorspellende signalen.
- Modulaire architectuur hanteren: `Ingestion | Analysis | Execution`.
- Risk-first implementatie met stop-loss, take-profit en position sizing.
- Systeem portable houden: geheimen uitsluitend via externe vault (`$HOME/.trading_vault`).

## Huidige Architectuur

### 1) Data Ingestion
- `app/services/ingestion.py`
  - Haalt OHLCV op via `yfinance`.
  - Haalt nieuws op via CryptoCompare API met RSS-backup.
- `app/services/data_aggregator.py`
  - Combineert Bitvavo candles + Fear & Greed + nieuwscontext tot één genormaliseerd DataFrame.

### 2) Analysis / Feature Layer
- `app/services/features.py`
  - Bouwt trend-features voor model.
  - Berekent simpele sentiment score uit headlines/beschrijvingen.

### 3) Signal Model
- `app/services/model.py`
  - Baseline lineaire regressie op recente slotkoersen.
- `app/ai/base.py`
  - BaseClasses: `TechnicalAnalyzer`, `SentimentAnalyzer`, `Judge`.
- `app/ai/technical/sklearn_technical.py`
  - Scikit-learn technical scoring model.
- `app/ai/sentiment/finbert_sentiment.py`
  - HuggingFace FinBERT sentiment scoring.
- `app/ai/judge/weighted_judge.py`
  - Combineert technical + sentiment via gewogen judge-score.
- `app/services/signal_engine.py`
  - Orkestreert beide modellen en levert unified judge-uitkomst.

### 4) Risk Engine
- `app/services/risk.py`
  - Signaal mapping (`BUY/SELL/HOLD`).
  - Risk controls: stop-loss, take-profit, position size fraction.
  - `RiskManager` gates:
    - max `3%` budget per trade,
    - volatility filter (skip bij te hoge spread in bps),
    - emergency exit bij sentiment-shock (`<= -0.8`).

### 5) Execution (Paper)
- `app/services/execution.py`
  - Simuleert orderuitvoering inclusief fee-inschatting.
- `app/services/paper.py`
  - Houdt paper portfolio (cash/positie/equity/PnL history) bij per cycle.
- `app/exchanges/base.py`
  - Abstracte exchange interface voor execution.
- `app/exchanges/bitvavo.py`
  - Bitvavo signed client voor balance en market orders.

### 6) State & Monitoring
- `app/services/state.py`
  - Houdt activity-log en laatste prediction/order in memory.
- `app/main.py`
  - API endpoints + portal integratie.

## API Endpoints

- `GET /health` - service health.
- `GET /predict?ticker=AAPL` - genereert prediction + risk + paper order.
- `POST /paper/run?ticker=AAPL` - draait volledige paper-trading cycle en update portfolio/PnL.
- `GET /dry-run/pnl/daily?date_utc=YYYY-MM-DD` - berekent fictieve dag-PnL uit `trades_log.csv`.
- `GET /activity` - actuele runtime status en events.
- `GET /` - webportal.

## Dry Run Decorator

Trades worden via een decorator gelogd in `trades_log.csv` zonder echte Bitvavo API call wanneer:

- `DRY_RUN=true` (default)

Gedrag:
- alle trade-attempts loggen met timestamp, market, side, prijs, amount en status;
- fake fee-rate: `0.15%` (`0.0015`);
- live call wordt gesimuleerd in dry-run mode.

Dagelijkse fictieve PnL:
- endpoint: `GET /dry-run/pnl/daily`
- rekent:
  - realized PnL,
  - unrealized PnL (open positie op laatste dagprijs),
  - net PnL,
  - totale betaalde fees.

## Data Aggregator (multi-source features)

De `DataAggregator` normaliseert drie bronnen op één tijdas (`timestamp`) zodat het model patronen tussen nieuws-pieken en prijsvolatiliteit kan leren:

1. Bitvavo candles (`open/high/low/close/volume`)
2. Crypto Fear & Greed (`fng_value`, `fng_classification`)
3. CryptoCompare/RSS nieuwsstream (`news_count`, `news_sentiment_raw`)

Extra afgeleide features:
- `returns`
- `volatility`
- `news_peak_zscore`
- `price_volatility_interaction`

Voorbeeldgebruik:

```python
import os
from app.services.data_aggregator import AggregatorConfig, DataAggregator

config = AggregatorConfig(
    market="BTC-EUR",
    interval="1h",
    candle_limit=240,
    news_query="crypto",
    news_api_key=os.getenv("CRYPTOCOMPARE_KEY"),
)
df = DataAggregator(config).build_normalized_frame()
print(df.tail(3))
```

## Reinforcement Learning (Stable Baselines3)

Voor RL-simulatie op Bitvavo-data is de volgende module toegevoegd:

- `app/rl/data.py`
  - Historische candles 2024-2025 ophalen vanaf Bitvavo.
  - Feature engineering inclusief event-flags voor grote nieuws-events (o.a. Bitcoin Halving).
- `app/rl/events.py`
  - Eventkalender met impact scores.
- `app/rl/env.py`
  - `BitvavoTradingEnv` met:
    - **Reward**: via `core/reward_function.py` (PnL-%, drawdown, friction, consistency, SL-shock; env sync met risk SL%).
    - max aantal gesimuleerde trades.
- `app/rl/train.py`
  - PPO training + evaluatie helpers.
- `train_rl_bot.py`
  - CLI startscript voor 10.000 tradesimulaties.

Start RL training:

```bash
cd "$HOME/AI_Trading"
python3 train_rl_bot.py
```

Default settings:
- Periode: 2024-01-01 t/m 2025-12-31
- Market: `BTC-EUR`
- Interval: `1h`
- Simulatie cap: `10.000` trades
- Algorithm: `PPO` (Stable Baselines3)

## Beslisflow (Judge voor execution)

1. `SklearnTechnicalAnalyzer` berekent technische score en verwachte return.
2. `FinBertSentimentAnalyzer` berekent sentimentscore op nieuws-headlines.
3. `WeightedJudge` combineert beide signalen naar één composite score.
4. Alleen dit judge-signaal gaat door naar risk/execution logic (Bitvavo-pad).

Formule:

`composite = technical_score * 0.65 + sentiment_score * 0.35`

Default thresholds:
- `composite > 0.20` -> `BUY`
- `composite < -0.20` -> `SELL`
- anders -> `HOLD`

## Timezone Policy (Verplicht)

Het systeem gebruikt **altijd en uitsluitend** de tijdzone **Europe/Amsterdam** (Amsterdam tijd). Geen enkele andere tijdzone is toegestaan in de configuratie, code of containers. Amsterdam is te allen tijde de default voor logging, cronjobs, UI-weergave, notificaties en Docker-synchronisatie.

## RiskManager beleid

Actieve riskregels vóór execution:

1. **Budget cap per trade**  
   `adjusted_size_fraction <= 0.03`

2. **Volatility filter op spread proxy**  
   Als `spread_bps > 45` -> signaal wordt `HOLD` (geen trade).

3. **Emergency Exit**  
   Als `news_sentiment <= -0.8` -> geforceerd `SELL` signaal.

## Security & Secrets Policy

- Geen `.env` in de projectfolder.
- Alle geheimen staan in: `$HOME/.trading_vault`.
- Scripts onder `scripts/` sourcen `scripts/lib_compose_root.sh` en laden `~/.trading_vault`; geen `.env` in de repo.

Voorbeeld vault:

```bash
export CRYPTOCOMPARE_KEY=jouw_key_hier
export DEFAULT_TICKER=AAPL
export LOOKBACK_DAYS=400
export LIVE_MODE=false
export BITVAVO_KEY_READ=...
export BITVAVO_SECRET_READ=...
export BITVAVO_KEY_TRADE=...
export BITVAVO_SECRET_TRADE=...
```

Nieuwsbronnen notitie:
- Primair: CryptoCompare via `CRYPTOCOMPARE_KEY`.
- Backup: keyless RSS feeds (Cointelegraph, CoinDesk, Decrypt).

Mode-regel:
- `LIVE_MODE` ontbreekt in de vault -> app draait standaard in paper mode.
- `LIVE_MODE=true` -> `BITVAVO_KEY_TRADE` en `BITVAVO_SECRET_TRADE` zijn verplicht (fail-fast bij startup).

### IP-Whitelisting (Cruciaal voor Live Trading)
Voor maximale veiligheid van je funds is het sterk aanbevolen om IP-whitelisting toe te passen op je Bitvavo API keys (zowel `READ` als `TRADE`).

1. Achterhaal het publieke IP-adres van je server (bijv. via `curl ifconfig.me` in je terminal).
2. Ga naar je **Bitvavo Dashboard** -> **API Instellingen**.
3. Bewerk je actieve API keys en vul bij **IP Whitelist** het IP-adres van je server in.
4. Resultaat: mocht de inhoud van je `~/.trading_vault` ooit onbedoeld uitlekken, dan kan een aanvaller absoluut niets met deze keys vanaf een ander netwerk.

## Starten (Docker Compose + vault)

Vanuit de **repo-root** (`cd` naar deze map). Geheimen komen uit **`~/.trading_vault`**.

**Optie A — hele stack (Compose bepaalt volgorde / depends_on):**

```bash
cd "$HOME/AI_Trading"
# shellcheck disable=SC1091
source ./scripts/lib_compose_root.sh
docker compose down --remove-orphans --timeout 30
docker compose up -d
```

(`down --remove-orphans` helpt tegen container-naamconflicten / orphans; de **`./scripts/run_*.sh`** scripts doen dit automatisch vóór build/up.)

**Optie B — per service (vault automatisch):**

```bash
cd "$HOME/AI_Trading"
./scripts/run_redis.sh build
./scripts/run_worker.sh build
./scripts/run_portal.sh build
```

Elk `run_*.sh`-script accepteert **`build`** (standaard) of **`rebuild`** (`docker compose build --no-cache` voor portal/worker). Zie `scripts/lib_run_container.sh`.

Logs: `docker compose logs -f portal` (of `worker` / `redis`). Stoppen: `docker compose stop`.

**Redis (host):** als je in de logs de melding over `vm.overcommit_memory` ziet, eenmalig op de **Linux-host** (niet in de container):

```bash
cd "$HOME/AI_Trading"
sudo ./scripts/install_vm_overcommit.sh
```

Dat schrijft `/etc/sysctl.d/99-ai-trading-redis.conf` en laadt `vm.overcommit_memory=1` direct.

Open daarna:
- `http://localhost:8000`

## Bitvavo advies (praktisch)

Bitvavo is een prima keuze voor een NL/EU crypto-flow, maar ik adviseer een gefaseerde inzet:

1. Start met paper-trading op dezelfde symbols/timeframes als je straks live wilt gebruiken.
2. Bouw daarna een `bitvavo` execution adapter met alleen:
   - market data pull,
   - order place/cancel,
   - position/balance sync.
3. Zet live trading pas aan na stabiele walk-forward en latency tests.

Belangrijkste live risico's bij Bitvavo (en elke CEX):
- onverwachte spread/slippage bij volatiele candles,
- API rate limits / tijdelijke timeouts,
- order partial fills en reconnect edge cases.

Mitigatie:
- strakke retry/backoff,
- idempotente order IDs,
- max order notional cap + daily kill switch.

## Bitvavo Rate Limit Aware client

`app/exchanges/bitvavo.py` bevat nu een rate-limit aware wrapper met:

- **Weight tracking**
  - Houdt lokaal request-weight bij per endpoint.
  - Verwerkt Bitvavo headers:
    - `bitvavo-ratelimit-limit`
    - `bitvavo-ratelimit-remaining`
    - `bitvavo-ratelimit-resetat`
- **Auto-throttle**
  - Pauzeert automatisch zodra usage >= 80% van het minuutlimiet tot reset-at.
- **429 exponential backoff**
  - Vangt HTTP 429 op.
  - Gebruikt `resetat` indien aanwezig, anders exponentiële wachttijd (`0.5s, 1s, 2s...`).
- **Idempotentie**
  - Ondersteunt `clientOrderId` op market orders.

## Risk Assessment (kritische valkuilen)

1. **Look-ahead bias**  
   Nieuws timestamps en feitelijke beschikbaarheid kunnen verschillen; zonder strict `available_at` discipline ontstaat leakage.

2. **False causality**  
   Correlatie tussen sentiment en prijs is regime-afhankelijk en vaak niet stabiel.

3. **Execution gap**  
   Paper resultaten zonder realistische spread/slippage/latency kunnen misleidend zijn.

4. **Data quality risk**  
   Duplicate/fake/rumor nieuws kan sentiment vervuilen zonder bronfilters.

5. **Overfitting**  
   Te veel features op te weinig regimes leidt tot slechte live generalisatie.

## GTX 1080 Efficiëntie-aanpak

- Houd feature engineering en risk engine CPU-gebaseerd.
- Gebruik GPU alleen voor lichte NLP inference (later: compacte FinBERT/DistilBERT).
- Gebruik kleine, snelle models voor prijssignalen (LightGBM/XGBoost als volgende stap).
- Werk met micro-batches voor nieuws-inference om VRAM usage stabiel te houden.

## Roadmap (volgende iteraties)

1. `available_at` timestamps toevoegen aan alle features/events.
2. Dedup + source reliability + novelty scoring in nieuws-pipeline.
3. Walk-forward backtesting met latency/slippage model.
4. Regime detector toevoegen en trades alleen toestaan in gevalideerde regimes.
5. Broker interface (paper -> live) met hard kill-switches.

## Documentation Policy

Deze README blijft leidend en wordt bij functionele wijzigingen bijgewerkt:
- architectuurwijzigingen,
- risk policy updates,
- nieuwe services/endpoints,
- deployment/startprocedure.

Kortom: documentatie wordt parallel onderhouden met codewijzigingen zodat ontwerp en implementatie synchroon blijven.
