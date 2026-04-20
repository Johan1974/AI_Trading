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
  - Haalt nieuws op via NewsAPI.

### 2) Analysis / Feature Layer
- `app/services/features.py`
  - Bouwt trend-features voor model.
  - Berekent simpele sentiment score uit headlines/beschrijvingen.

### 3) Signal Model
- `app/services/model.py`
  - Baseline lineaire regressie op recente slotkoersen.

### 4) Risk Engine
- `app/services/risk.py`
  - Signaal mapping (`BUY/SELL/HOLD`).
  - Risk controls: stop-loss, take-profit, position size fraction.

### 5) Execution (Paper)
- `app/services/execution.py`
  - Simuleert orderuitvoering inclusief fee-inschatting.

### 6) State & Monitoring
- `app/services/state.py`
  - Houdt activity-log en laatste prediction/order in memory.
- `app/main.py`
  - API endpoints + portal integratie.

## API Endpoints

- `GET /health` - service health.
- `GET /predict?ticker=AAPL` - genereert prediction + risk + paper order.
- `GET /activity` - actuele runtime status en events.
- `GET /` - webportal.

## Security & Secrets Policy

- Geen `.env` in de projectfolder.
- Alle geheimen staan in: `$HOME/.trading_vault`.
- Startscript: `./run_bot.sh` laadt alleen externe vault en faalt bij lokale `.env` bestanden.

Voorbeeld vault:

```bash
export NEWS_API_KEY=jouw_key_hier
export NEWS_QUERY=stock market OR economy OR inflation
export DEFAULT_TICKER=AAPL
export LOOKBACK_DAYS=400
```

## Starten

```bash
cd "$HOME/AI_Trading"
./run_bot.sh
```

Open daarna:
- `http://localhost:8000`

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
