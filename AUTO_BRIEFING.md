# AUTO_BRIEFING — 2026-05-13 01:00 Europe/Amsterdam

## ⚠️ CRITICAL ALERTS

> 🚨 **CONTAINER HERSTART:** startup_mode='auto' → de worker is onverwacht herstart (crash-loop of restart:always). Controleer de logs.

---

## Global Steps Groei per Markt

- Onvoldoende data (< 2 chunks beschikbaar).

## AI Prediction Accuraatheid

*Afwijking = predicted_price[0] t.o.v. huidige prijs. Policy confidence = dominant_policy_prob.*

- Redis niet beschikbaar of geen prediction data.

## Learning Plateau Detectie

- ✅ Geen plateau gedetecteerd op basis van beschikbare data.

## RL Intelligence Samenvatting

| Metric | Waarde |
|--------|--------|
| Replay buffer entries | 3228 |
| Entries met reward_pct | 40 |
| Training chunks totaal | 0 |
| Training chunks (24u) | 0 |
| Laatste training | — |
| Global step (laatste) | — |
| Avg loss (laatste uur) | -0.00496964 |
| Avg cumulative reward | -3.550311 |
| Policy probs buy/hold/sell | 0.079 / 0.809 / 0.094 |
| Signalen 24u BUY/HOLD/SELL | 35 / 1290 / 1863 |

## Anomaly Detection

- ⚠️ Geen training chunks in 24u (laatste: —). Model leert niet.

## 🎯 Action Plan

> **Voer deze acties uit bij het begin van je volgende sessie.**

**1.** **Crashoorzaak opsporen** — `cat _logs_hub/persistent_crash.log | tail -50`. Zet `EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART=1` in `docker-compose.env` voor uitgebreid startup-rapport bij volgende herstart.

**2.** **Forceer RL-training** — `POST /api/v1/rl-train` met `{"force": true}`, of zet `RL_BACKGROUND_TRAIN=1` + `RL_TRAIN_INTERVAL_SEC=60` in `docker-compose.env`.

## Model Versies

- **ppo_ADA-EUR_models.json** | ts: `20260510T210729Z` | reward_score: `-6773.845104` | `artifacts/rl/ppo_ADA-EUR_20260510T210729Z.zip`
- **ppo_BTC-EUR_models.json** | ts: `20260510T210005Z` | reward_score: `-7975.209176` | `artifacts/rl/ppo_BTC-EUR_20260510T210005Z.zip`
- **ppo_ETH-EUR_models.json** | ts: `20260510T210134Z` | reward_score: `-1134.551437` | `artifacts/rl/ppo_ETH-EUR_20260510T210134Z.zip`
- **ppo_HYPE-EUR_models.json** | ts: `20260512T081926Z` | reward_score: `-2184.293187` | `artifacts/rl/ppo_HYPE-EUR_20260512T081926Z.zip`
- **ppo_LINK-EUR_models.json** | ts: `20260512T133356Z` | reward_score: `-633.202765` | `artifacts/rl/ppo_LINK-EUR_20260512T133356Z.zip`
- **ppo_SOL-EUR_models.json** | ts: `20260512T210722Z` | reward_score: `-477.391812` | `artifacts/rl/ppo_SOL-EUR_20260512T210722Z.zip`
- **ppo_TON-EUR_models.json** | ts: `20260511T233226Z` | reward_score: `-192.676552` | `artifacts/rl/ppo_TON-EUR_20260511T233226Z.zip`
- **ppo_XRP-EUR_models.json** | ts: `20260510T210258Z` | reward_score: `-4822.248383` | `artifacts/rl/ppo_XRP-EUR_20260510T210258Z.zip`

## Reward Trend (laatste uurblokken)

```
2026-05-12 12:10 UTC     -4.90  █████████████████████████
2026-05-12 13:10 UTC     -4.93  █████████████████████████
2026-05-12 14:10 UTC     -9.01  ████████████████████
2026-05-12 15:10 UTC     -5.18  ████████████████████████
2026-05-12 16:10 UTC     -5.08  ████████████████████████
2026-05-12 17:10 UTC     -5.02  ████████████████████████
2026-05-12 18:10 UTC     -4.76  █████████████████████████
2026-05-12 19:10 UTC     -4.09  █████████████████████████
2026-05-12 21:03 UTC     -4.80  █████████████████████████
2026-05-12 22:25 UTC     -3.55  ██████████████████████████
```

---
*Gegenereerd: 2026-05-13 01:00 Europe/Amsterdam — `scripts/generate_briefing.py`*
*Volgende update: over ~1 uur (via `_rl_hourly_checkpoint_and_metrics_loop`)*
