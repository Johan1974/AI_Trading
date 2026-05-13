# AUTO_BRIEFING — 2026-05-13 16:00 Europe/Amsterdam

## ⚠️ CRITICAL ALERTS

> ✅ Geen kritieke meldingen.

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
| Replay buffer entries | 6764 |
| Entries met reward_pct | 83 |
| Training chunks totaal | 0 |
| Training chunks (24u) | 0 |
| Laatste training | — |
| Global step (laatste) | — |
| Avg loss (laatste uur) | -0.00165598 |
| Avg cumulative reward | -1.968847 |
| Policy probs buy/hold/sell | 0.062 / 0.846 / 0.076 |
| Signalen 24u BUY/HOLD/SELL | 67 / 3198 / 3416 |

## Anomaly Detection

- ⚠️ Geen training chunks in 24u (laatste: —). Model leert niet.

## 🎯 Action Plan

> **Voer deze acties uit bij het begin van je volgende sessie.**

**1.** **Forceer RL-training** — `POST /api/v1/rl-train` met `{"force": true}`, of zet `RL_BACKGROUND_TRAIN=1` + `RL_TRAIN_INTERVAL_SEC=60` in `docker-compose.env`.

## Model Versies

- **ppo_ADA-EUR_models.json** | ts: `20260510T210729Z` | reward_score: `-6773.845104` | `artifacts/rl/ppo_ADA-EUR_20260510T210729Z.zip`
- **ppo_BTC-EUR_models.json** | ts: `20260510T210005Z` | reward_score: `-7975.209176` | `artifacts/rl/ppo_BTC-EUR_20260510T210005Z.zip`
- **ppo_DOGE-EUR_models.json** | ts: `20260513T093340Z` | reward_score: `-345.479935` | `artifacts/rl/ppo_DOGE-EUR_20260513T093340Z.zip`
- **ppo_ETH-EUR_models.json** | ts: `20260510T210134Z` | reward_score: `-1134.551437` | `artifacts/rl/ppo_ETH-EUR_20260510T210134Z.zip`
- **ppo_LINK-EUR_models.json** | ts: `20260512T133356Z` | reward_score: `-633.202765` | `artifacts/rl/ppo_LINK-EUR_20260512T133356Z.zip`
- **ppo_SOL-EUR_models.json** | ts: `20260513T093819Z` | reward_score: `-624.511038` | `artifacts/rl/ppo_SOL-EUR_20260513T093819Z.zip`
- **ppo_TON-EUR_models.json** | ts: `20260511T233226Z` | reward_score: `-192.676552` | `artifacts/rl/ppo_TON-EUR_20260511T233226Z.zip`
- **ppo_XRP-EUR_models.json** | ts: `20260513T124739Z` | reward_score: `-453.05931` | `artifacts/rl/ppo_XRP-EUR_20260513T124739Z.zip`

## Reward Trend (laatste uurblokken)

```
2026-05-13 01:25 UTC     -2.00  ███████████████████████████
2026-05-13 02:25 UTC     -2.03  ███████████████████████████
2026-05-13 03:25 UTC     -2.00  ███████████████████████████
2026-05-13 04:25 UTC     -1.95  ████████████████████████████
2026-05-13 05:25 UTC     -3.60  ██████████████████████████
2026-05-13 06:25 UTC     -2.07  ███████████████████████████
2026-05-13 07:25 UTC     -2.05  ███████████████████████████
2026-05-13 08:25 UTC     -2.01  ███████████████████████████
2026-05-13 09:25 UTC     -2.05  ███████████████████████████
2026-05-13 11:04 UTC     -1.97  ████████████████████████████
```

---
*Gegenereerd: 2026-05-13 16:00 Europe/Amsterdam — `scripts/generate_briefing.py`*
*Volgende update: over ~1 uur (via `_rl_hourly_checkpoint_and_metrics_loop`)*
