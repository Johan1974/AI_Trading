# Canonical Schema

Dit document beschrijft het universele (canonical) data schema voor de communicatie tussen de backend (FastAPI/Redis) en de frontend (Portal). Het doel is het uitbannen van inconsistente en variabele veldnamen.

## Object Structuur

Elk object dat over `/api/v1/stats` of een gecombineerde WebSocket wordt verstuurd, moet **exact** de volgende structuur bevatten:

- `market` (string): Het geselecteerde handelspaar (bijv. "BTC-EUR")
- `price` (float): De huidige prijs van het geselecteerde paar
- `cpu_load` (float): CPU belasting in procenten (0.0 - 100.0)
- `gpu_temp` (float): De maximale temperatuur van de actieve GPU in graden Celsius
- `ram_usage` (float): Werkgeheugengebruik in procenten (0.0 - 100.0)
- `decision_reasoning` (string): De AI-beredenering en context rondom het handelssignaal
- `bot_status` (string): De operationele status (bijv. "running", "paused")
- `trade_ledger` (list): Een lijst met de laatste 5 trades (bevat timestamp, side, price, pnl)
- `ai_weights` (object): Bevat de wegingen voor correlation, news en price voor de meters