#!/usr/bin/env bash
# Bestand: run_bot.sh
# Functie: Ondersteunde Docker-start met vault + API-checks; optioneel **self-healing** (--heal)
#   dat build → up → health → log-patrooncontrole herhaalt tot stabiel of max pogingen.
#
#   ./run_bot.sh                    -> achtergrond, compose up --build -d
#   ./run_bot.sh -f                 -> voorgrond compose up --build (logs in terminal)
#   ./run_bot.sh --heal             -> na start: poll logs op CUDA/driver/API-fouten; bij fout: down + retry
#   ./run_bot.sh --heal --follow    -> na succesvolle heal: `docker compose logs -f` (Ctrl+C stopt alleen logs)
#   ./run_bot.sh --no-cache         -> build --no-cache vóór start
#   ./run_bot.sh --clean            -> down --volumes --rmi local (zwaar)
#
# Data-onderhoud: na compose down draait `scripts/optimize_data.py` op de host (geen open container,
#   dus geen file locks door Docker). Python: $AI_TRADING_PYTHON, .venv/, venv/, anders python3;
#   zelfde mappen als compose: AI_TRADING_STORAGE_ROOT=$PROJECT_DIR/storage, RL_MODEL_DIR=$PROJECT_DIR/artifacts/rl.
#   Overslaan: --skip-optimize. PEP 668: python3 -m venv .venv && .venv/bin/pip install pandas pyarrow
#   of: sudo apt install python3-pandas python3-pyarrow
#
# Omgeving: GENESIS_MAX_HEAL (default 6) max heal-cycli; HEALTH_PORT (default 8000).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_FILE="${HOME}/.trading_vault"
MODE="background"
OPTIMIZE_SCRIPT="$PROJECT_DIR/scripts/optimize_data.py"
CLEAN=0
NO_CACHE=0
SKIP_OPTIMIZE=0
HEAL=0
FOLLOW_LOGS=0

for arg in "$@"; do
  case "$arg" in
    --foreground|-f|--interactive|-i)
      MODE="interactive"
      ;;
    --background|-b)
      MODE="background"
      ;;
    --clean)
      CLEAN=1
      ;;
    --no-cache)
      NO_CACHE=1
      ;;
    --skip-optimize)
      SKIP_OPTIMIZE=1
      ;;
    --heal)
      HEAL=1
      ;;
    --follow|--follow-logs)
      FOLLOW_LOGS=1
      ;;
  esac
done

COMPOSE_CMD=(docker compose -f "$PROJECT_DIR/docker-compose.yml")
MAX_HEAL="${GENESIS_MAX_HEAL:-6}"
HEALTH_PORT="${HEALTH_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${HEALTH_PORT}/health"

if [[ -f "$VAULT_FILE" ]]; then
  echo "Laad vault-variabelen uit $VAULT_FILE"
  set -a
  if ! source <(sed -E 's/^[[:space:]]*EXPORT[[:space:]]+/export /' "$VAULT_FILE"); then
    echo "Waarschuwing: vault kon niet volledig worden gesourced." >&2
  fi
  set +a
else
  echo "Geen vault gevonden op $VAULT_FILE, ga door met bestaande environment."
fi

if [[ -z "${COINMARKETCAP_KEY:-}" && -n "${CMC_API_KEY:-}" ]]; then
  export COINMARKETCAP_KEY="${CMC_API_KEY}"
fi

if [[ -z "${CRYPTOCOMPARE_KEY:-}" ]]; then
  echo "Waarschuwing: CRYPTOCOMPARE_KEY ontbreekt."
fi
if [[ -z "${COINMARKETCAP_KEY:-}" ]]; then
  echo "Waarschuwing: COINMARKETCAP_KEY ontbreekt."
fi

echo "Voer API Health Check uit (Bitvavo, CryptoCompare, CoinMarketCap)..."
BITVAVO_OK=0
CRYPTOCOMPARE_OK=0
CMC_OK=0

if curl -sS --max-time 10 "https://api.bitvavo.com/v2/ticker/price?market=BTC-EUR" >/dev/null; then
  BITVAVO_OK=1
fi

if [[ -n "${CRYPTOCOMPARE_KEY:-}" ]]; then
  if curl -sS --max-time 12 -H "authorization: Apikey ${CRYPTOCOMPARE_KEY}" \
    "https://min-api.cryptocompare.com/data/v2/news/?lang=EN" >/dev/null; then
    CRYPTOCOMPARE_OK=1
  fi
fi

if [[ -n "${COINMARKETCAP_KEY:-}" ]]; then
  if curl -sS --max-time 12 -H "X-CMC_PRO_API_KEY: ${COINMARKETCAP_KEY}" -H "Accept: application/json" \
    "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest" >/dev/null; then
    CMC_OK=1
  fi
fi

echo "Health Check result -> Bitvavo:${BITVAVO_OK} CryptoCompare:${CRYPTOCOMPARE_OK} CMC:${CMC_OK}"
if [[ "$BITVAVO_OK" -ne 1 || "$CRYPTOCOMPARE_OK" -ne 1 || "$CMC_OK" -ne 1 ]]; then
  echo "Waarschuwing: niet alle externe API checks zijn OK; startup gaat door."
fi

echo "Ruim bestaande bot-container(s) op..."
if [[ "$CLEAN" -eq 1 ]]; then
  echo "Modus --clean: compose down met --volumes en --rmi local."
  "${COMPOSE_CMD[@]}" down --remove-orphans --volumes --rmi local || true
else
  "${COMPOSE_CMD[@]}" down --remove-orphans || true
fi
docker rm -f ai-trading-bot >/dev/null 2>&1 || true

if [[ "$SKIP_OPTIMIZE" -ne 1 && -f "$OPTIMIZE_SCRIPT" ]]; then
  OPTIMIZE_PYTHON=""
  if [[ -n "${AI_TRADING_PYTHON:-}" && -x "${AI_TRADING_PYTHON}" ]]; then
    OPTIMIZE_PYTHON="$AI_TRADING_PYTHON"
  elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    OPTIMIZE_PYTHON="$PROJECT_DIR/.venv/bin/python"
  elif [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
    OPTIMIZE_PYTHON="$PROJECT_DIR/.venv/bin/python3"
  elif [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    OPTIMIZE_PYTHON="$PROJECT_DIR/venv/bin/python"
  else
    OPTIMIZE_PYTHON="$(command -v python3)"
  fi
  echo "Data-onderhoud op host (${OPTIMIZE_PYTHON} → repo storage/artifacts): scripts/optimize_data.py"
  if "$OPTIMIZE_PYTHON" -c "import pandas, pyarrow" >/dev/null 2>&1; then
    env \
      AI_TRADING_STORAGE_ROOT="$PROJECT_DIR/storage" \
      RL_MODEL_DIR="$PROJECT_DIR/artifacts/rl" \
      "$OPTIMIZE_PYTHON" "$OPTIMIZE_SCRIPT" || echo "Waarschuwing: optimize_data.py faalde; ga door."
  else
    echo "Waarschuwing: ${OPTIMIZE_PYTHON} mist pandas/pyarrow; sla optimize_data.py over."
    echo "  Opties: --skip-optimize | .venv: python3 -m venv .venv && .venv/bin/pip install pandas pyarrow | apt: python3-pandas python3-pyarrow"
  fi
fi

docker_build() {
  if [[ "$NO_CACHE" -eq 1 ]]; then
    echo "Docker build --no-cache..."
    "${COMPOSE_CMD[@]}" build --no-cache
  else
    echo "Docker compose build..."
    "${COMPOSE_CMD[@]}" build
  fi
}

wait_for_http_health() {
  local deadline=$((SECONDS + 45))
  while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf --max-time 4 "$HEALTH_URL" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# Fatale patronen (Error/Traceback/CUDA) — los van CUDA-device succesregel.
logs_show_fatal_patterns() {
  local blob
  blob=$("${COMPOSE_CMD[@]}" logs --no-color --tail=800 2>&1 || true)
  if echo "$blob" | grep -Eiq \
    'Traceback \(most recent call last\)|CRITICAL: GPU|CUDA mismatch|Driver error|driver mismatch|CUDA out of memory|out of memory|RuntimeError.*CUDA|CUDA error|version mismatch|NVML.*Failed|Could not load library libcudnn|libcudart|no CUDA-capable device|Address already in use|Killed process|OOMKilled|API Connection Error|Connection refused.*8000'; then
    echo "---- Gevonden foutfragment (laatste treffers) ----"
    echo "$blob" | grep -Ei \
      'Traceback \(most recent call last\)|CRITICAL: GPU|CUDA mismatch|Driver error|driver mismatch|CUDA out of memory|out of memory|RuntimeError.*CUDA|CUDA error|version mismatch|NVML|libcudart|no CUDA-capable|Address already in use|Killed process|OOMKilled|API Connection Error|Connection refused.*8000' \
      | tail -n 16 || true
    echo "------------------------------------------------"
    genesis_suggest_fix "$blob"
    return 0
  fi
  return 1
}

# Succes: backend moet `[DEVICE] Using device: cuda:0 (...)` loggen (GPU actief).
wait_for_cuda_device_in_logs() {
  local deadline=$((SECONDS + 120))
  echo "Wacht op Torch device-log ([DEVICE] Using device: ...) ..."
  while [[ $SECONDS -lt $deadline ]]; do
    local blob
    blob=$("${COMPOSE_CMD[@]}" logs --no-color --tail=900 2>&1 || true)
    if echo "$blob" | grep -Fq "[DEVICE] Using device: cuda:0"; then
      echo "[GENESIS] CUDA-device bevestigd in containerlogs."
      return 0
    fi
    if [[ "${GENESIS_HEAL_ALLOW_CPU_LOG:-0}" == "1" ]] && echo "$blob" | grep -Fq "[DEVICE] Using device: cpu"; then
      echo "[GENESIS] CPU-device in logs geaccepteerd (GENESIS_HEAL_ALLOW_CPU_LOG=1)."
      return 0
    fi
    if logs_show_fatal_patterns; then
      return 1
    fi
    sleep 2
  done
  echo "[GENESIS] Timeout: geen [DEVICE] Using device:-regel in logs binnen 120s."
  echo "  Tip: GPU vereist? Check nvidia-smi op host + compose GENESIS_REQUIRE_GPU=1."
  echo "  CPU-only heal: export GENESIS_HEAL_ALLOW_CPU_LOG=1 (en eventueel GENESIS_REQUIRE_GPU=0)."
  "${COMPOSE_CMD[@]}" logs --tail=120 || true
  return 1
}

genesis_suggest_fix() {
  local blob="$1"
  echo ""
  echo "[GENESIS] Suggesties (voer uit in repo of host, daarna opnieuw ./run_bot.sh --heal):"
  if echo "$blob" | grep -Eiq 'CUDA out of memory|out of memory|OOMKilled'; then
    echo "  - Zet in compose/vault: FINBERT_USE_CUDA=0 of verlaag batch/lookback; sluit andere GPU-processen."
  fi
  if echo "$blob" | grep -Eiq 'driver mismatch|version mismatch|NVML|libcudart|no CUDA-capable|CRITICAL: GPU'; then
    echo "  - Host: nvidia-smi + NVIDIA Container Toolkit; Dockerfile: CUDA 12.4 runtime + torch 2.2.0+cu121 (PyTorch cu121 index)."
  fi
  if echo "$blob" | grep -Eiq 'Address already in use'; then
    echo "  - Poort ${HEALTH_PORT} bezet: stop andere uvicorn of wijzig poortmapping in docker-compose.yml."
  fi
  if echo "$blob" | grep -Eiq 'API Connection Error|Connection refused'; then
    echo "  - Controleer vault-keys (Bitvavo/CryptoCompare/CMC) en netwerk/firewall."
  fi
  echo ""
}

genesis_heal_loop() {
  local attempt=0
  while [[ $attempt -lt $MAX_HEAL ]]; do
    attempt=$((attempt + 1))
    echo ""
    echo "========== GENESIS HEAL cyclus $attempt / $MAX_HEAL =========="
    docker_build
    echo "Compose up -d..."
    "${COMPOSE_CMD[@]}" up -d
    echo "Wacht op HTTP health ($HEALTH_URL)..."
    if ! wait_for_http_health; then
      echo "[GENESIS] Health-endpoint niet bereikbaar binnen timeout."
      "${COMPOSE_CMD[@]}" logs --tail=80 || true
      "${COMPOSE_CMD[@]}" down --remove-orphans || true
      sleep 4
      continue
    fi
    sleep 2
    if ! wait_for_cuda_device_in_logs; then
      echo "[GENESIS] CUDA-logcheck faalde — compose down en retry."
      "${COMPOSE_CMD[@]}" down --remove-orphans || true
      sleep 5
      continue
    fi
    if logs_show_fatal_patterns; then
      echo "[GENESIS] Nadien toch fatale patronen — compose down en retry."
      "${COMPOSE_CMD[@]}" down --remove-orphans || true
      sleep 5
      continue
    fi
    echo "[GENESIS] Logs OK: CUDA-device + geen bekende fatale patronen."
    if [[ "$FOLLOW_LOGS" -eq 1 ]]; then
      echo "Volg logs (Ctrl+C stopt alleen logstream):"
      exec "${COMPOSE_CMD[@]}" logs -f
    fi
    return 0
  done
  echo "[GENESIS] Max heal-pogingen ($MAX_HEAL) bereikt. Bekijk logs: ${COMPOSE_CMD[*]} logs --tail=200"
  return 1
}

start_stack_plain() {
  if [[ "$NO_CACHE" -eq 1 ]]; then
    "${COMPOSE_CMD[@]}" build --no-cache
    if [[ "$MODE" == "background" ]]; then
      "${COMPOSE_CMD[@]}" up -d
    else
      exec "${COMPOSE_CMD[@]}" up
    fi
  elif [[ "$MODE" == "background" ]]; then
    "${COMPOSE_CMD[@]}" up --build -d
  else
    exec "${COMPOSE_CMD[@]}" up --build
  fi
}

if [[ "$HEAL" -eq 1 ]]; then
  if [[ "$MODE" == "interactive" ]]; then
    echo "[GENESIS] --heal met voorgrondmodus: heal-loop draait; daarna blijft compose in voorgrond niet automatisch — gebruik achtergrond + --follow."
  fi
  genesis_heal_loop
  exit $?
fi

if [[ "$MODE" == "background" ]]; then
  echo "Start bot in achtergrond op http://localhost:${HEALTH_PORT}"
  start_stack_plain
  echo "Klaar. Logs: docker compose -f \"$PROJECT_DIR/docker-compose.yml\" logs -f"
  echo "Self-heal bij problemen: ./run_bot.sh --heal   of   GENESIS_MAX_HEAL=8 ./run_bot.sh --heal --follow"
else
  echo "Start bot in voorgrond (logs in terminal)"
  start_stack_plain
fi
