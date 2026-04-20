#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_FILE="${1:-$PROJECT_DIR/../trading_vault}"

if [[ ! -f "$VAULT_FILE" ]]; then
  echo "Vault-bestand niet gevonden: $VAULT_FILE" >&2
  echo "Gebruik: ./run_bot.sh /absoluut/pad/naar/trading_vault" >&2
  exit 1
fi

echo "Laad variabelen uit vault: $VAULT_FILE"
set -a
# shellcheck source=/dev/null
source "$VAULT_FILE"
set +a

echo "Ruim bestaande bot-container en resources op..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" down --remove-orphans --volumes --rmi local || true
docker rm -f ai-trading-bot >/dev/null 2>&1 || true

echo "Start schone rebuild van AI trading bot..."
docker compose -f "$PROJECT_DIR/docker-compose.yml" up --build
