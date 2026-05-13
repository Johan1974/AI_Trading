#!/bin/bash
# AI Trading Bot - Smart Build 2026 (Final Stable Edition)
clear
set -euo pipefail

source "${VAULT_PATH:-$HOME/.trading_vault}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export TZ="Europe/Amsterdam"

echo "--- 🛠️  Pre-flight & Hygiëne ---"
if command -v nvidia-smi &> /dev/null && ! nvidia-smi > /dev/null 2>&1; then
    echo "🚨 KRITIEK: NVIDIA Driver faalt!" && exit 1
fi

# Log schoonmaak met behoud van metrics
if [ "${PRESERVE_RL_METRICS_ON_RESET:-1}" = "1" ]; then
    find ./_logs_hub -mindepth 1 ! -name 'rl_training_metrics.sqlite' ! -name 'rl_hourly_metrics.jsonl' -delete 2>/dev/null || true
else
    find ./_logs_hub -mindepth 1 -delete 2>/dev/null || true
fi
docker container prune -f && docker image prune -f

echo "--- 🚀 Infra Opstarten ---"
# Stap 1: Redis & Ollama
docker compose up -d redis ollama
echo "   - Wachten op health..."
sleep 5

# Stap 2: Model Garantie (MyAssistant)
echo "   - Basismodel pull (llama3)..."
docker exec ai-trading-ollama ollama pull llama3
echo "   - MyAssistant alias creëren..."
docker exec ai-trading-ollama sh -c "echo 'FROM llama3\nPARAMETER temperature 0.7' > TemporaryModelfile"
docker exec ai-trading-ollama ollama create MyAssistant -f TemporaryModelfile
docker exec ai-trading-ollama rm TemporaryModelfile

# Stap 3: Janitor & Redis Policy
echo "🧹 Janitor activeren..."
docker compose up -d janitor
echo "   - Dwingen van Redis policy (allkeys-lru)..."
docker exec ai-trading-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Stap 4: Core Services (Inclusief News & Trainer)
# We voegen hier expliciet 'news' en 'trainer' toe die nu ontbreken in je docker ps
SERVICES="worker portal trainer news"
if [ "${1:-}" = "--force" ]; then
    echo "🚀 Full Rebuild..."
    docker compose up -d --build --force-recreate $SERVICES
else
    echo "♻️  Graceful Restart..."
    docker compose up -d $SERVICES
    docker compose restart worker portal
fi

echo "--- 🔍 Status Audit ---"
docker compose ps
bash ./scripts/fix_logs_hub_permissions.sh

echo ""
echo "--- 🕒 Container Sync ---"
for SERVICE in $(docker compose ps --services); do
    CNAME="ai-trading-${SERVICE}"
    if [ "$(docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" = "true" ]; then
        D_STR=$(docker exec "$CNAME" env TZ=Europe/Amsterdam date +"%H:%M:%S (%Z)" 2>/dev/null || echo "N/A")
        printf "✅ %-25s | %s\n" "$CNAME" "$D_STR"
    fi
done