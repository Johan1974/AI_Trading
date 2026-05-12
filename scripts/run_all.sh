#!/bin/bash
# AI Trading Bot - Smart Build 2026 (Optimized for Speed & Disk Space)
clear

set -euo pipefail

# Stap 1: Laad de geheimen (VAULT_PATH overschrijfbaar via env, default naast project-root)
source "${VAULT_PATH:-$HOME/.trading_vault}"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# AFDWINGEN TIJDZONE (MANDAAT)
export TZ="Europe/Amsterdam"

# --- NUCLEAR RESET ON STARTUP ---
echo "🧹 Diagnostische logs opschonen (met behoud van RL-metrics)..."
# Uitzetten: PRESERVE_RL_METRICS_ON_RESET=0 ./scripts/run_all.sh
if [ "${PRESERVE_RL_METRICS_ON_RESET:-1}" = "1" ]; then
    find ./_logs_hub -mindepth 1 \
      ! -name 'rl_training_metrics.sqlite' \
      ! -name 'rl_hourly_metrics.jsonl' \
      -delete 2>/dev/null || true
else
    find ./_logs_hub -mindepth 1 -delete 2>/dev/null || true
fi
chmod 775 ./_logs_hub

# --- IMAGE HYGIËNE ---
echo "🧹 Verwijder spook-containers en dangling images..."
docker container prune -f
docker image prune -f

# Stap 2: Redis eerst — worker en portal zijn afhankelijk van healthy Redis.
echo "🚀 Redis starten / bevestigen..."
docker compose up -d redis

# Wacht tot Redis healthy is (max 30s)
echo "   - Wachten op Redis health..."
for i in $(seq 1 15); do
    if docker compose ps redis | grep -q "healthy"; then
        echo "   - Redis healthy."
        break
    fi
    sleep 2
done

# Stap 3: Worker en portal bouwen en/of starten.
if [ "${1:-}" = "--force" ]; then
    echo "🚀 Services bouwen en starten (--force rebuild)..."
    docker compose up -d --build --force-recreate worker portal validator
else
    echo "♻️  Code herladen (Graceful restart worker + portal)..."
    docker compose up -d worker portal validator
    docker compose restart worker portal
fi

# Stap 4: Volumes opschonen (alleen volledig ontkoppelde volumes)
echo "🧹 Schijfruimte optimaliseren..."
docker volume prune -f

echo "✅ Klaar! Gebruik './scripts/run_all.sh --force' voor een volledige rebuild."

# Stap 5: Status check
echo ""
echo "🔍 Status:"
docker compose ps

# Stap 6: Permissies herstellen op _logs_hub
echo ""
echo "🔐 Permissies herstellen op _logs_hub..."
bash ./scripts/fix_logs_hub_permissions.sh

# Stap 7: Tijdzone-audit over alle containers
echo ""
echo "--- Universal Amsterdam Sync Audit ---"
echo "[INFO] Host (TZ=$TZ): $(date +"%F %T %z (%Z)")  |  epoch=$(date +%s)"

for SERVICE in $(docker compose ps -a --services); do
    CONTAINER_NAME="ai-trading-${SERVICE}"
    STOPPED=0
    EXEC_FAIL=0
    INSPECT_ENV=""

    IS_RUNNING=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo "false")

    if [ "$IS_RUNNING" = "true" ]; then
        DATE_STRING=$(docker exec "$CONTAINER_NAME" /bin/sh -c 'date +"%F %T %z (%Z)"' 2>/dev/null || true)
        [ -z "$DATE_STRING" ] && DATE_STRING=$(docker exec "$CONTAINER_NAME" date +"%F %T %z (%Z)" 2>/dev/null || true)
        [ -z "$DATE_STRING" ] && EXEC_FAIL=1
        EPOCH=$(docker exec "$CONTAINER_NAME" /bin/sh -c 'date +%s' 2>/dev/null || true)
        [ -z "$EPOCH" ] && EPOCH=$(docker exec "$CONTAINER_NAME" date +%s 2>/dev/null || true)
        TZ_IN=$(docker exec "$CONTAINER_NAME" /bin/sh -c 'printf %s "${TZ:-}"' 2>/dev/null || true)
        LT=$(docker exec "$CONTAINER_NAME" /bin/sh -c 'readlink -f /etc/localtime 2>/dev/null || true' 2>/dev/null || true)
    else
        STOPPED=1
        INSPECT_ENV=$(docker inspect "$CONTAINER_NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true)
        TZ_IN=$(echo "$INSPECT_ENV" | sed -n 's/^TZ=//p' | head -1)
        LT=$(docker inspect "$CONTAINER_NAME" --format '{{range .Mounts}}{{if eq .Destination "/etc/localtime"}}{{.Source}}{{"\n"}}{{end}}{{end}}' 2>/dev/null | head -1 || true)
    fi

    OK=0
    if [ "$STOPPED" -eq 1 ]; then
        echo "$INSPECT_ENV" | grep -q '^TZ=Europe/Amsterdam' && OK=1
        [ "$OK" -eq 0 ] && echo "$LT" | grep -q 'Amsterdam' && OK=1
    else
        [ "$TZ_IN" = "Europe/Amsterdam" ] && OK=1
        [ "$OK" -eq 0 ] && [ -n "${LT:-}" ] && echo "$LT" | grep -q 'Amsterdam' && OK=1
        [ "$OK" -eq 0 ] && echo "${DATE_STRING:-}" | grep -qE 'CEST|CET' && OK=1
    fi

    if [ "$EXEC_FAIL" -eq 1 ]; then
        printf "⚠️  %-28s  draait, maar 'docker exec date' gaf geen output\n" "$CONTAINER_NAME"
    elif [ "$STOPPED" -eq 1 ]; then
        if [ "$OK" -eq 1 ]; then
            printf "⏭️  %-28s  gestopt — config OK (TZ=%s)\n" "$CONTAINER_NAME" "${TZ_IN:-<geen TZ env>}"
        else
            printf "⚠️  %-28s  gestopt — TZ=%s (controleer compose)\n" "$CONTAINER_NAME" "${TZ_IN:-<leeg>}"
        fi
    elif [ "$OK" -eq 1 ]; then
        printf "✅ %-28s  %s  epoch=%s\n" "$CONTAINER_NAME" "$DATE_STRING" "${EPOCH:-?}"
        if [ -n "${EPOCH:-}" ]; then
            now_e=$(date +%s)
            skew=$((now_e - EPOCH))
            [ "$skew" -lt 0 ] && skew=$((0 - skew))
            [ "$skew" -gt 5 ] && printf "   %-28s  ⚠️  Tijdskew met host: %ss\n" "" "$skew"
        fi
    else
        printf "⚠️  %-25s: %s -> Incorrect Timezone!\n" "$CONTAINER_NAME" "${DATE_STRING:-?}"
    fi
done
echo "------------------------------------"

echo ""
echo "--- Laatste logs Portal ---"
docker compose logs --tail 10 portal
echo "--- Laatste logs Worker ---"
docker compose logs --tail 10 worker
