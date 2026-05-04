#!/bin/bash
# AI Trading Bot - Smart Build 2026 (Optimized for Speed & Disk Space)
clear

# Stap 1: Laad de geheimen
source "$HOME/.trading_vault"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

# AFDWINGEN TIJDZONE (MANDAAT)
# Zorg dat de policy altijd wordt nageleefd, zelfs als de vault deze mist.
export TZ="Europe/Amsterdam"

# --- NUCLEAR RESET ON STARTUP ---
echo "🧹 Nuclear Reset: Diagnostische logs & databases wissen (100% clean slate)..."
# Leeg de map volledig (inclusief databases en verborgen bestanden), maar behoud de map zelf voor Docker mounts
find ./_logs_hub -mindepth 1 -delete
chmod 777 ./_logs_hub

# --- SCHOON VEEG PROTOCOL ---
echo "🧹 Image Hygiëne: Verwijder spook-containers en dangling images vóór de build..."
docker container prune -f
docker image prune -f

# Stap 2: Slimme herstart. 
# We gebruiken GEEN 'down' want dat stopt de boel te bruusk.
# We gebruiken GEEN '--build' standaard, omdat je code via volumes (live) wordt geladen.
if [ "$1" == "--force" ]; then
    echo "🚀 Services bouwen en starten (--force)..."
    docker compose up -d --build
else
    echo "🚀 Services controleren/starten..."
    docker compose up -d
fi

# Stap 3: Alleen de Python processen herstarten om nieuwe code te laden.
# Omdat je ./app en ./core hebt gemount in compose, ziet de bot je wijzigingen 
# direct ZONDER dat er een nieuwe image van 16GB gebouwd hoeft te worden.
echo "♻️  Code herladen (Graceful restart)..."
docker compose stop portal worker
echo "   - Containers gestopt. Herstarten in juiste volgorde (worker eerst)..."
docker compose up -d worker portal

# Stap 4: De "Hygiëne" (Zonder je snelheid te verpesten)
echo "🧹 Schijfruimte optimaliseren (Achtergrond)..."

# VERWIJDER DE BUILDER PRUNE HIER: 
# Doe dit alleen handmatig als je echt ruimtegebrek hebt. 
# De cache zorgt namelijk voor die 0.0s herstarts.

# Verwijder alleen volumes die echt nergens meer aan vast zitten.
docker volume prune -f

echo "✅ Klaar! Gebruik './scripts/run_all.sh --force' als je echt een nieuwe image wilt bouwen."

# Stap 5: De "Reality Check"
echo "🔍 Status check:"
sleep 2
docker compose ps

echo ""

echo "🔐 Permissies herstellen op _logs_hub (zonder sudo wachtwoord via Docker)..."
docker run --rm -v "$(pwd)/_logs_hub:/logs" redis:7-alpine chown -R $(id -u):$(id -g) /logs
chmod -R 775 ./_logs_hub

echo "--- Universal Amsterdam Sync Audit ---"

echo "[INFO] Timezone loaded from .trading_vault: $TZ"

# Loop through all services defined in docker-compose and check their time
for SERVICE in $(docker compose ps -a --services); do
    CONTAINER_NAME="ai-trading-${SERVICE}"
    
    # Check if container is running
    IS_RUNNING=$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)
    
    if [ "$IS_RUNNING" == "true" ]; then
        DATE_STRING=$(docker exec "$CONTAINER_NAME" date 2>/dev/null)
    else
        # Run a temporary fast container to check the timezone for exited services
        DATE_STRING=$(docker compose run --rm --entrypoint date "$SERVICE" 2>/dev/null)
        DATE_STRING="$DATE_STRING (Exited)"
    fi
    
    if echo "$DATE_STRING" | grep -q -E 'CEST|Europe/Amsterdam'; then
        printf "✅ %-25s: %s\n" "$CONTAINER_NAME" "$DATE_STRING"
    else
        printf "⚠️  [WARNING] %-22s: %s -> Incorrect Timezone!\n" "$CONTAINER_NAME" "$DATE_STRING"
    fi
done
echo "------------------------------------"

echo "--- Laatste logs Portal ---"
docker compose logs --tail 10 portal
echo "--- Laatste logs Worker ---"
docker compose logs --tail 10 worker