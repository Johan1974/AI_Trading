#!/bin/bash
# AI Trading Bot - Smart Build 2026 (Optimized for Speed & Disk Space)
clear

# Stap 1: Laad de geheimen
source "$HOME/.trading_vault"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

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
echo "♻️  Code herladen (Razendsnel)..."
docker compose restart worker portal

# Stap 4: De "Hygiëne" (Zonder je snelheid te verpesten)
echo "🧹 Schijfruimte optimaliseren (Achtergrond)..."

# Verwijder oude images van mislukte of vorige builds (de echte schijf-vreters).
docker image prune -f

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

echo "--- Laatste logs Portal ---"
docker compose logs --tail 10 portal
echo "--- Laatste logs Worker ---"
docker compose logs --tail 10 worker