#!/bin/bash
# Watchdog System for AI Trading Bot

# Laad vault voor Telegram configuratie
source "$HOME/.trading_vault" 2>/dev/null || true

# AFDWINGEN TIJDZONE (MANDAAT)
export TZ="Europe/Amsterdam"

LOGS_DIR="$(dirname "$0")/../_logs_hub"
OUTPUT_FILE="${LOGS_DIR}/assistant_focus.json"
PACKET_LOG="${LOGS_DIR}/packet_tracker.log"
RETRY_FILE="${LOGS_DIR}/watchdog_retries.txt"
LAST_ACTION="None"

# Stap 1: Mapstructuur en rechten
mkdir -p "$LOGS_DIR"
chmod 775 "$LOGS_DIR"

# Stap 2: Log Scanner (fouten tellen)
ERROR_COUNT=0
if ls "$LOGS_DIR"/portal_api.log "$LOGS_DIR"/worker_execution.log 1> /dev/null 2>&1; then
    ERROR_COUNT=$(grep -E 'ERROR|CRITICAL' "$LOGS_DIR/portal_api.log" "$LOGS_DIR/worker_execution.log" 2>/dev/null | wc -l | tr -d ' ')
fi
ERROR_COUNT=${ERROR_COUNT:-0}

# Stap 3: Redis Audit (state controle)
REDIS_KEYS=$(docker exec ai-trading-redis redis-cli keys '*' 2>/dev/null | tr '\n' ',' | sed 's/,$//')
HAS_SNAPSHOT=false
if echo "$REDIS_KEYS" | grep -q "worker_snapshot"; then
    HAS_SNAPSHOT=true
fi

# Stap 3a: Packet Tracking (Data Inspectie)
RAW_DATA="EMPTY"
RAW_PREVIEW=""
DATA_ERROR="None"
if [ "$HAS_SNAPSHOT" = true ]; then
    RAW_DATA=$(docker exec ai-trading-redis redis-cli hget worker_snapshot data 2>/dev/null)
    # Fallback als de snapshot als plain string is opgeslagen (WRONGTYPE op hget)
    if [[ "$RAW_DATA" == *"WRONGTYPE"* ]] || [[ -z "$RAW_DATA" ]]; then
        RAW_DATA=$(docker exec ai-trading-redis redis-cli get worker_snapshot 2>/dev/null)
    fi
    echo "$RAW_DATA" > "$PACKET_LOG"
    RAW_PREVIEW=$(echo "$RAW_DATA" | cut -c 1-100 | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr -d '\n' | tr -d '\r')
    
    # Stap 3b: Data Structure Validation
    if ! echo "$RAW_DATA" | grep -q "ETH-EUR"; then
        DATA_ERROR="Market data missing in snapshot (ETH-EUR not found)"
    fi
else
    echo "No snapshot found in Redis." > "$PACKET_LOG"
fi

# Stap 3c: Portal-Bridge Check
PORTAL_BRIDGE_ERRORS=0
if [ -f "$LOGS_DIR/portal_api.log" ]; then
    PORTAL_BRIDGE_ERRORS=$(grep -E -i 'NoneType.*object has no attribute|key not found' "$LOGS_DIR/portal_api.log" 2>/dev/null | wc -l | tr -d ' ')
fi
PORTAL_BRIDGE_ERRORS=${PORTAL_BRIDGE_ERRORS:-0}

# Initieer Retry Teller
if [ ! -f "$RETRY_FILE" ]; then
    echo "0" > "$RETRY_FILE"
fi
RETRIES=$(cat "$RETRY_FILE" | tr -dc '0-9')
RETRIES=${RETRIES:-0}

# Stap 4: Freshness Check (worker activiteit)
WORKER_LOG="$LOGS_DIR/worker_execution.log"
FRESH_WORKER=false
if [ -f "$WORKER_LOG" ] && [ -n "$(find "$WORKER_LOG" -mmin -10 2>/dev/null)" ]; then
    FRESH_WORKER=true
fi

# Stap 5: Telegram & Auto-Healing Logic
send_telegram() {
    local msg="$1"
    if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=${msg}" \
            -d "parse_mode=Markdown" > /dev/null
    fi
}

# --- PRODUCTIE: Disk Guard & Log Rotation ---
# Verwijder oude logs (ouder dan 7 dagen)
find "$LOGS_DIR" -type f \( -name "*.log*" -o -name "*.gz" -o -name "*.txt" \) -mtime +7 -exec rm -f {} \;

DISK_USAGE=$(df / | grep -v 'Filesystem' | awk '{ print $5 }' | sed 's/%//g' | head -n 1)
if [ -n "$DISK_USAGE" ] && [ "$DISK_USAGE" -gt 90 ]; then
    MSG="🚨 *CRITICAL DISK ALERT*%0AVPS Schijfruimte is ${DISK_USAGE}% vol! Gevaar voor database corruptie."
    send_telegram "$MSG"
fi

if [ "$FRESH_WORKER" = true ]; then
    echo "0" > "$RETRY_FILE" # Reset retries bij een gezonde worker
else
    if [ "$RETRIES" -lt 2 ]; then
        NEW_RETRIES=$((RETRIES + 1))
        echo "$NEW_RETRIES" > "$RETRY_FILE"
        LAST_ACTION="Auto-restart performed due to stall (Retry $NEW_RETRIES/2)"
        MSG="⚠️ *Watchdog Auto-Heal*

Worker logs gestopt voor >10 min. Initiating auto-restart ($NEW_RETRIES/2)..."
        send_telegram "$MSG"
        docker restart ai-trading-worker
    else
        LAST_ACTION="Emergency Stop performed after 2 failed retries"
        MSG="🚨 *Watchdog EMERGENCY STOP*

Worker kon na 2 auto-restarts niet herstellen. Bot is gestopt om schade te voorkomen."
        send_telegram "$MSG"
        docker stop ai-trading-worker ai-trading-portal
    fi
fi

# Stap 5b: Force Refresh Portal if no 200 OK
PORTAL_STALLED=false
if [ -f "$LOGS_DIR/portal_api.log" ]; then
    # Zoek naar de laatste snapshot data-request of API-FLOW log
    LAST_SNAPSHOT_LINE=$(grep -E 'GET /api/v1/snapshot HTTP/.* 200 OK|sent .* bytes from Redis' "$LOGS_DIR/portal_api.log" | tail -n 1)
    if [ -n "$LAST_SNAPSHOT_LINE" ]; then
        # Haal de ISO timestamp eruit
        LAST_TS=$(echo "$LAST_SNAPSHOT_LINE" | grep -oE '20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
        if [ -n "$LAST_TS" ]; then
            FIVE_MINS_AGO=$(date -d "5 minutes ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-5m +%Y-%m-%dT%H:%M:%S 2>/dev/null)
            # Lexicografische string vergelijking is 100% veilig en omzeilt timezone wiskunde
            if [ -n "$FIVE_MINS_AGO" ] && [[ "$LAST_TS" < "$FIVE_MINS_AGO" ]]; then
                PORTAL_STALLED=true
            fi
        fi
    else
        PORTAL_STALLED=true
    fi
fi

if [ "$PORTAL_STALLED" = true ]; then
    LAST_ACTION="Portal restarted (No successful 200 OK for snapshot in 5 mins)"
    MSG="⚠️ *Watchdog Auto-Heal*%0APortal gaf langer dan 5 min geen 200 OK voor de data request. Restarting ai-trading-portal..."
    send_telegram "$MSG"
    docker restart ai-trading-portal
fi

# Stap 6: DB Check (RL metrics database)
DB_PATH="$LOGS_DIR/rl_training_metrics.sqlite"
DB_SIZE_KB=0
DB_VALID=false
if [ -f "$DB_PATH" ]; then
    DB_SIZE_KB=$(du -k "$DB_PATH" 2>/dev/null | cut -f1)
    DB_SIZE_KB=${DB_SIZE_KB:-0}
    if [ "$DB_SIZE_KB" -gt 0 ]; then
        DB_VALID=true
    fi
fi

# Stap 8: Data Integriteit Check
DATA_VALID=false
if [[ "$RAW_DATA" == *"\"p\":"* ]]; then
    DATA_VALID=true
fi

# Stap 7: JSON Output genereren
FINAL_RETRIES=$(cat "$RETRY_FILE" | tr -dc '0-9')
FINAL_RETRIES=${FINAL_RETRIES:-0}

cat <<EOF > "$OUTPUT_FILE"
{
  "timestamp": "$(date +"%Y-%m-%dT%H:%M:%S%z")",
  "last_action": "$LAST_ACTION",
  "log_scanner": {
    "error_critical_count": $ERROR_COUNT
  },
  "portal_bridge_check": {
    "missing_key_errors": $PORTAL_BRIDGE_ERRORS
  },
  "redis_audit": {
    "keys_found": "$REDIS_KEYS",
    "has_worker_snapshot": $HAS_SNAPSHOT,
    "data_format_valid": $DATA_VALID,
    "data_error": "$DATA_ERROR"
  },
  "packet_preview": "${RAW_PREVIEW}...",
  "freshness_check": {
    "worker_log_updated_last_10m": $FRESH_WORKER,
    "retries_used": $FINAL_RETRIES
  },
  "db_check": {
    "rl_training_metrics_sqlite_size_kb": $DB_SIZE_KB,
    "is_valid_size": $DB_VALID
  }
}
EOF

echo "[WATCHDOG] Audit & Packet Trace voltooid. Resultaten opgeslagen in $OUTPUT_FILE"