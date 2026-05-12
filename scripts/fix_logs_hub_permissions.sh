#!/usr/bin/env bash
# Herstel eigenaar en schrijfrechten op ./_logs_hub na containers die als root schrijven
# (bijv. dashboard-validator). Gebruik: ./scripts/fix_logs_hub_permissions.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HUB="${ROOT}/_logs_hub"
mkdir -p "$HUB"
if docker info >/dev/null 2>&1; then
  docker run --rm -v "${HUB}:/logs" alpine:3.20 chown -R "$(id -u):$(id -g)" /logs
else
  echo "Waarschuwing: Docker niet bereikbaar; chown overgeslagen." >&2
fi
chmod -R u+rwX,go+rX "$HUB" 2>/dev/null || chmod -R 775 "$HUB"
echo "_logs_hub: $HUB (uid=$(id -u) gid=$(id -g))"
