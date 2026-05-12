#!/usr/bin/env bash
# Herstel schrijfrechten op _logs_hub na Docker-runs als root (gemengde root/johan files).
# Gebruik: ./scripts/fix_logs_hub.sh
# Bij 'Permission denied' op root-owned files: sudo ./scripts/fix_logs_hub.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HUB="${LOGS_HUB:-$ROOT/_logs_hub}"
U="$(id -un)"
G="$(id -gn)"

mkdir -p "$HUB"

if ! chown -R "$U:$G" "$HUB" 2>/dev/null; then
  echo "chown failed (likely root-owned files). Run:" >&2
  echo "  sudo chown -R $U:$G $HUB" >&2
  exit 1
fi

echo "OK: $HUB owned by $U:$G"
exit 0
