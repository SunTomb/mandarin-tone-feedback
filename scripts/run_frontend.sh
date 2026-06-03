#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/mandarin-tone-feedback/frontend
mkdir -p ../logs

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not available on this server. Use scripts/run_web_static.sh with a synced frontend/dist build instead." >&2
  exit 1
fi

npm run dev -- --host 0.0.0.0 --port "${MTF_FRONTEND_PORT:-5173}" \
  2>&1 | tee "../logs/frontend_$(date +%Y%m%d_%H%M%S).log"
