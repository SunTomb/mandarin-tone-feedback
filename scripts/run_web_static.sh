#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/mandarin-tone-feedback
source ./activate.sh

if [ ! -f frontend/dist/index.html ]; then
  echo "frontend/dist/index.html is missing. Build frontend locally with 'npm run build' and sync frontend/dist before running this script." >&2
  exit 1
fi

mkdir -p logs
python -m uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "${MTF_WEB_PORT:-8000}" \
  2>&1 | tee "logs/web_$(date +%Y%m%d_%H%M%S).log"
