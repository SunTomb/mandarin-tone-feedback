#!/usr/bin/env bash
set -euo pipefail

cd /NAS/yesh/mandarin-tone-feedback
source ./activate.sh

mkdir -p logs
python -m uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "${MTF_API_PORT:-8000}" \
  2>&1 | tee "logs/api_$(date +%Y%m%d_%H%M%S).log"
