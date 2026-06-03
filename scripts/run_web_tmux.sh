#!/usr/bin/env bash
set -euo pipefail

SESSION="${MTF_TMUX_SESSION:-mtf-demo}"
PROJECT_DIR="/NAS/yesh/mandarin-tone-feedback"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n api "cd '$PROJECT_DIR' && ./scripts/run_api.sh"
tmux new-window -t "$SESSION" -n frontend "cd '$PROJECT_DIR' && ./scripts/run_frontend.sh"

echo "Started tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo "Local tunnel example: ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 <server>"
