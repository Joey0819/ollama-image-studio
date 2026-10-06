#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/app/outputs/logs"
PID_FILE="$ROOT/app/outputs/ollama-chat-image-bridge.pid"
LOG_FILE="$LOG_DIR/ollama-chat-image-bridge.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running / 已在运行: $(cat "$PID_FILE")"
  exit 0
fi

nohup "$ROOT/scripts/run_ollama_chat_bridge.sh" "$@" >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Started / 已启动: $(cat "$PID_FILE")"
echo "Log / 日志: $LOG_FILE"
