#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/app/outputs/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Virtualenv missing / 虚拟环境不存在. Run scripts/setup.sh first / 请先运行 scripts/setup.sh" >&2
  exit 1
fi

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/ollama_chat_image_bridge.py" --watch "$@"
