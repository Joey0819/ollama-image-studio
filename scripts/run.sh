#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-7861}"
HOST="${HOST:-127.0.0.1}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b}"
export IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-local-cell-renderer}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Virtualenv missing / 虚拟环境不存在. Run scripts/setup.sh first / 请先运行 scripts/setup.sh" >&2
  exit 1
fi

cd "$ROOT"
exec "$ROOT/.venv/bin/python" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
