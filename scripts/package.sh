#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="/Users/chengneng/Documents/Codex/2026-06-21/sol/outputs"
mkdir -p "$OUT_DIR"
tar --exclude='.venv' --exclude='__pycache__' --exclude='.pytest_cache' -czf "$OUT_DIR/ollama-image-studio.tar.gz" -C "$ROOT/.." "$(basename "$ROOT")"
echo "$OUT_DIR/ollama-image-studio.tar.gz"
