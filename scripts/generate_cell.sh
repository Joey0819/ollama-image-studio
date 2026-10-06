#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7861}"
curl -sS "http://127.0.0.1:${PORT}/api/generate" \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "生一个细胞图片",
    "ollama_model": "gemma4:31b",
    "image_model": "local-cell-renderer",
    "use_ollama": true,
    "style": "scientific",
    "width": 512,
    "height": 512,
    "steps": 6
  }'
