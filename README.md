# Ollama Image Studio / Ollama 本地生图工作台

This local app fixes the “I cannot see the picture” problem by separating the jobs correctly:

这个本地应用用于解决“图片没有显示”的问题，它把工作链路分清楚：

- Ollama LLM improves the Chinese or English prompt. / Ollama 大语言模型负责优化中文或英文提示词。
- A local renderer or optional Stable Diffusion model renders the actual PNG image. / 本地渲染器或可选 Stable Diffusion 图像模型负责真正生成 PNG 图片。
- The browser UI displays the saved image file directly. / 浏览器界面直接展示保存后的图片文件。

## Run / 运行

```bash
cd /Users/chengneng/Documents/Codex/2026-06-21/sol/work/ollama-image-studio
scripts/setup.sh
scripts/run.sh
```

Open / 打开:

```text
http://127.0.0.1:7861
```

## Defaults / 默认设置

- Ollama URL / Ollama 地址: `http://127.0.0.1:11434`
- Ollama model / Ollama 模型: `gemma4:31b`
- Image model / 图像模型: `local-cell-renderer`
- Output folder / 输出目录: `app/outputs/images`

You can override the model before running:

你可以在运行前替换模型：

```bash
OLLAMA_MODEL=qwen3.6:latest IMAGE_MODEL_ID=stabilityai/sd-turbo scripts/run.sh
```

`local-cell-renderer` does not download model weights. It uses Ollama to plan the prompt, then renders a scientific cell PNG locally with Pillow. Enter a Diffusers model ID such as `segmind/tiny-sd` or `stabilityai/sd-turbo` only when you want the optional Stable Diffusion backend.

`local-cell-renderer` 不需要下载模型权重。它使用 Ollama 规划提示词，然后用 Pillow 在本地渲染科学细胞 PNG。只有在你需要可选的 Stable Diffusion 后端时，才输入 `segmind/tiny-sd` 或 `stabilityai/sd-turbo` 这类 Diffusers 模型 ID。

## Why the Ollama chat failed / 为什么 Ollama 聊天里失败

The Ollama chat displayed a JSON-like string such as `{"action":"dalle.text2im" ...}`. That was only text. The local Ollama app did not have a real DALL-E tool connected, so no image file was produced.

Ollama 聊天界面里显示的 `{"action":"dalle.text2im" ...}` 只是普通文字。本地 Ollama 应用没有真正连接 DALL-E 工具，所以不会产生图片文件。

## Show Images Inside Ollama Chat / 让图片显示在 Ollama 聊天里

The Ollama desktop UI currently suppresses Markdown images in assistant messages. The bridge script watches Ollama's local SQLite chat database. When it sees a fake `dalle.text2im` message, it generates a PNG through this local app and inserts the PNG back into the same Ollama chat as an image attachment.

Ollama 桌面端目前会抑制 assistant 消息里的 Markdown 图片。桥接脚本会监听 Ollama 本地 SQLite 聊天数据库；当它发现假的 `dalle.text2im` 消息时，会通过本应用生成 PNG，并把 PNG 作为图片附件插回同一个 Ollama 聊天。

Run once / 运行一次：

```bash
cd /Users/chengneng/Documents/Codex/2026-06-21/sol/work/ollama-image-studio
scripts/ollama_chat_image_bridge.py --chat-id 019eea1f-0318-7bfa-8081-746c761694cc
```

Keep watching / 持续监听：

```bash
scripts/start_ollama_chat_bridge.sh
```
