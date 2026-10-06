#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = Path.home() / "Library" / "Application Support" / "Ollama" / "db.sqlite"
DEFAULT_STATE = ROOT / "app" / "outputs" / "ollama-chat-image-bridge-state.json"
DEFAULT_IMAGE_API = "http://127.0.0.1:7861/api/generate"
MARKER = "[Ollama Image Bridge]"


def now_sql() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f%z")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_message_ids": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed_message_ids": []}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_action_prompt(content: str) -> str | None:
    if "dalle.text2im" not in content and "text2im" not in content:
        return None
    for obj in iter_json_objects(content):
        action = str(obj.get("action") or "")
        if "text2im" not in action:
            continue
        action_input = obj.get("action_input")
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except json.JSONDecodeError:
                action_input = {"prompt": action_input}
        if isinstance(action_input, dict):
            prompt = str(action_input.get("prompt") or "").strip()
            if prompt:
                return prompt
    return None


def generate_image(prompt: str, api_url: str) -> tuple[Path, dict[str, Any]]:
    payload = {
        "prompt": prompt,
        "ollama_model": os.environ.get("OLLAMA_MODEL", "gemma4:31b"),
        "image_model": os.environ.get("IMAGE_MODEL_ID", "local-cell-renderer"),
        "use_ollama": False,
        "style": "scientific",
        "width": 512,
        "height": 512,
        "steps": 6,
    }
    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    path = Path(data["image_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path, data


def insert_image_attachment(
    db_path: Path,
    chat_id: str,
    source_message_id: int,
    prompt: str,
    image_path: Path,
    generation: dict[str, Any],
) -> int:
    image_bytes = image_path.read_bytes()
    created_at = now_sql()
    content = (
        f"{MARKER}\n"
        f"已生成图片 / Generated image\n"
        f"来源消息 / Source message: {source_message_id}\n"
        f"提示词 / Prompt: {prompt}\n"
        f"文件 / File: {image_path.name}"
    )
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        cursor = conn.execute(
            """
            INSERT INTO messages (chat_id, role, content, thinking, stream, model_name, created_at, updated_at)
            VALUES (?, 'user', ?, '', 0, ?, ?, ?)
            """,
            (chat_id, content, "ollama-image-bridge", created_at, created_at),
        )
        message_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO attachments (message_id, filename, data) VALUES (?, ?, ?)",
            (message_id, image_path.name, sqlite3.Binary(image_bytes)),
        )
        metadata = {
            "source_message_id": source_message_id,
            "generation": generation,
            "image_path": str(image_path),
        }
        conn.execute(
            """
            INSERT INTO tool_calls (message_id, type, function_name, function_arguments, function_result)
            VALUES (?, 'function', 'ollama_image_bridge', ?, ?)
            """,
            (
                message_id,
                json.dumps({"prompt": prompt}, ensure_ascii=False),
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
        conn.commit()
    return message_id


def candidate_messages(db_path: Path, chat_id: str | None = None) -> list[tuple[int, str, str]]:
    query = """
        SELECT id, chat_id, content
        FROM messages
        WHERE role = 'assistant'
          AND content LIKE '%text2im%'
        ORDER BY id ASC
    """
    params: tuple[Any, ...] = ()
    if chat_id:
        query = """
            SELECT id, chat_id, content
            FROM messages
            WHERE role = 'assistant'
              AND chat_id = ?
              AND content LIKE '%text2im%'
            ORDER BY id ASC
        """
        params = (chat_id,)
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        return [(int(row[0]), str(row[1]), str(row[2])) for row in conn.execute(query, params)]


def process_once(db_path: Path, state_path: Path, api_url: str, chat_id: str | None = None) -> int:
    state = load_state(state_path)
    processed = {int(item) for item in state.get("processed_message_ids", [])}
    created = 0
    for message_id, row_chat_id, content in candidate_messages(db_path, chat_id):
        if message_id in processed:
            continue
        prompt = parse_action_prompt(content)
        if not prompt:
            processed.add(message_id)
            continue
        image_path, generation = generate_image(prompt, api_url)
        inserted_id = insert_image_attachment(db_path, row_chat_id, message_id, prompt, image_path, generation)
        processed.add(message_id)
        created += 1
        print(
            f"Inserted generated image into chat {row_chat_id}: source={message_id} image_message={inserted_id} file={image_path}",
            flush=True,
        )
    state["processed_message_ids"] = sorted(processed)
    state["updated_at"] = datetime.now().astimezone().isoformat()
    save_state(state_path, state)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Ollama fake text2im tool calls as image attachments.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--api-url", default=DEFAULT_IMAGE_API)
    parser.add_argument("--chat-id", default=os.environ.get("OLLAMA_CHAT_ID"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=4.0)
    args = parser.parse_args()

    if args.watch:
        print(f"Watching Ollama DB / 正在监听 Ollama 数据库: {args.db}", flush=True)
        while True:
            try:
                process_once(args.db, args.state, args.api_url, args.chat_id)
            except Exception as exc:
                print(f"Bridge error / 桥接错误: {exc}", flush=True)
            time.sleep(args.interval)
    else:
        created = process_once(args.db, args.state, args.api_url, args.chat_id)
        print(f"Created {created} generated image message(s). / 已创建 {created} 条图片消息。")


if __name__ == "__main__":
    main()
