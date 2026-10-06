from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from random import Random
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
DEFAULT_IMAGE_MODEL = os.environ.get("IMAGE_MODEL_ID", "local-cell-renderer")

app = FastAPI(title="Ollama Image Studio / Ollama 本地生图工作台")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7861", "http://localhost:7861"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

_pipeline_cache: dict[str, Any] = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    image_model: str = DEFAULT_IMAGE_MODEL
    use_ollama: bool = True
    style: str = "scientific"
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    steps: int = Field(6, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=15.0)
    seed: int | None = None


class GenerateResponse(BaseModel):
    ok: bool
    image_url: str
    image_path: str
    seed: int
    prompt_original: str
    prompt_used: str
    negative_prompt: str
    zh_summary: str
    en_summary: str
    ollama_model: str
    image_model: str
    backend: dict[str, Any]
    elapsed_seconds: float


def _bilingual_error(zh: str, en: str, status_code: int = 500) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"zh": zh, "en": en})


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        texts: list[str] = []
        for key in ("response", "content", "thinking"):
            if key in value:
                texts.append(_extract_text(value[key]))
        if "message" in value:
            texts.append(_extract_text(value["message"]))
        return "\n".join(t for t in texts if t.strip())
    if isinstance(value, list):
        return "\n".join(_extract_text(item) for item in value)
    return ""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def list_ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return []
    return [item.get("model") or item.get("name") for item in payload.get("models", []) if item.get("model") or item.get("name")]


def expand_prompt_with_ollama(prompt: str, model: str, style: str) -> dict[str, str]:
    system = (
        "You are a bilingual Chinese-English image prompt engineer for a local image-generation backend. "
        "Return only valid JSON with keys english_prompt, negative_prompt, zh_summary, en_summary. "
        "Do not claim that you generated the image. The local backend will generate pixels after you return JSON."
    )
    user = (
        f"User request / 用户需求: {prompt}\n"
        f"Style / 风格: {style}\n"
        "Create a concise but detailed English image prompt. Preserve scientific accuracy for biology requests."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.35},
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        raw = response.json()
    except requests.RequestException as exc:
        raise _bilingual_error(
            "Ollama 本地接口不可用，请确认 Ollama 正在运行。",
            f"Ollama local API is unavailable. Confirm Ollama is running. Details: {exc}",
            503,
        ) from exc

    text = _extract_text(raw)
    parsed = _extract_json(text) or {}
    english_prompt = str(parsed.get("english_prompt") or "").strip()
    if not english_prompt:
        english_prompt = fallback_prompt(prompt, style)
    return {
        "english_prompt": english_prompt,
        "negative_prompt": str(parsed.get("negative_prompt") or default_negative_prompt()).strip(),
        "zh_summary": str(parsed.get("zh_summary") or f"已将需求转换为本地图像提示词：{prompt}").strip(),
        "en_summary": str(parsed.get("en_summary") or f"Converted the request into a local image prompt: {prompt}").strip(),
    }


def default_negative_prompt() -> str:
    return (
        "text, watermark, logo, blurry, low quality, distorted anatomy, extra labels, "
        "fake user interface, caption, cropped subject"
    )


def fallback_prompt(prompt: str, style: str) -> str:
    lower = prompt.lower()
    if "细胞" in prompt or "cell" in lower:
        return (
            "high quality scientific educational illustration of a biological animal cell, "
            "clear cross-section, visible nucleus, mitochondria, golgi apparatus, endoplasmic reticulum, ribosomes, "
            "translucent cytoplasm, professional medical visualization, clean background, vivid but accurate colors"
        )
    return f"high quality {style} image, detailed composition, clear subject, clean lighting, {prompt}"


def get_diffusion_pipeline(model_id: str) -> tuple[Any, dict[str, Any]]:
    if model_id in _pipeline_cache:
        return _pipeline_cache[model_id]["pipe"], _pipeline_cache[model_id]["meta"]

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except Exception as exc:
        raise _bilingual_error(
            "本地图像生成依赖还没有安装，请先运行 scripts/setup.sh。",
            f"Local image-generation dependencies are not installed. Run scripts/setup.sh first. Details: {exc}",
            503,
        ) from exc

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    except TypeError:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as exc:
        raise _bilingual_error(
            f"无法加载本地图像模型：{model_id}。请检查网络或模型名称。",
            f"Could not load local image model: {model_id}. Check network access or the model name. Details: {exc}",
            503,
        ) from exc

    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    meta = {"device": device, "dtype": str(dtype).replace("torch.", ""), "model_id": model_id}
    _pipeline_cache[model_id] = {"pipe": pipe, "meta": meta}
    return pipe, meta


def render_image(
    prompt: str,
    negative_prompt: str,
    model_id: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    if model_id in {"local-cell-renderer", "pillow-cell-renderer"}:
        return render_cell_with_pillow(prompt, negative_prompt, seed, width, height)

    try:
        import torch
    except Exception as exc:
        raise _bilingual_error(
            "PyTorch 未安装，无法进行本地生图。",
            f"PyTorch is not installed, so local image generation cannot run. Details: {exc}",
            503,
        ) from exc

    pipe, meta = get_diffusion_pipeline(model_id)
    generator_device = "cpu" if meta["device"] == "mps" else meta["device"]
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    except RuntimeError as exc:
        if meta["device"] == "mps" and "out of memory" in str(exc).lower():
            raise _bilingual_error(
                "Apple GPU 显存不足。请把尺寸降到 512x512 或减少步数。",
                "Apple GPU memory is insufficient. Reduce size to 512x512 or lower the step count.",
                507,
            ) from exc
        raise

    image = result.images[0]
    filename = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
    path = OUTPUT_DIR / filename
    image.save(path)
    return path, meta


def render_cell_with_pillow(
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
) -> tuple[Path, dict[str, Any]]:
    from PIL import Image, ImageDraw, ImageFilter

    rng = Random(seed)
    size = max(width, height)
    image = Image.new("RGB", (width, height), (12, 16, 18))
    draw = ImageDraw.Draw(image, "RGBA")

    for radius, alpha in ((0.88, 30), (0.7, 24), (0.5, 18)):
        inset_x = int(width * (1 - radius) / 2)
        inset_y = int(height * (1 - radius) / 2)
        draw.ellipse((inset_x, inset_y, width - inset_x, height - inset_y), fill=(39, 196, 141, alpha))

    cell_box = (int(width * 0.08), int(height * 0.13), int(width * 0.92), int(height * 0.87))
    draw.ellipse(cell_box, fill=(64, 168, 198, 112), outline=(141, 237, 221, 230), width=max(3, size // 150))
    draw.ellipse(
        (cell_box[0] + 10, cell_box[1] + 10, cell_box[2] - 10, cell_box[3] - 10),
        outline=(225, 255, 250, 92),
        width=max(1, size // 260),
    )

    nucleus_box = (int(width * 0.33), int(height * 0.28), int(width * 0.68), int(height * 0.64))
    draw.ellipse(nucleus_box, fill=(125, 84, 201, 222), outline=(226, 207, 255, 240), width=max(3, size // 170))
    draw.ellipse(
        (int(width * 0.45), int(height * 0.39), int(width * 0.56), int(height * 0.51)),
        fill=(214, 127, 218, 225),
        outline=(251, 220, 255, 220),
        width=max(2, size // 250),
    )

    def draw_mitochondrion(cx: float, cy: float, angle_shift: int = 0) -> None:
        rx = int(width * rng.uniform(0.045, 0.065))
        ry = int(height * rng.uniform(0.022, 0.032))
        box = (int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry))
        draw.ellipse(box, fill=(242, 132, 64, 230), outline=(255, 210, 129, 235), width=max(2, size // 260))
        for i in range(4):
            x0 = box[0] + int((i + 0.9) * (box[2] - box[0]) / 5)
            draw.arc(
                (x0 - rx // 3, box[1] + 4, x0 + rx // 3, box[3] - 4),
                90 + angle_shift,
                270 + angle_shift,
                fill=(255, 237, 166, 210),
                width=max(1, size // 330),
            )

    for cx, cy in (
        (width * 0.24, height * 0.30),
        (width * 0.74, height * 0.34),
        (width * 0.28, height * 0.68),
        (width * 0.72, height * 0.69),
        (width * 0.50, height * 0.76),
    ):
        draw_mitochondrion(cx + rng.uniform(-10, 10), cy + rng.uniform(-10, 10), rng.randint(-25, 25))

    for offset in range(-3, 4):
        points: list[tuple[float, float]] = []
        for step in range(90):
            t = step / 89
            x = width * (0.18 + 0.64 * t)
            y = height * (0.52 + 0.13 * offset / 4) + 22 * math.sin(t * 9 + offset)
            y += rng.uniform(-3, 3)
            if nucleus_box[0] < x < nucleus_box[2] and nucleus_box[1] < y < nucleus_box[3]:
                if len(points) > 1:
                    draw.line(points, fill=(76, 217, 134, 210), width=max(2, size // 180))
                points = []
            else:
                points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=(76, 217, 134, 210), width=max(2, size // 180))

    golgi_x = int(width * 0.66)
    golgi_y = int(height * 0.50)
    for i in range(6):
        y = golgi_y + i * int(height * 0.025)
        draw.arc(
            (golgi_x, y, golgi_x + int(width * 0.16), y + int(height * 0.055)),
            190,
            350,
            fill=(91, 156, 255, 230),
            width=max(4, size // 115),
        )

    for _ in range(165):
        x = rng.randint(cell_box[0] + 24, cell_box[2] - 24)
        y = rng.randint(cell_box[1] + 24, cell_box[3] - 24)
        if nucleus_box[0] < x < nucleus_box[2] and nucleus_box[1] < y < nucleus_box[3]:
            continue
        r = rng.choice([1, 1, 2])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(237, 246, 255, rng.randint(130, 220)))

    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=3))
    filename = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
    path = OUTPUT_DIR / filename
    image.save(path)
    return path, {
        "device": "local-cpu",
        "dtype": "pillow-rgba",
        "model_id": "local-cell-renderer",
        "renderer": "Ollama prompt plan + deterministic Pillow cell renderer",
        "prompt_seen": prompt,
        "negative_prompt": negative_prompt,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    models = list_ollama_models()
    return {
        "ok": True,
        "zh": "本地服务正在运行。",
        "en": "Local service is running.",
        "ollama_url": OLLAMA_URL,
        "ollama_available": bool(models),
        "ollama_models": models,
        "default_ollama_model": DEFAULT_OLLAMA_MODEL,
        "default_image_model": DEFAULT_IMAGE_MODEL,
        "outputs_dir": str(OUTPUT_DIR),
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    started = time.time()
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % 2_147_483_647

    if req.use_ollama:
        prompt_pack = expand_prompt_with_ollama(req.prompt, req.ollama_model, req.style)
    else:
        prompt_pack = {
            "english_prompt": fallback_prompt(req.prompt, req.style),
            "negative_prompt": default_negative_prompt(),
            "zh_summary": "未使用 Ollama，已用本地模板生成提示词。",
            "en_summary": "Ollama was skipped; a local template prompt was used.",
        }

    image_path, backend = render_image(
        prompt=prompt_pack["english_prompt"],
        negative_prompt=prompt_pack["negative_prompt"],
        model_id=req.image_model,
        width=req.width,
        height=req.height,
        steps=req.steps,
        guidance_scale=req.guidance_scale,
        seed=seed,
    )

    metadata = {
        "prompt_original": req.prompt,
        "prompt_used": prompt_pack["english_prompt"],
        "negative_prompt": prompt_pack["negative_prompt"],
        "seed": seed,
        "ollama_model": req.ollama_model,
        "image_model": req.image_model,
        "backend": backend,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    image_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return GenerateResponse(
        ok=True,
        image_url=f"/images/{image_path.name}",
        image_path=str(image_path),
        seed=seed,
        prompt_original=req.prompt,
        prompt_used=prompt_pack["english_prompt"],
        negative_prompt=prompt_pack["negative_prompt"],
        zh_summary=prompt_pack["zh_summary"],
        en_summary=prompt_pack["en_summary"],
        ollama_model=req.ollama_model,
        image_model=req.image_model,
        backend=backend,
        elapsed_seconds=round(time.time() - started, 2),
    )
