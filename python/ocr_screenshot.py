from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Optional

import requests
from PIL import Image

log = logging.getLogger("ocr_screenshot")

GROQ_VISION_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
MAX_SIDE_PX = 1600
JPEG_QUALITY = 85
MIN_TEXT_LEN = 25

class OcrError(Exception):
    pass

def _load_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise OcrError(f"Некоректне зображення: {e}") from e
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    return img

def _prepare_jpeg_bytes(image_bytes: bytes) -> tuple[bytes, str]:
    img = _load_image(image_bytes)
    w, h = img.size
    scale = min(1.0, MAX_SIDE_PX / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), "image/jpeg"

def _clean_extracted(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:\w+)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _ocr_via_groq(image_bytes: bytes, api_key: str, model: str) -> str:
    jpeg_bytes, mime = _prepare_jpeg_bytes(image_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    system = (
        "Ти OCR для скріншотів новин. "
        "Витягни ЛИШЕ той текст, який реально видно на зображенні. "
        "Збережи мову оригіналу (українська/англійська/російська). "
        "Ігноруй UI телефона (годинник, батарея, кнопки, рекламу, іконки). "
        "НЕ вигадуй, НЕ доповнюй і НЕ переказуй текст. "
        "Якщо текст нечитабельний або новини немає — поверни рівно: EMPTY"
    )
    user_text = (
        "Зчитай текст новини зі скріншота. Поверни лише суцільний текст "
        "(заголовок + основний текст), без пояснень і без markdown."
    )

    payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 4096,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    r = requests.post(
        GROQ_VISION_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=90,
    )
    data = r.json() if r.content else {}
    if r.status_code >= 400:
        err = data.get("error") if isinstance(data, dict) else data
        msg = err.get("message") if isinstance(err, dict) else str(err or r.text)
        raise OcrError(f"Groq Vision помилка: {msg}")

    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise OcrError("Groq Vision повернув неочікувану відповідь")
    cleaned = _clean_extracted(str(content))
    if cleaned.upper() == "EMPTY" or cleaned.upper().startswith("EMPTY\n"):
        return ""
    return cleaned

def _ocr_via_rapidocr(image_bytes: bytes) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise OcrError(
            "Немає OCR-бекенду. Встановіть: pip install rapidocr-onnxruntime"
        ) from e

    jpeg_bytes, _ = _prepare_jpeg_bytes(image_bytes)
    engine = RapidOCR()
    result, _ = engine(jpeg_bytes)
    if not result:
        return ""
    lines = []
    for item in result:
        if len(item) >= 2 and item[1]:
            lines.append(str(item[1]).strip())
    return _clean_extracted("\n".join(lines))

def extract_text_from_image(
    image_bytes: bytes,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    if not image_bytes:
        raise OcrError("Порожнє зображення")

    key = (api_key or os.environ.get("GROQ_API_KEY") or "").strip()
    vision_model = (model or os.environ.get("GROQ_VISION_MODEL") or DEFAULT_VISION_MODEL).strip()

    last_err: Exception | None = None
    if key:
        try:
            text = _ocr_via_groq(image_bytes, key, vision_model)
            if len(text) >= MIN_TEXT_LEN:
                return text
            log.warning("Groq Vision повернув занадто короткий текст (%s символів)", len(text))
            if text:
                last_err = OcrError("На скріні замало тексту новини")
        except Exception as e:
            log.warning("Groq Vision failed: %s", e)
            last_err = e

    try:
        text = _ocr_via_rapidocr(image_bytes)
        if len(text) >= MIN_TEXT_LEN:
            return text
        raise OcrError(
            "Не вдалося зчитати достатньо тексту зі скріна. "
            "Спробуйте чіткіший скріншот або надішліть текст / посилання."
        )
    except OcrError:
        raise
    except Exception as e:
        if last_err:
            raise OcrError(f"OCR не вдався: {last_err}") from e
        raise OcrError(f"OCR не вдався: {e}") from e

def main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="OCR скріншота новини")
    parser.add_argument("image", nargs="?", help="Шлях до зображення")
    parser.add_argument("--stdin-b64", action="store_true", help="Читати base64 зображення зі stdin")
    parser.add_argument("--json", action="store_true", help="Вивести JSON {text, error}")
    args = parser.parse_args()

    try:
        if args.stdin_b64:
            raw = sys.stdin.buffer.read()
            try:
                b64 = raw.decode("utf-8").strip()
                image_bytes = base64.b64decode(b64, validate=False)
            except Exception:
                image_bytes = raw
        elif args.image:
            with open(args.image, "rb") as f:
                image_bytes = f.read()
        else:
            raise OcrError("Передайте шлях до зображення або --stdin-b64")

        text = extract_text_from_image(image_bytes)
        if args.json:
            print(json.dumps({"text": text}, ensure_ascii=False))
        else:
            print(text)
    except OcrError as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
            sys.exit(1)
        print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
            sys.exit(1)
        print(f"Помилка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
