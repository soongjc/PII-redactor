"""Ollama-backed LLM clients for vision OCR (Qwen2.5-VL) and PII tagging (Qwen3)."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ._logging import get_logger
from .config import settings

logger = get_logger("pii.llm")

_CHAT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


PII_TYPES = [
    "NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SSN_NRIC",
    "DOB",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "IP",
]


VL_PROMPT = """Extract every visible text region from the image.

Return ONLY a JSON object with this exact shape:
{"regions": [{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}, ...]}

Rules:
- bbox_2d uses pixel coordinates of the input image (NOT normalized): x1,y1 is the top-left corner, x2,y2 is the bottom-right corner.
- Group text into the smallest semantically meaningful units (a name, a phone number, an email, a single line of an address, a date). Do not merge unrelated lines.
- Preserve original casing, punctuation, and spacing inside the chunk.
- If the image has no text, return {"regions": []}.
- Do NOT include any prose, markdown, code fences, or explanations. JSON only.
"""


# /no_think disables Qwen3's internal chain-of-thought which otherwise adds a lot of latency.
PII_PROMPT_TEMPLATE = """/no_think
You are a PII detection engine. Identify Personally Identifiable Information in the input text.

Allowed PII types (use these exact strings):
NAME, EMAIL, PHONE, ADDRESS, SSN_NRIC, DOB, CREDIT_CARD, BANK_ACCOUNT, IP

Return ONLY a JSON object with this exact shape:
{{"entities": [{{"type": "TYPE", "text": "exact substring from input"}}, ...]}}

Rules:
- "text" MUST be an exact substring copied verbatim from the input (same casing/punctuation/whitespace).
- Do NOT infer or invent values that are not present.
- If nothing is found, return {{"entities": []}}.
- No prose, no markdown, no code fences. JSON only.

Input text:
---
{text}
---
"""


def _strip_to_json(raw: str) -> str:
    """Best-effort: strip code fences / surrounding text and return the JSON object substring."""
    s = raw.strip()
    # Drop Qwen3 <think>...</think> blocks if they somehow sneak through.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    return s


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_strip_to_json(raw))


def _preview(s: str, n: int = 400) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + f"... (+{len(s) - n} chars)"


class LLMError(RuntimeError):
    """Raised when the upstream LLM call fails in a way we can display."""


def _post_chat(payload: dict[str, Any], *, label: str) -> tuple[dict[str, Any], float]:
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    model = payload.get("model")
    msg = f"[{label}] -> {url} model={model}"
    logger.info(msg)
    print(msg, flush=True)
    t0 = time.time()
    try:
        with httpx.Client(timeout=_CHAT_TIMEOUT) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError as e:
        raise LLMError(f"Cannot reach Ollama at {url}: {e}") from e
    except httpx.TimeoutException as e:
        raise LLMError(f"Ollama request timed out: {e}") from e

    elapsed = time.time() - t0
    if resp.status_code >= 400:
        body = resp.text[:500]
        err_msg = f"[{label}] Ollama returned {resp.status_code}: {body}"
        logger.warning(err_msg)
        print(err_msg, flush=True)
        raise LLMError(
            f"Ollama {resp.status_code} from model '{model}'. "
            f"If you set a non-vision model as VL_MODEL, switch to a multimodal tag "
            f"like qwen2.5vl:7b. Body: {body}"
        )

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    msg = f"[{label}] <- {elapsed:.2f}s | response: {_preview(content)}"
    logger.info(msg)
    print(msg, flush=True)
    return data, elapsed


def _coerce_bbox(
    bbox: Any, *, fmt: str, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Convert `bbox` to (x, y, w, h) pixel ints.

    fmt: "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h). Normalized 0-1000 values
    (Qwen-VL quirk) are rescaled to pixels when detected.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        a, b, c, d = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    mx = max(a, b, c, d)
    if mx <= 1000 and mx > max(img_w, img_h) * 1.1:
        a = a / 1000.0 * img_w
        b = b / 1000.0 * img_h
        c = c / 1000.0 * img_w
        d = d / 1000.0 * img_h

    if fmt == "xyxy":
        x, y, w, h = a, b, c - a, d - b
    else:  # xywh
        x, y, w, h = a, b, c, d

    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if w <= 0 or h <= 0:
        return None
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def _extract_regions_list(parsed: Any) -> list[dict[str, Any]]:
    """Find the list of regions from various shapes Qwen-VL returns."""
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if not isinstance(parsed, dict):
        return []
    for key in ("regions", "chunks", "text_regions", "results", "data", "items"):
        v = parsed.get(key)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    # Fallback: the dict itself might be one region, or values may contain it.
    if "bbox_2d" in parsed or "bbox" in parsed or "box" in parsed:
        return [parsed]
    return []


def _preresize_for_vl(image_path: Path, max_side: int) -> tuple[bytes, int, int, int, int]:
    """Resize the page image so longest side == max_side. Returns (png_bytes, orig_w, orig_h, new_w, new_h).

    Qwen-VL resizes images internally ("smart_resize") and returns bboxes in
    THAT resized space — not the original. By resizing ourselves to a known
    size we avoid having to replicate the model-specific preprocessor.
    """
    import io
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        longest = max(orig_w, orig_h)
        if longest <= max_side:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue(), orig_w, orig_h, orig_w, orig_h
        scale = max_side / longest
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = im.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue(), orig_w, orig_h, new_w, new_h


def vl_extract_chunks(image_path: Path) -> tuple[list[dict[str, Any]], float]:
    """Call Qwen2.5-VL. Returns (chunks, elapsed_seconds). Each chunk is {text, bbox:[x,y,w,h]}
    in the ORIGINAL image's pixel coordinates."""
    png_bytes, orig_w, orig_h, sent_w, sent_h = _preresize_for_vl(
        image_path, max_side=settings.vl_input_max_side
    )
    scale_x = orig_w / sent_w
    scale_y = orig_h / sent_h
    logger.info(
        "[VL] sending %dx%d (from %dx%d); bbox scale x=%.3f y=%.3f",
        sent_w, sent_h, orig_w, orig_h, scale_x, scale_y,
    )

    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = {
        "model": settings.vl_model,
        "messages": [
            {"role": "user", "content": VL_PROMPT, "images": [b64]},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": settings.vl_num_ctx,
            "num_predict": settings.vl_num_predict,
        },
    }
    data, elapsed = _post_chat(payload, label="VL")
    content = data.get("message", {}).get("content", "")
    try:
        parsed = _safe_json_loads(content)
    except json.JSONDecodeError as e:
        logger.warning("[VL] JSON parse failed: %s | raw: %s", e, _preview(content, 2000))
        return [], elapsed

    # bbox_2d / bbox / box are interpreted in the coord space of the image we SENT.
    img_w, img_h = sent_w, sent_h

    regions = _extract_regions_list(parsed)
    out: list[dict[str, Any]] = []
    for r in regions:
        text = str(r.get("text_content") or r.get("text") or r.get("content") or "").strip()
        if not text:
            continue
        if "bbox_2d" in r:
            bbox, fmt = r["bbox_2d"], "xyxy"
        elif "bbox" in r:
            bbox, fmt = r["bbox"], "xywh"
        elif "box" in r:
            bbox, fmt = r["box"], "xywh"
        else:
            continue
        coerced = _coerce_bbox(bbox, fmt=fmt, img_w=img_w, img_h=img_h)
        if coerced is None:
            continue
        sx, sy, sw, sh = coerced
        # Scale from sent-image coords back to original-image coords.
        ox = int(round(sx * scale_x))
        oy = int(round(sy * scale_y))
        ow = max(1, int(round(sw * scale_x)))
        oh = max(1, int(round(sh * scale_y)))
        # Clip to original image bounds.
        ox = max(0, min(ox, orig_w))
        oy = max(0, min(oy, orig_h))
        ow = max(1, min(ow, orig_w - ox))
        oh = max(1, min(oh, orig_h - oy))
        out.append({"text": text, "bbox": [ox, oy, ow, oh]})

    if not out:
        logger.warning(
            "[VL] parsed 0 regions from response. sent=%dx%d orig=%dx%d | raw: %s",
            sent_w, sent_h, orig_w, orig_h, _preview(content, 2000),
        )
    else:
        logger.info(
            "[VL] parsed %d chunks (sent %dx%d, original %dx%d)",
            len(out), sent_w, sent_h, orig_w, orig_h,
        )
    return out, elapsed


def pii_tag_text(text: str) -> tuple[list[dict[str, str]], float]:
    """Call Qwen3. Returns (entities, elapsed_seconds)."""
    if not text.strip():
        return [], 0.0
    payload = {
        "model": settings.pii_model,
        "messages": [
            {"role": "user", "content": PII_PROMPT_TEMPLATE.format(text=text)},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": settings.pii_num_ctx,
            "num_predict": settings.pii_num_predict,
        },
    }
    data, elapsed = _post_chat(payload, label="PII")
    content = data.get("message", {}).get("content", "")
    try:
        parsed = _safe_json_loads(content)
    except json.JSONDecodeError as e:
        logger.warning("[PII] JSON parse failed: %s", e)
        return [], elapsed
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
    out: list[dict[str, str]] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type", "")).strip().upper()
        etext = str(e.get("text", "")).strip()
        if not etype or not etext or etype not in PII_TYPES:
            continue
        out.append({"type": etype, "text": etext})
    logger.info("[PII] parsed %d entities", len(out))
    return out, elapsed
