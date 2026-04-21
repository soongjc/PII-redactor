"""Ollama-backed LLM clients for vision OCR (Qwen2.5-VL) and PII tagging (Qwen3)."""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx

from .config import settings

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


VL_PROMPT = """You are an OCR engine. Extract every visible text region from the image.

Return ONLY a JSON object with this exact shape:
{"chunks": [{"text": "...", "bbox": [x, y, w, h]}, ...]}

Rules:
- bbox uses pixel coordinates of the input image: x, y are top-left; w, h are width/height.
- Group text into the smallest semantically meaningful units (a name, a phone number, an email, a single line of an address, a date). Do not merge unrelated lines.
- Preserve original casing, spacing inside the chunk, punctuation.
- If the image has no text, return {"chunks": []}.
- Do NOT include any prose, markdown, code fences, or explanations. JSON only.
"""


PII_PROMPT_TEMPLATE = """You are a PII detection engine. Identify Personally Identifiable Information in the input text.

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
    # Strip ```json ... ``` fences if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Find first '{' and matching last '}' (greedy).
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


def _post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{settings.ollama_host.rstrip('/')}/api/chat"
    with httpx.Client(timeout=_CHAT_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def vl_extract_chunks(image_path: Path) -> list[dict[str, Any]]:
    """Call Qwen2.5-VL to extract {text, bbox} chunks from a page image."""
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": settings.vl_model,
        "messages": [
            {
                "role": "user",
                "content": VL_PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    data = _post_chat(payload)
    content = data.get("message", {}).get("content", "")
    parsed = _safe_json_loads(content)
    chunks = parsed.get("chunks", []) if isinstance(parsed, dict) else []
    out: list[dict[str, Any]] = []
    for c in chunks:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        bbox = c.get("bbox") or c.get("box")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, w, h = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append({"text": text, "bbox": [x, y, w, h]})
    return out


def pii_tag_text(text: str) -> list[dict[str, str]]:
    """Call Qwen3 to identify PII spans inside `text`. Returns [{type, text}]."""
    if not text.strip():
        return []
    payload = {
        "model": settings.pii_model,
        "messages": [
            {"role": "user", "content": PII_PROMPT_TEMPLATE.format(text=text)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }
    data = _post_chat(payload)
    content = data.get("message", {}).get("content", "")
    parsed = _safe_json_loads(content)
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
    return out
