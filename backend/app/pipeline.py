"""Glue: VL chunk extraction + Qwen3 PII tagging -> entities with bboxes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import llm

CHUNK_SEP = " "


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _find_chunk_span(
    chunks: list[dict[str, Any]], needle: str, skip: set[int]
) -> list[int] | None:
    """Find consecutive chunk indices whose joined text contains `needle` (normalized).

    Skips any starting position whose chunk is already claimed.
    """
    target = _normalize(needle)
    if not target:
        return None
    n = len(chunks)
    for start in range(n):
        if start in skip:
            continue
        joined_parts: list[str] = []
        for end in range(start, n):
            if end in skip:
                break
            joined_parts.append(chunks[end]["text"])
            joined = _normalize(CHUNK_SEP.join(joined_parts))
            if target in joined:
                return list(range(start, end + 1))
            if len(joined) > len(target) + 64:
                break
    return None


def _union_bbox(chunks: list[dict[str, Any]], indices: list[int]) -> tuple[int, int, int, int]:
    xs1, ys1, xs2, ys2 = [], [], [], []
    for i in indices:
        x, y, w, h = chunks[i]["bbox"]
        xs1.append(x)
        ys1.append(y)
        xs2.append(x + w)
        ys2.append(y + h)
    x1, y1 = min(xs1), min(ys1)
    x2, y2 = max(xs2), max(ys2)
    return x1, y1, x2 - x1, y2 - y1


def detect_page_pii(image_path: Path) -> list[dict[str, Any]]:
    """Run the full VL + PII pipeline for a single page image.

    Returns: [{type, text, x, y, w, h}, ...]
    """
    chunks = llm.vl_extract_chunks(image_path)
    if not chunks:
        return []

    full_text = CHUNK_SEP.join(c["text"] for c in chunks)
    pii = llm.pii_tag_text(full_text)

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for ent in pii:
        span = _find_chunk_span(chunks, ent["text"], used)
        if not span:
            continue
        x, y, w, h = _union_bbox(chunks, span)
        out.append({"type": ent["type"], "text": ent["text"], "x": x, "y": y, "w": w, "h": h})
        used.update(span)
    return out
