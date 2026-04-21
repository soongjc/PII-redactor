"""Glue: VL chunk extraction + Qwen3 PII tagging -> entities with bboxes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from . import llm
from .bbox_refine import refine_entities

CHUNK_SEP = " "


def _normalize(s: str) -> str:
    return " ".join(s.split()).lower()


def _find_chunk_span(
    chunks: list[dict[str, Any]], needle: str, skip: set[int]
) -> list[int] | None:
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


def _stitch(chunks: list[dict[str, Any]], pii: list[dict[str, str]]) -> list[dict[str, Any]]:
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


def detect_page_pii(image_path: Path) -> list[dict[str, Any]]:
    """Non-streaming convenience: run full pipeline and return entities."""
    chunks, _ = llm.vl_extract_chunks(image_path)
    if not chunks:
        return []
    full_text = CHUNK_SEP.join(c["text"] for c in chunks)
    pii, _ = llm.pii_tag_text(full_text)
    return _stitch(chunks, pii)


def detect_page_pii_stream(image_path: Path) -> Iterator[dict[str, Any]]:
    """Yield stage events for the detection pipeline.

    Events:
      {"stage": "vl_ocr", "status": "start"}
      {"stage": "vl_ocr", "status": "done", "count": N, "elapsed_ms": ms}
      {"stage": "pii_tag", "status": "start", "text_preview": "..."}
      {"stage": "pii_tag", "status": "done", "count": N, "elapsed_ms": ms}
      {"stage": "done", "entities": [...]}
    """
    yield {"stage": "vl_ocr", "status": "start", "model": "VL (OCR)"}
    try:
        chunks, vl_ms = llm.vl_extract_chunks(image_path)
    except llm.LLMError as e:
        yield {"stage": "error", "where": "vl_ocr", "message": str(e)}
        return
    yield {
        "stage": "vl_ocr",
        "status": "done",
        "count": len(chunks),
        "elapsed_ms": int(vl_ms * 1000),
    }

    if not chunks:
        yield {"stage": "done", "entities": []}
        return

    full_text = CHUNK_SEP.join(c["text"] for c in chunks)
    preview = full_text if len(full_text) <= 240 else full_text[:240] + "…"
    yield {"stage": "pii_tag", "status": "start", "model": "PII (Qwen3)", "text_preview": preview}
    try:
        pii, pii_ms = llm.pii_tag_text(full_text)
    except llm.LLMError as e:
        yield {"stage": "error", "where": "pii_tag", "message": str(e)}
        return
    yield {
        "stage": "pii_tag",
        "status": "done",
        "count": len(pii),
        "elapsed_ms": int(pii_ms * 1000),
    }

    entities = _stitch(chunks, pii)
    entities = refine_entities(image_path, entities)
    yield {"stage": "done", "entities": entities, "matched": len(entities), "pii_found": len(pii)}
