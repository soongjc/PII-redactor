"""Tighten VL-reported bboxes to the actual text content using PIL."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import settings


def _refine_one(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    expand_pct: float,
    threshold: int,
) -> tuple[int, int, int, int]:
    W, H = img.size
    pad_x = int(round(w * expand_pct))
    pad_y = int(round(h * expand_pct))

    ex = max(0, x - pad_x)
    ey = max(0, y - pad_y)
    erx = min(W, x + w + pad_x)
    ery = min(H, y + h + pad_y)
    if erx <= ex or ery <= ey:
        return x, y, w, h

    crop = img.crop((ex, ey, erx, ery)).convert("L")
    # Any pixel darker than `threshold` is treated as "text". Map to white on
    # black so getbbox() returns the tight bounding box of dark content.
    bw = crop.point(lambda p: 255 if p < threshold else 0)
    bbox = bw.getbbox()
    if bbox is None:
        return x, y, w, h  # no text detected — keep original

    left, upper, right, lower = bbox
    new_x = ex + left
    new_y = ey + upper
    new_w = right - left
    new_h = lower - upper
    if new_w <= 0 or new_h <= 0:
        return x, y, w, h
    return new_x, new_y, new_w, new_h


def refine_entities(
    image_path: Path, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not entities or not settings.refine_bboxes:
        return entities
    with Image.open(image_path) as im:
        im.load()
        out: list[dict[str, Any]] = []
        for e in entities:
            try:
                nx, ny, nw, nh = _refine_one(
                    im,
                    int(e["x"]),
                    int(e["y"]),
                    int(e["w"]),
                    int(e["h"]),
                    expand_pct=settings.refine_expand_pct,
                    threshold=settings.refine_threshold,
                )
            except Exception:
                out.append(e)
                continue
            out.append({**e, "x": nx, "y": ny, "w": nw, "h": nh})
    return out
