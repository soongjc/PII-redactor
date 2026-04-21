from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .schemas import EntityIn


def render_masked(src_image: Path, dst_image: Path, entities: Iterable[EntityIn], padding: int = 2) -> None:
    img = Image.open(src_image).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for e in entities:
        x0 = max(0, e.x - padding)
        y0 = max(0, e.y - padding)
        x1 = min(W, e.x + e.w + padding)
        y1 = min(H, e.y + e.h + padding)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_image, "PNG")


def combine_pdf(masked_pngs: list[Path], dst_pdf: Path) -> None:
    if not masked_pngs:
        raise ValueError("no pages to combine")
    images = [Image.open(p).convert("RGB") for p in masked_pngs]
    first, rest = images[0], images[1:]
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    first.save(dst_pdf, "PDF", save_all=True, append_images=rest)
