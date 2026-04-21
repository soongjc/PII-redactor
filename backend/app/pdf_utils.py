from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from .config import settings


def rasterize_pdf(pdf_path: Path, out_dir: Path) -> list[tuple[int, int, int]]:
    """Convert each page to PNG. Returns [(page_number, width, height), ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = settings.pdf_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    info: list[tuple[int, int, int]] = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            path = out_dir / f"page_{idx}.png"
            pix.save(path.as_posix())
            info.append((idx, pix.width, pix.height))
    return info


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
