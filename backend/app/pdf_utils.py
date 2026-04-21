from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

from .config import settings


def rasterize_pdf(pdf_path: Path, out_dir: Path) -> list[tuple[int, int, int]]:
    """Convert each page to PNG. Returns [(page_number, width, height), ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=settings.pdf_dpi)
    info: list[tuple[int, int, int]] = []
    for idx, img in enumerate(images, start=1):
        img = img.convert("RGB")
        path = out_dir / f"page_{idx}.png"
        img.save(path, "PNG")
        info.append((idx, img.width, img.height))
    return info


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
