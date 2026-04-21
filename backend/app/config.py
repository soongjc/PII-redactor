from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    vl_model: str = "qwen2.5vl:7b"
    pii_model: str = "qwen3:8b"
    database_url: str = "sqlite:///./storage/pii.db"
    storage_dir: str = "./storage"
    pdf_dpi: int = 150

    # Pre-resize page images to this max edge (longest side, px) before sending to VL.
    # Qwen-VL internally resizes images and returns bboxes in THAT space, so we resize
    # ourselves to make the math deterministic. Smaller also = faster + less VRAM.
    vl_input_max_side: int = 1280

    # Ollama generation knobs. 8192 is a safe ceiling on macOS Metal; larger
    # values trigger a llama.cpp GGML_ASSERT crash with Qwen-VL on some builds.
    # If VL output truncates on dense pages, raise gradually and watch ollama logs.
    vl_num_ctx: int = 8192         # context window (tokens)
    vl_num_predict: int = 4096     # max new tokens
    pii_num_ctx: int = 8192
    pii_num_predict: int = 2048

    # Bbox calibration.
    refine_bboxes: bool = True           # tighten VL bboxes to dark-pixel content
    refine_expand_pct: float = 0.10      # expand VL bbox by this % before tightening
    refine_threshold: int = 180          # pixel value: < threshold is "text"
    mask_padding_px: int = 2             # extra pixels around each bbox when masking

    @property
    def storage_path(self) -> Path:
        p = Path(self.storage_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads_dir(self) -> Path:
        p = self.storage_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def pages_dir(self) -> Path:
        p = self.storage_path / "pages"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def masked_dir(self) -> Path:
        p = self.storage_path / "masked"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
