from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    vl_model: str = "qwen2.5vl:7b"
    pii_model: str = "qwen3:8b"

    # VL backend: "ollama" (default, Ollama /api/chat) or "llamacpp" (upstream
    # llama-server /v1/chat/completions — use this for models Ollama hasn't
    # wired vision for yet, like dots.ocr).
    vl_backend: str = "ollama"
    llamacpp_host: str = "http://localhost:11500"

    # Prompt preset: "qwen" (our JSON prompt, works with Qwen2.5-VL) or
    # "dotsocr" (rednote dots.ocr's native prompt_layout_all_en).
    vl_prompt_mode: str = "qwen"

    database_url: str = "sqlite:///./storage/pii.db"
    storage_dir: str = "./storage"
    pdf_dpi: int = 150

    # Pre-resize page images so they match Qwen-VL's internal smart_resize output.
    # Total pixels are capped at this; dims rounded to multiples of 28 (Qwen's
    # vision patch/merge). Qwen2.5-VL's upstream default is 1003520 (1280 visual
    # tokens). 802816 = 1024 tokens is a faster/less-VRAM alternative.
    vl_max_pixels: int = 802816

    # Ollama token limits. 0 = don't send the option, let Ollama use its
    # Modelfile default. Sending a num_ctx with images triggers a GGML_ASSERT
    # crash on some Ollama/llama.cpp builds — 0 is the safe default.
    vl_num_ctx: int = 0
    vl_num_predict: int = 0
    pii_num_ctx: int = 0
    pii_num_predict: int = 0

    # HTTP timeout (seconds) for each Ollama request. Cold-loading a VL model
    # on CPU/Metal can take a minute; raise if you see read timeouts.
    ollama_timeout_s: float = 1200.0
    ollama_connect_timeout_s: float = 30.0

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
