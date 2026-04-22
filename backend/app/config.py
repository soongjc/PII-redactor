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

    # Ollama's `format: "json"` forces structured JSON output. Great when
    # supported, but breaks on some brand-new vision models (e.g. qwen3-vl on
    # certain Ollama builds — you'll see "stream>" appear with no tokens).
    # Set to false to disable; we still strip/parse JSON on our side.
    vl_format_json: bool = True
    pii_format_json: bool = True

    # Disable Qwen3-family "thinking" mode. Qwen3 / qwen3-vl route reasoning
    # into a separate `thinking` field; if the model stays in reasoning for too
    # long it can emit zero content. Setting `think: false` in Ollama options
    # makes the model answer directly. Safe on non-qwen3 models (ignored).
    vl_disable_think: bool = True
    pii_disable_think: bool = True

    database_url: str = "sqlite:///./storage/pii.db"
    storage_dir: str = "./storage"
    pdf_dpi: int = 150

    # Pre-resize page images so they match Qwen-VL's internal smart_resize output.
    # Total pixels are capped at this; dims rounded to multiples of 28 (Qwen's
    # vision patch/merge). Qwen2.5-VL's upstream default is 1003520 (1280 visual
    # tokens). 802816 = 1024 tokens is a faster/less-VRAM alternative.
    vl_max_pixels: int = 802816

    # Ollama token limits. num_ctx=0 means "don't send", let Ollama use its
    # Modelfile default — this dodges a GGML_ASSERT crash on macOS Metal when
    # a custom num_ctx is sent with images. num_predict caps generated tokens;
    # too low causes mid-JSON truncation. We send num_predict explicitly with
    # a generous ceiling so dense pages don't get cut off.
    vl_num_ctx: int = 0
    vl_num_predict: int = 8192
    pii_num_ctx: int = 0
    pii_num_predict: int = 4096

    # HTTP timeout (seconds) for each Ollama request. Cold-loading a VL model
    # on CPU/Metal can take a minute; raise if you see read timeouts.
    ollama_timeout_s: float = 1200.0
    ollama_connect_timeout_s: float = 30.0

    # Bbox calibration.
    refine_bboxes: bool = True           # tighten VL bboxes to dark-pixel content
    # Expand bbox by this fraction BEFORE tightening. Split by axis: vertical
    # expansion is dangerous on tightly-packed documents (grabs neighbouring
    # lines, making boxes look too tall), so we only expand horizontally by
    # default. Raise vertical if boxes clip ascenders/descenders.
    refine_expand_x_pct: float = 0.10
    refine_expand_y_pct: float = 0.0
    refine_threshold: int = 180          # pixel value: < threshold is "text"
    mask_padding_px: int = 2             # extra pixels around each bbox when masking

    # Drop VL regions that look like pure money values (e.g. "233.50", "RM 1,200",
    # "$100.00"). They aren't PII and just waste tokens downstream.
    vl_filter_money: bool = True

    # Drop VL regions that look like pure dates (e.g. "2026-04-22", "Apr 22, 2026").
    # NOTE: this also removes DOB detection. Set to false if processing IDs,
    # medical records, or anywhere date-of-birth matters.
    vl_filter_dates: bool = True

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
