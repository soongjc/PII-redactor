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
