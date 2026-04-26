from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    data_path: Path
    results_dir: Path
    ollama_base_url: str
    ollama_model: str
    request_timeout_seconds: int = 60


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_DATA_PATH = Path("data/locomo10.json")
DEFAULT_RESULTS_DIR = Path("results")


def load_settings() -> Settings:
    return Settings(
        data_path=Path(os.getenv("LOCOMO_DATA_PATH", str(DEFAULT_DATA_PATH))),
        results_dir=Path(os.getenv("RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )
