from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file — system environment variables take precedence
load_dotenv(override=False)


class Settings:
    """
    Central configuration object. Reads from environment variables
    (populated from .env file by load_dotenv above).
    Fails fast on startup if any required variable is missing.
    """

    # Required
    SQLITE_DB_PATH: str
    CHROMA_PERSIST_DIR: str
    EMBEDDING_MODEL_NAME: str
    LLM_API_KEY: str

    # Optional with defaults
    PROPHET_YEARLY_SEASONALITY: bool
    PROPHET_WEEKLY_SEASONALITY: bool
    LOG_LEVEL: str

    REQUIRED_VARS = [
        "SQLITE_DB_PATH",
        "CHROMA_PERSIST_DIR",
        "EMBEDDING_MODEL_NAME",
        "LLM_API_KEY",
    ]

    def __init__(self):
        self._validate_required()
        self._load()

    def _validate_required(self) -> None:
        missing = [v for v in self.REQUIRED_VARS if not os.getenv(v)]
        if missing:
            for var in missing:
                print(
                    f"ERROR: Required environment variable '{var}' is missing. "
                    f"Set it in your .env file or system environment.",
                    file=sys.stderr,
                )
            sys.exit(1)

    def _load(self) -> None:
        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
        self.LLM_API_KEY = os.getenv("LLM_API_KEY")

        self.PROPHET_YEARLY_SEASONALITY = (
            os.getenv("PROPHET_YEARLY_SEASONALITY", "true").lower() == "true"
        )
        self.PROPHET_WEEKLY_SEASONALITY = (
            os.getenv("PROPHET_WEEKLY_SEASONALITY", "true").lower() == "true"
        )
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    def configure_logging(self) -> None:
        """Configure Python logging based on LOG_LEVEL setting."""
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Returns singleton Settings instance.
    Safe to call multiple times — only loads once.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Resets the singleton — used in tests to reload settings
    with different environment variables.
    """
    global _settings
    _settings = None