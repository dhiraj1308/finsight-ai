import os
import sys
import pytest
from unittest.mock import patch

# Reset settings singleton before each test
@pytest.fixture(autouse=True)
def reset_settings_singleton():
    import config
    config.reset_settings()
    yield
    config.reset_settings()


VALID_ENV = {
    "SQLITE_DB_PATH": "data/test.db",
    "CHROMA_PERSIST_DIR": "data/vector",
    "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
    "LLM_API_KEY": "test_key_123",
}


def test_settings_load_with_all_required_vars():
    with patch.dict(os.environ, VALID_ENV, clear=False):
        from config import get_settings
        settings = get_settings()
        assert settings.SQLITE_DB_PATH == "data/test.db"
        assert settings.CHROMA_PERSIST_DIR == "data/vector"
        assert settings.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
        assert settings.LLM_API_KEY == "test_key_123"


def test_optional_vars_use_defaults_when_absent():
    env = VALID_ENV.copy()
    with patch.dict(os.environ, env, clear=False):
        # Remove optional vars if present
        for key in ["PROPHET_YEARLY_SEASONALITY", "PROPHET_WEEKLY_SEASONALITY", "LOG_LEVEL"]:
            os.environ.pop(key, None)
        from config import get_settings
        settings = get_settings()
        assert settings.PROPHET_YEARLY_SEASONALITY is True
        assert settings.PROPHET_WEEKLY_SEASONALITY is True
        assert settings.LOG_LEVEL == "INFO"


def test_missing_sqlite_db_path_causes_exit():
    env = {k: v for k, v in VALID_ENV.items() if k != "SQLITE_DB_PATH"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit):
            from config import Settings
            Settings()


def test_missing_llm_api_key_causes_exit():
    env = {k: v for k, v in VALID_ENV.items() if k != "LLM_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit):
            from config import Settings
            Settings()


def test_get_settings_returns_singleton():
    with patch.dict(os.environ, VALID_ENV, clear=False):
        from config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


def test_log_level_is_uppercased():
    env = {**VALID_ENV, "LOG_LEVEL": "debug"}
    with patch.dict(os.environ, env, clear=False):
        from config import get_settings
        settings = get_settings()
        assert settings.LOG_LEVEL == "DEBUG"


def test_seasonality_false_when_set_to_false():
    env = {
        **VALID_ENV,
        "PROPHET_YEARLY_SEASONALITY": "false",
        "PROPHET_WEEKLY_SEASONALITY": "false",
    }
    with patch.dict(os.environ, env, clear=False):
        from config import get_settings
        settings = get_settings()
        assert settings.PROPHET_YEARLY_SEASONALITY is False
        assert settings.PROPHET_WEEKLY_SEASONALITY is False