"""Tests for app/config.py (dotenv loading and gateway settings)."""
import os

import pytest

from app.config import GatewaySettings, Settings


def test_dotenv_is_loaded_but_real_env_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_PRIMARY=from-dotenv/primary\n"
        "MODEL_SMALL=from-dotenv/small\n"
        "ANTHROPIC_API_KEY=sk-test-from-dotenv\n"
        "LLM_MAX_RETRIES=4\n",
        encoding="utf-8",
    )
    # Isolate os.environ so load_dotenv cannot leak into other tests.
    monkeypatch.setattr(os, "environ", {"MODEL_SMALL": "from-real-env/small"})
    settings = Settings.from_env(dotenv_path=env_file)
    assert settings.gateway.models["primary"] == "from-dotenv/primary"  # loaded from .env
    assert settings.gateway.models["small"] == "from-real-env/small"  # real env wins
    assert settings.gateway.max_retries == 4
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-from-dotenv"  # exported for LiteLLM


def test_explicit_env_mapping_skips_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MODEL_PRIMARY=should-not-load\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {})
    s = Settings.from_env({}, dotenv_path=tmp_path / ".env")
    assert "primary" not in s.gateway.models and "MODEL_PRIMARY" not in os.environ


def test_missing_dotenv_file_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    assert Settings.from_env(dotenv_path=tmp_path / "nope.env").gateway.max_retries == 2


def test_gateway_settings_parse_defaults_and_overrides():
    gs = GatewaySettings.from_env({
        "MODEL_PRIMARY": " a/p ", "MODEL_STRONG": "a/s", "MODEL_SMALL": "a/m",
        "MODEL_PRIMARY_FALLBACK": "b/p", "TIMEOUT_STRONG_S": "45",
        "BREAKER_FAILURES": "3", "BREAKER_COOLDOWN_S": "12.5",
    })
    assert gs.models == {"primary": "a/p", "strong": "a/s", "small": "a/m"}
    assert gs.fallback_models == {"primary": "b/p"}
    assert gs.timeout_s == {"small": 10.0, "primary": 20.0, "strong": 45.0}
    assert (gs.max_retries, gs.breaker_failures, gs.breaker_cooldown_s) == (2, 3, 12.5)


@pytest.mark.parametrize("env", [{"LLM_MAX_RETRIES": "many"}, {"BACKOFF_BASE_S": "-1"},
                                 {"TIMEOUT_SMALL_S": "fast"}])
def test_gateway_settings_reject_bad_numbers(env):
    with pytest.raises(ValueError):
        GatewaySettings.from_env(env)