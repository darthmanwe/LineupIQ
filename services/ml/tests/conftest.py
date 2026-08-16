"""Test-suite guardrails.

A bare ``pytest`` must not be able to spend money or reach the network, even on
a developer machine that has credentials sitting in the environment or a local
``.env``. Clearing the environment alone is not enough: pydantic-settings reads
the dotenv file itself, so a `.env` on disk would repopulate every field the
environment scrub just removed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from lineupiq import config

_CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "LINEUPIQ_ANTHROPIC_API_KEY",
    "LINEUPIQ_LLM_MAX_CALLS",
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "LINEUPIQ_SNOWFLAKE_ACCOUNT",
    "LINEUPIQ_SNOWFLAKE_USER",
    "CLOUDFLARE_API_TOKEN",
)


@pytest.fixture(autouse=True)
def _scrub_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every credential from the environment for the duration of a test."""
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop pydantic-settings loading a developer's local ``.env``.

    Without this the environment scrub above is cosmetic: ``Settings()`` would
    read the file directly and hand back the very keys the test is trying to
    prove are absent.
    """
    monkeypatch.setitem(config.Settings.model_config, "env_file", None)
    yield


@pytest.fixture
def offline_settings() -> config.Settings:
    """A freshly constructed Settings with no credentials and no spend budget."""
    return config.Settings()
