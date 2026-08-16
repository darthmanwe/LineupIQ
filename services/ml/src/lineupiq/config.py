"""Runtime settings.

Nothing here is required to reproduce a published number. Every field has a
working default, and the ones that would cost money or need credentials default
to absent -- so a clean clone runs offline and free without a `.env` at all.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["SEED", "Settings", "settings"]

#: Global random seed. Every fit, split, bootstrap and restart derives from this
#: one integer, so `train --verify` can assert bit-level reproduction.
SEED: Final[int] = 20260815


class Settings(BaseSettings):
    """Environment-backed configuration.

    Secrets are read from the environment or a local ``.env``; neither is
    committed and neither is needed for the offline path.
    """

    model_config = SettingsConfigDict(
        env_prefix="LINEUPIQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = SEED

    # -- LLM layer (optional; the committed cache means the demo needs none) --
    anthropic_api_key: str | None = Field(
        default=None,
        description="Only needed to regenerate narratives. The committed cache serves the demo.",
    )
    llm_max_calls: int = Field(
        default=0,
        ge=0,
        description=(
            "Hard ceiling on billed calls per invocation, checked before each request. "
            "Defaults to 0 so no code path can spend money by accident -- raising it is "
            "an explicit, per-run decision."
        ),
    )

    # -- Snowflake adapter (optional; off the demo path entirely) ------------
    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_role: str | None = None
    snowflake_warehouse: str | None = None

    @property
    def has_snowflake_credentials(self) -> bool:
        return bool(self.snowflake_account and self.snowflake_user)

    @property
    def can_spend(self) -> bool:
        """True only when a key exists *and* a positive call budget was set."""
        return bool(self.anthropic_api_key) and self.llm_max_calls > 0


settings = Settings()
