"""Environment-driven settings (loads .env via pydantic-settings)."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_request_timeout: float = Field(default=30.0, alias="LLM_REQUEST_TIMEOUT")
    llm_confidence_threshold: float = Field(
        default=0.6,
        alias="LLM_CONFIDENCE_THRESHOLD",
    )


@lru_cache
def get_settings() -> Settings:
    # Single cached instance avoids re-reading .env on every LLM call.
    return Settings()
