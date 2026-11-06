"""Configuration management.

Settings are read from (in order of priority):
1. Streamlit secrets (when running on Streamlit Community Cloud), passed in as a mapping
2. Environment variables (optionally loaded from a local ``.env`` file by ``app.py``)
3. Built-in defaults

No credential is ever hardcoded in this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

DEFAULT_TOKEN_BUDGET = 4000
MIN_TOKEN_BUDGET = 500
MAX_TOKEN_BUDGET = 16000
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_MAX_RESPONSE_TOKENS = 600
DEFAULT_KEEP_RECENT_TURNS = 2

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use the conversation memory summary "
    "(if present) together with the recent messages to stay consistent with "
    "facts, preferences and decisions the user shared earlier."
)

# Placeholder values that must never be treated as a real key.
_PLACEHOLDER_KEYS = {"", "your-api-key-here", "changeme", "sk-xxxx", "your_api_key"}


def clamp_budget(value: object) -> int:
    """Return a valid token budget: parse ``value`` and clamp it to the allowed range."""
    try:
        budget = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_TOKEN_BUDGET
    return max(MIN_TOKEN_BUDGET, min(MAX_TOKEN_BUDGET, budget))


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    encoding: str = DEFAULT_ENCODING
    token_budget: int = DEFAULT_TOKEN_BUDGET
    offline_mode: bool = False
    max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS

    @property
    def has_credentials(self) -> bool:
        return self.api_key.strip().lower() not in _PLACEHOLDER_KEYS

    @property
    def masked_key(self) -> str:
        """Safe-to-display version of the key (never the full secret)."""
        if not self.has_credentials:
            return "(not set)"
        key = self.api_key.strip()
        return f"{key[:3]}…{key[-2:]}" if len(key) > 8 else "set"


def _lookup(name: str, secrets: Optional[Mapping], env: Mapping) -> Optional[str]:
    if secrets is not None:
        try:
            if name in secrets:
                return str(secrets[name])
        except Exception:  # secrets backends can raise if no secrets file exists
            pass
    value = env.get(name)
    return None if value is None else str(value)


def load_settings(
    env: Optional[Mapping] = None, secrets: Optional[Mapping] = None
) -> Settings:
    """Build :class:`Settings` from secrets/environment, applying safe defaults."""
    env = os.environ if env is None else env

    def get(name: str, default: str) -> str:
        value = _lookup(name, secrets, env)
        return default if value is None or value.strip() == "" else value.strip()

    return Settings(
        api_key=get("OPENAI_API_KEY", ""),
        base_url=get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        model=get("LLM_MODEL", DEFAULT_MODEL),
        encoding=get("TOKEN_ENCODING", DEFAULT_ENCODING),
        token_budget=clamp_budget(get("TOKEN_BUDGET", str(DEFAULT_TOKEN_BUDGET))),
        offline_mode=_parse_bool(get("OFFLINE_MODE", "false")),
    )
