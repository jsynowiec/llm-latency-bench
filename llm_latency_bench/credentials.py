"""The OpenRouter API key from the environment, and redaction of it from any text we emit."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ENV_API_KEY = "OPENROUTER_API_KEY"
REDACTED = "[REDACTED]"
# The value in `.env.example`; running with it would only produce auth failures.
_PLACEHOLDER_MARKER = "replace-with-"


class CredentialsError(ValueError):
    pass


def load_api_key(env_file: Path | None, environ: Mapping[str, str] = os.environ) -> str:
    """Read the API key from the process environment, falling back to `env_file`.

    The file is parsed into a local dict and never copied into `os.environ`, and the key is never logged.
    """
    file_values = dotenv_values(env_file) if env_file is not None and env_file.exists() else {}
    api_key = (environ.get(ENV_API_KEY) or file_values.get(ENV_API_KEY) or "").strip()
    if not api_key:
        raise CredentialsError(f"missing environment variable: {ENV_API_KEY} (see .env.example)")
    if _PLACEHOLDER_MARKER in api_key:
        raise CredentialsError(f"{ENV_API_KEY} still holds the .env.example placeholder")
    return api_key


@dataclass(frozen=True)
class Redactor:
    secrets: tuple[str, ...]

    def __call__(self, text: str) -> str:
        # Longest first, so a secret that contains another secret is removed whole.
        for secret in sorted(filter(None, self.secrets), key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        return text
