"""Token counting.

Uses ``tiktoken`` (default encoding ``cl100k_base``) when it can be loaded, and
falls back to a conservative heuristic estimator otherwise.

IMPORTANT: every count produced here is a LOCAL ESTIMATE. Real token usage
depends on the provider's model, tokenizer, message formatting and API
overhead. Only usage reported by the API itself is "exact", and the UI labels
the two kinds of numbers differently.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping

# Approximate chat-format overhead (OpenAI-style): every message carries a few
# framing tokens and the reply is primed with a few more. These are estimates.
PER_MESSAGE_OVERHEAD = 4
REPLY_PRIMING_TOKENS = 3

_WORD_RE = re.compile(r"\S+")


def estimate_tokens_fallback(text: str) -> int:
    """Heuristic token estimate used when tiktoken is unavailable.

    ASCII text averages roughly 4 characters per token, while non-ASCII
    characters (e.g. Urdu, Chinese, emoji) often cost about one token each.
    We also enforce a floor of 1 token per word, so the estimate errs on the
    high (safe) side for budget enforcement.
    """
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    by_chars = math.ceil(ascii_chars / 4) + non_ascii
    by_words = len(_WORD_RE.findall(text))
    return max(by_chars, by_words)


class TokenCounter:
    """Counts tokens for text and chat messages."""

    def __init__(self, encoding_name: str = "cl100k_base", force_fallback: bool = False):
        self.requested_encoding = encoding_name
        self._encoding = None
        self.load_error = ""
        if not force_fallback:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(encoding_name)
            except Exception as exc:  # missing package, bad name, or no network for BPE file
                self._encoding = None
                self.load_error = f"{type(exc).__name__}: {exc}"

    @property
    def using_tiktoken(self) -> bool:
        return self._encoding is not None

    @property
    def method_label(self) -> str:
        """Human-readable description of how counts are produced."""
        if self.using_tiktoken:
            return f"tiktoken ({self.requested_encoding}) — local estimate"
        return "heuristic fallback (~4 chars/token) — local estimate"

    def count_text(self, text: str) -> int:
        """Tokens in a raw string (no message framing)."""
        if not text:
            return 0
        if self._encoding is not None:
            try:
                return len(self._encoding.encode(text, disallowed_special=()))
            except Exception:
                pass  # fall through to the heuristic
        return estimate_tokens_fallback(text)

    def count_message(self, message: Mapping[str, str]) -> int:
        """Tokens for one chat message including approximate framing overhead."""
        return (
            self.count_text(message.get("content", ""))
            + self.count_text(message.get("role", ""))
            + PER_MESSAGE_OVERHEAD
        )

    def count_messages(self, messages: Iterable[Mapping[str, str]]) -> int:
        """Tokens for a list of chat messages including reply-priming overhead."""
        return sum(self.count_message(m) for m in messages) + REPLY_PRIMING_TOKENS
