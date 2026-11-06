"""Shared test helpers: mocked providers so no API key or internet is needed."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pytest

from src.config import DEFAULT_SYSTEM_PROMPT
from src.conversation import Conversation
from src.llm_provider import ChatResult, LLMProvider, ProviderError
from src.memory_manager import MemoryManager
from src.token_counter import TokenCounter


class FakeProvider(LLMProvider):
    """Mocked provider. Summaries are deterministic: existing summary + one line per turn."""

    name = "fake"

    def __init__(self) -> None:
        self.summarize_calls: List[dict] = []
        self.chat_calls = 0

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> ChatResult:
        self.chat_calls += 1
        return ChatResult(text="fake reply", input_tokens=10, output_tokens=2, exact_usage=True)

    def summarize(self, existing_summary: str, turns: Sequence[Dict[str, str]], max_tokens: int) -> str:
        self.summarize_calls.append(
            {"existing": existing_summary, "turns": [dict(t) for t in turns], "max_tokens": max_tokens}
        )
        lines = [existing_summary] if existing_summary else []
        for t in turns:
            if t["role"] == "user":
                lines.append("- " + t["content"][:60])
        return "\n".join(lines)


class FailingProvider(LLMProvider):
    name = "failing"

    def chat(self, messages, max_tokens):
        raise ProviderError("boom")

    def summarize(self, existing_summary, turns, max_tokens):
        raise ProviderError("summarizer is down")


@pytest.fixture
def counter() -> TokenCounter:
    # Force the heuristic so tests are deterministic and need no network.
    return TokenCounter(force_fallback=True)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


def make_manager(counter, provider, budget=400, keep_recent=2) -> MemoryManager:
    return MemoryManager(counter, provider, budget, DEFAULT_SYSTEM_PROMPT, keep_recent)


def add_turns(conv: Conversation, n: int, words: int = 30, prefix: str = "Topic") -> None:
    for i in range(n):
        conv.add_user_message(f"{prefix} {i}: " + "alpha " * words)
        conv.add_assistant_message(f"Answer {i}: " + "beta " * words)
