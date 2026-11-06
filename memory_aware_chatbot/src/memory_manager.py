"""Context-budget enforcement and memory compression.

``MemoryManager.prepare_context`` is called BEFORE every model request:

1. Measure the context (system prompt + memory summary + active messages).
2. While it exceeds the budget, pick the oldest complete turns (keeping the most
   recent turns whenever possible), summarize them with the configured provider,
   and merge the result into the existing summary.
3. Commit (flag the messages as summarized + store the summary) ONLY after the
   summary was produced successfully. If summarization fails, nothing is
   changed and no request is allowed to go out.
4. If the context still cannot fit (e.g. one gigantic message or an impossible
   budget) the result is ``ok=False`` with a clear error: an oversized request is
   never sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import DEFAULT_KEEP_RECENT_TURNS, DEFAULT_SYSTEM_PROMPT
from .conversation import CompressionEvent, Conversation, Message
from .llm_provider import LLMProvider
from .token_counter import REPLY_PRIMING_TOKENS, PER_MESSAGE_OVERHEAD, TokenCounter

SUMMARY_HEADER = "Memory summary of earlier conversation (older turns, compressed):"
MAX_COMPRESSION_ROUNDS = 50
TARGET_FILL = 0.70  # after compressing, aim for ~70% of budget to avoid compressing every turn


@dataclass
class TokenUsage:
    """Local ESTIMATE of the input context (never an API-exact figure)."""

    budget: int
    system_tokens: int
    summary_tokens: int
    history_tokens: int
    overhead_tokens: int
    method: str = ""

    @property
    def total(self) -> int:
        return self.system_tokens + self.summary_tokens + self.history_tokens + self.overhead_tokens

    @property
    def remaining(self) -> int:
        return self.budget - self.total

    @property
    def utilization(self) -> float:
        return self.total / self.budget if self.budget > 0 else 1.0

    @property
    def within_budget(self) -> bool:
        return self.total <= self.budget


@dataclass
class ContextResult:
    ok: bool
    messages: List[Dict[str, str]]
    usage: TokenUsage
    events: List[CompressionEvent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def compressed(self) -> bool:
        return bool(self.events)


class MemoryManager:
    def __init__(
        self,
        counter: TokenCounter,
        provider: LLMProvider,
        budget: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    ) -> None:
        self.counter = counter
        self.provider = provider
        self.budget = budget
        self.system_prompt = system_prompt
        self.keep_recent_turns = max(0, keep_recent_turns)

    # ----- sizing ------------------------------------------------------------
    @property
    def summary_max_tokens(self) -> int:
        """Size limit requested for the summary (a fraction of the budget)."""
        return max(80, min(600, self.budget // 5))

    def _summary_block(self, summary: str) -> str:
        return f"{SUMMARY_HEADER}\n{summary}" if summary else ""

    def measure(self, conv: Conversation) -> TokenUsage:
        """Estimate the tokens the model would receive for the current state."""
        system_tokens = self.counter.count_text(self.system_prompt) + self.counter.count_text("system") + PER_MESSAGE_OVERHEAD
        summary_tokens = self.counter.count_text(self._summary_block(conv.summary))
        history_tokens = sum(self.counter.count_message(m.as_dict()) for m in conv.active_messages())
        return TokenUsage(
            budget=self.budget,
            system_tokens=system_tokens,
            summary_tokens=summary_tokens,
            history_tokens=history_tokens,
            overhead_tokens=REPLY_PRIMING_TOKENS,
            method=self.counter.method_label,
        )

    def build_messages(self, conv: Conversation) -> List[Dict[str, str]]:
        """Exactly what is sent to the model: ONE system message + active messages."""
        system = self.system_prompt
        if conv.summary:
            system = f"{system}\n\n{self._summary_block(conv.summary)}"
        return [{"role": "system", "content": system}] + [m.as_dict() for m in conv.active_messages()]

    # ----- selecting what to summarize --------------------------------------------
    def _select_batch(self, conv: Conversation, usage: TokenUsage):
        """Return (messages, turns, kept_recent) for the oldest turns to summarize, or None."""
        completed = conv.completed_turns()
        keep = self.keep_recent_turns
        eligible: List[List[Message]] = []
        while keep >= 0:
            eligible = completed[: max(0, len(completed) - keep)]
            if eligible:
                break
            keep -= 1
        if not eligible:
            return None

        target = int(self.budget * TARGET_FILL)
        reserve = self.summary_max_tokens
        batch_turns: List[List[Message]] = []
        freed = 0
        batch_tokens = 0
        for turn in eligible:
            turn_tokens = sum(self.counter.count_message(m.as_dict()) for m in turn)
            if batch_turns and batch_tokens + turn_tokens > max(self.budget, 1000):
                break  # keep the summarization request itself reasonably small
            batch_turns.append(turn)
            freed += turn_tokens
            batch_tokens += turn_tokens
            if usage.total - freed + reserve <= target:
                break
        messages = [m for turn in batch_turns for m in turn]
        return messages, batch_turns, keep

    # ----- main entry point ------------------------------------------------------------
    def prepare_context(self, conv: Conversation) -> ContextResult:
        events: List[CompressionEvent] = []
        warnings: List[str] = []
        usage = self.measure(conv)
        condensed_once = False

        def fail(message: str) -> ContextResult:
            conv.last_error = message
            return ContextResult(False, [], self.measure(conv), events, warnings, message)

        rounds = 0
        while usage.total > self.budget:
            rounds += 1
            if rounds > MAX_COMPRESSION_ROUNDS:
                return fail("Compression did not converge; request blocked to avoid exceeding the budget.")

            selection = self._select_batch(conv, usage)

            if selection is None:
                # Nothing older left to fold in. Last resorts: condense the summary once.
                if conv.summary and not condensed_once and usage.summary_tokens > self.budget // 4:
                    condensed_once = True
                    before = usage.total
                    try:
                        condensed = self.provider.summarize(conv.summary, [], max(60, self.summary_max_tokens // 2))
                    except Exception as exc:
                        return fail(
                            "The memory summary is too large and could not be condensed "
                            f"({type(exc).__name__}: {exc}). Request blocked; no data was lost."
                        )
                    if not condensed.strip():
                        return fail("Condensing the memory summary returned nothing. Request blocked; no data was lost.")
                    conv.summary = condensed.strip()
                    usage = self.measure(conv)
                    events.append(CompressionEvent(0, 0, before, usage.total, usage.summary_tokens,
                                                   "Condensed the existing memory summary"))
                    conv.stats.record(events[-1])
                    continue
                return fail(
                    f"The context needs about {usage.total} tokens but the budget is {self.budget}. "
                    "Nothing older remains to summarize, so the request was NOT sent. "
                    "Shorten your message or raise the token budget in the sidebar."
                )

            messages, turns, kept = selection
            if kept < self.keep_recent_turns:
                warnings.append(
                    "Budget is very tight: some of the most recent turns had to be summarized too."
                )
            before = usage.total
            try:
                new_summary = self.provider.summarize(
                    conv.summary, [m.as_dict() for m in messages], self.summary_max_tokens
                )
            except Exception as exc:
                return fail(
                    f"Memory summarization failed ({type(exc).__name__}: {exc}). "
                    "The original messages were kept and the request was NOT sent."
                )
            if not new_summary or not new_summary.strip():
                return fail("Memory summarization returned an empty summary. "
                            "The original messages were kept and the request was NOT sent.")

            conv.apply_summary(new_summary.strip(), messages)  # commit only after success
            usage = self.measure(conv)
            event = CompressionEvent(
                turns_summarized=len(turns),
                messages_summarized=len(messages),
                tokens_before=before,
                tokens_after=usage.total,
                summary_tokens=usage.summary_tokens,
                note="Summarized oldest turns and merged into memory",
            )
            events.append(event)
            conv.stats.record(event)
            if usage.total >= before:
                warnings.append("A summary was not smaller than the turns it replaced; compressing further.")

        conv.last_error = ""
        return ContextResult(True, self.build_messages(conv), usage, events, warnings, None)
