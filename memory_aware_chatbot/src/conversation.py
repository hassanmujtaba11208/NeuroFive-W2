"""Conversation state: full transcript, memory summary and statistics.

The conversation keeps TWO separate things:

* ``messages``  - the full transcript, in order. Nothing is ever deleted from it
  (except by an explicit reset). Messages that have been folded into the memory
  summary are flagged ``summarized=True``; the UI still shows them.
* ``summary``   - the compressed long-term memory that stands in for the
  summarized messages when a request is sent to the model.

Only messages with ``summarized=False`` ("active" messages) are sent to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    summarized: bool = False

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ApiUsage:
    """Token usage as REPORTED BY THE API for the latest response (exact)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class CompressionEvent:
    turns_summarized: int
    messages_summarized: int
    tokens_before: int
    tokens_after: int
    summary_tokens: int
    note: str = ""


@dataclass
class MemoryStats:
    compression_events: int = 0
    turns_summarized: int = 0
    messages_summarized: int = 0
    log: List[CompressionEvent] = field(default_factory=list)

    def record(self, event: CompressionEvent) -> None:
        self.compression_events += 1
        self.turns_summarized += event.turns_summarized
        self.messages_summarized += event.messages_summarized
        self.log.append(event)


class Conversation:
    """Holds the transcript, the memory summary and compression statistics."""

    def __init__(self) -> None:
        self.messages: List[Message] = []
        self.summary: str = ""
        self.stats = MemoryStats()
        self.last_api_usage: Optional[ApiUsage] = None
        self.last_error: str = ""

    # ----- adding messages -------------------------------------------------
    def add_user_message(self, content: str) -> Message:
        msg = Message("user", content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        msg = Message("assistant", content)
        self.messages.append(msg)
        return msg

    def discard_pending_user_message(self) -> Optional[str]:
        """Remove a trailing user message that never received a reply.

        Used only when a request could not be sent (e.g. context cannot be made
        to fit). Returns the text so the UI can show it back to the user.
        """
        if self.messages and self.messages[-1].role == "user":
            return self.messages.pop().content
        return None

    # ----- views -----------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.messages and not self.summary

    def active_messages(self) -> List[Message]:
        """Messages that are still sent to the model verbatim."""
        return [m for m in self.messages if not m.summarized]

    def summarized_messages(self) -> List[Message]:
        """Messages already folded into the summary (kept for display/audit)."""
        return [m for m in self.messages if m.summarized]

    def completed_turns(self) -> List[List[Message]]:
        """Active (user, assistant) pairs. A trailing unanswered user message is excluded."""
        active = self.active_messages()
        turns: List[List[Message]] = []
        i = 0
        while i + 1 < len(active):
            if active[i].role == "user" and active[i + 1].role == "assistant":
                turns.append([active[i], active[i + 1]])
                i += 2
            else:
                i += 1  # skip a malformed/unpaired message defensively
        return turns

    # ----- memory commit ---------------------------------------------------
    def apply_summary(self, new_summary: str, summarized: List[Message]) -> None:
        """Commit a successful summarization: store summary, flag the messages.

        Called ONLY after the summary was generated successfully, so a failure
        can never lose data.
        """
        self.summary = new_summary
        ids = {id(m) for m in summarized}
        for m in self.messages:
            if id(m) in ids:
                m.summarized = True

    # ----- reset -----------------------------------------------------------
    def reset(self) -> None:
        """Clear everything: messages, summary, counters and temporary state."""
        self.messages = []
        self.summary = ""
        self.stats = MemoryStats()
        self.last_api_usage = None
        self.last_error = ""
