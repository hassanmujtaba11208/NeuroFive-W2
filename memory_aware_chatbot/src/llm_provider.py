"""AI provider abstraction.

* :class:`OpenAICompatibleProvider` - real LLM calls (OpenAI, OpenRouter, Groq,
  Together, local servers... anything that speaks the OpenAI chat API).
* :class:`OfflineProvider`         - clearly labelled demo mode that needs no key.
  It is NOT an AI model: replies come from simple local rules and summaries are
  built by a rule-based extractor. The token budgeting and compression logic
  that surrounds it is the real thing.

The same provider object is used for chat replies AND memory summarization.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Settings
from .token_counter import TokenCounter


class ProviderError(Exception):
    """Raised when a provider cannot produce a response."""


@dataclass
class ChatResult:
    text: str
    input_tokens: Optional[int] = None  # API-reported, else None
    output_tokens: Optional[int] = None  # API-reported, else None
    exact_usage: bool = False  # True only when the numbers came from the API


@dataclass
class ProviderStatus:
    mode: str  # "live" or "offline"
    label: str
    detail: str


SUMMARY_SYSTEM_PROMPT = (
    "You maintain the long-term memory of a chat assistant. You will receive the "
    "EXISTING MEMORY SUMMARY (may be empty) and OLDER CONVERSATION TURNS. Produce "
    "ONE updated summary that merges both. Rules:\n"
    "- Preserve concrete facts (names, numbers, dates), user preferences, decisions, "
    "open questions, constraints, and unfinished tasks.\n"
    "- Keep everything from the existing summary that is still useful; do not drop "
    "details just because they are old. Remove only duplicates.\n"
    "- Never write vague lines such as 'the user and assistant talked'.\n"
    "- Use short bullet points grouped under the headings: Facts, Preferences, "
    "Decisions, Constraints, Open questions / tasks, Other.\n"
    "- Stay under the requested token limit. Output only the summary."
)


def build_summary_messages(
    existing_summary: str, turns: Sequence[Dict[str, str]], max_tokens: int
) -> List[Dict[str, str]]:
    transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
    user_prompt = (
        f"Token limit for the new summary: about {max_tokens} tokens.\n\n"
        f"EXISTING MEMORY SUMMARY:\n{existing_summary or '(empty)'}\n\n"
        f"OLDER CONVERSATION TURNS TO MERGE IN:\n{transcript or '(none - just condense the summary)'}"
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class LLMProvider(ABC):
    name: str = "provider"
    is_offline: bool = False

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> ChatResult:
        """Return the assistant reply for ``messages``."""

    def summarize(
        self, existing_summary: str, turns: Sequence[Dict[str, str]], max_tokens: int
    ) -> str:
        """Merge ``turns`` into ``existing_summary`` using the same provider."""
        result = self.chat(build_summary_messages(existing_summary, turns, max_tokens), max_tokens)
        text = result.text.strip()
        if not text:
            raise ProviderError("The model returned an empty summary.")
        return text


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0):
        self.model = model
        self.name = f"{model} via {base_url}"
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ProviderError("The 'openai' package is not installed.") from exc
            self._client = OpenAI(
                api_key=self._api_key, base_url=self._base_url, timeout=self._timeout
            )
        return self._client

    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> ChatResult:
        client = self._get_client()
        try:
            try:
                response = client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens
                )
            except Exception as first:
                # Some newer models only accept max_completion_tokens.
                if "max_completion_tokens" in str(first):
                    response = client.chat.completions.create(
                        model=self.model, messages=messages, max_completion_tokens=max_tokens
                    )
                else:
                    raise
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

        try:
            text = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:
            raise ProviderError("Unexpected response format from the provider.") from exc
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None
        return ChatResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            exact_usage=in_tok is not None or out_tok is not None,
        )


# --------------------------------------------------------------------------
# Offline demo provider
# --------------------------------------------------------------------------
OFFLINE_BANNER = "🧪 *Offline demo mode - rule-based reply, not generated by an AI model.*"

# (category, regex). Order = priority when a sentence matches several.
_CATEGORY_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("Facts", re.compile(
        r"\b(my name is|i am called|call me|i am|i'm|i live|i work|i study|i have|"
        r"my (?:\w+ ){0,2}(?:is|are))\b", re.I)),
    ("Preferences", re.compile(
        r"\b(i (?:really )?(?:like|love|prefer|hate|dislike|enjoy)|please (?:always|never)|"
        r"i want|i would like|always|never)\b", re.I)),
    ("Decisions", re.compile(
        r"\b(we decided|i decided|decided to|let's go with|i'll go with|going with|"
        r"the decision|let's use)\b", re.I)),
    ("Constraints", re.compile(
        r"\b(must|must not|cannot|can't|do not|don't|deadline|due|at most|at least|"
        r"no more than|limit|budget|only)\b", re.I)),
    ("Open tasks", re.compile(
        r"\b(todo|to-do|remind me|still need|next step|follow up|later|unfinished|need to)\b", re.I)),
]
_CATEGORY_ORDER = ["Facts", "Preferences", "Decisions", "Constraints", "Open tasks",
                   "Questions", "User said", "Assistant said"]
_MERGE_SEP = " | "
_LOW_PRIORITY = ["Assistant said", "User said", "Questions"]
_BULLET_RE = re.compile(r"^- \[([A-Za-z ]+)\] (.+)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


class OfflineProvider(LLMProvider):
    """Rule-based stand-in that lets the whole memory pipeline be tried without a key."""

    name = "Offline demo (rule-based, no AI model)"
    is_offline = True

    def __init__(self, counter: Optional[TokenCounter] = None):
        self.counter = counter or TokenCounter()

    # ----- chat ------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], max_tokens: int) -> ChatResult:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        context_text = "\n".join(m["content"] for m in messages[:-1])
        n_history = sum(1 for m in messages if m["role"] in ("user", "assistant")) - 1
        has_summary = "MEMORY SUMMARY" in "\n".join(
            m["content"] for m in messages if m["role"] == "system"
        ).upper()

        lines = [OFFLINE_BANNER, ""]
        recalled = self._try_recall(last_user, context_text)
        if recalled:
            lines.append(recalled)
        else:
            lines.append(f"I received your message ({len(last_user)} characters).")
        lines.append(
            f"The context sent to me contained {max(n_history, 0)} earlier message(s) "
            f"and {'a memory summary' if has_summary else 'no memory summary yet'}."
        )
        return ChatResult(text="\n".join(lines))  # no API usage reported offline

    @staticmethod
    def _try_recall(question: str, context: str) -> str:
        q = question.lower()
        if "my name" in q and "?" in q or "what is my name" in q or "who am i" in q:
            found = re.findall(r"my name is ([A-Za-z][\w'-]*)", context, re.I)
            if found:
                return f"From the context I can see your name is {found[-1]}."
            return "I can't find your name anywhere in my current context."
        if "favorite" in q or "favourite" in q or "what do i like" in q:
            found = re.findall(r"i (?:really )?(?:like|love|prefer) ([^.!?\n]+)", context, re.I)
            if found:
                return f"From the context, you said you like/prefer: {found[-1].strip()}."
            return "I can't find a stated preference in my current context."
        return ""

    # ----- summarization (rule-based, extractive) ----------------------------
    def summarize(
        self, existing_summary: str, turns: Sequence[Dict[str, str]], max_tokens: int
    ) -> str:
        bullets = self._parse(existing_summary)
        for item in self._extract(turns):
            if item[1].lower() not in {b[1].lower() for b in bullets}:
                bullets.append(item)
        if not bullets:
            raise ProviderError("Nothing to summarize.")
        return self._render_within(bullets, max_tokens)

    @staticmethod
    def _parse(summary: str) -> List[Tuple[str, str]]:
        out = []
        for line in summary.splitlines():
            m = _BULLET_RE.match(line.strip())
            if m:
                out.append((m.group(1), m.group(2)))
        return out

    @staticmethod
    def _extract(turns: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for t in turns:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(t["content"]) if s.strip()]
            if t["role"] == "assistant":
                if sentences:
                    # Skip the offline banner line when summarizing our own replies.
                    body = [s for s in sentences if "Offline demo mode" not in s] or sentences
                    items.append(("Assistant said", body[0]))
                continue
            matched_any = False
            for s in sentences:
                for cat, rx in _CATEGORY_RULES:
                    if rx.search(s):
                        items.append((cat, s))
                        matched_any = True
                        break
                else:
                    if s.endswith("?"):
                        items.append(("Questions", s))
                        matched_any = True
            if not matched_any and sentences:
                items.append(("User said", sentences[0]))
        return items

    def _render(self, bullets: List[Tuple[str, str]], limit: int) -> str:
        ordered = sorted(
            enumerate(bullets),
            key=lambda p: (_CATEGORY_ORDER.index(p[1][0]) if p[1][0] in _CATEGORY_ORDER else 99, p[0]),
        )
        # Merged bullets (joined with " | ") are already compact and are not clipped again.
        return "\n".join(
            f"- [{cat}] {text if _MERGE_SEP in text else _clip(text, limit)}"
            for _, (cat, text) in ordered
        )

    def _fits(self, bullets: List[Tuple[str, str]], limit: int, max_tokens: int) -> bool:
        return self.counter.count_text(self._render(bullets, limit)) <= max_tokens

    def _render_within(self, bullets: List[Tuple[str, str]], max_tokens: int) -> str:
        """Render the bullets so the summary fits ``max_tokens``.

        Escalation (least lossy first):
        1. shorten each bullet's text;
        2. merge low-priority bullets (questions / paraphrases) into one compact bullet per
           category, keeping a short fragment of every item;
        3. only if still too large, drop the OLDEST fragments of those low-priority bullets.
        Facts, preferences, decisions, constraints and open tasks are never dropped.
        """
        bullets = list(bullets)
        for limit in (240, 160, 110, 80, 60):
            if self._fits(bullets, limit, max_tokens):
                return self._render(bullets, limit)

        merged: List[Tuple[str, str]] = []
        for cat in _LOW_PRIORITY:
            fragments: List[str] = []
            for c, text in bullets:
                if c == cat:
                    fragments.extend(_clip(f, 60) for f in text.split(_MERGE_SEP))
            fragments = list(dict.fromkeys(fragments))  # drop exact duplicates, keep order
            if fragments:
                merged.append((cat, _MERGE_SEP.join(fragments)))
        bullets = [b for b in bullets if b[0] not in _LOW_PRIORITY] + merged
        if self._fits(bullets, 60, max_tokens):
            return self._render(bullets, 60)

        for cat in _LOW_PRIORITY:
            while not self._fits(bullets, 60, max_tokens):
                idx = next((i for i, b in enumerate(bullets) if b[0] == cat), None)
                if idx is None:
                    break
                frags = bullets[idx][1].split(_MERGE_SEP)
                if len(frags) > 1:
                    bullets[idx] = (cat, _MERGE_SEP.join(frags[1:]))  # drop oldest fragment
                else:
                    bullets.pop(idx)
        return self._render(bullets, 60)


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def create_provider(
    settings: Settings, counter: Optional[TokenCounter] = None
) -> Tuple[LLMProvider, ProviderStatus]:
    """Pick the provider and describe the choice honestly (never fakes a live LLM)."""
    if settings.offline_mode:
        return OfflineProvider(counter), ProviderStatus(
            "offline", "Offline demo mode", "Forced by OFFLINE_MODE / sidebar toggle. No AI model is used."
        )
    if not settings.has_credentials:
        return OfflineProvider(counter), ProviderStatus(
            "offline",
            "Offline demo mode (no API key)",
            "OPENAI_API_KEY is missing, so the rule-based demo provider is active. "
            "Add a key to use a real LLM.",
        )
    return (
        OpenAICompatibleProvider(settings.api_key, settings.base_url, settings.model),
        ProviderStatus(
            "live", f"Live: {settings.model}", f"Key {settings.masked_key} · {settings.base_url}"
        ),
    )
