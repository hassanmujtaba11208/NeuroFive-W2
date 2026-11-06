"""Memory-Aware Chatbot with Streamlit UI  -  application entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

try:  # optional: load a local .env file during development
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - python-dotenv is optional at runtime
    pass

from src.config import (
    DEFAULT_SYSTEM_PROMPT,
    MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    load_settings,
)
from src.conversation import ApiUsage, Conversation
from src.llm_provider import ProviderError, create_provider
from src.memory_manager import MemoryManager
from src.token_counter import TokenCounter

st.set_page_config(
    page_title="Memory-Aware Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1100px;}
.app-title {font-size: 1.9rem; font-weight: 700; color: #4F46E5; margin-bottom: 0;}
.app-sub {color: #6B7280; margin-top: .1rem; margin-bottom: .8rem;}
div[data-testid="stMetric"] {background: #F3F4F8; border-radius: 10px; padding: .6rem .8rem;}
div[data-testid="stMetricLabel"] p {font-size: .78rem; color: #4B5563;}
.tag-est {background:#FEF3C7;color:#92400E;border-radius:6px;padding:1px 7px;font-size:.72rem;font-weight:600;}
.tag-exact {background:#DCFCE7;color:#166534;border-radius:6px;padding:1px 7px;font-size:.72rem;font-weight:600;}
@media (max-width: 640px) {.app-title {font-size: 1.4rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- state
@st.cache_resource(show_spinner=False)
def get_counter(encoding: str) -> TokenCounter:
    return TokenCounter(encoding)


def init_state() -> None:
    if "conv" not in st.session_state:
        st.session_state.conv = Conversation()
    st.session_state.setdefault("last_events", [])
    st.session_state.setdefault("last_warnings", [])
    st.session_state.setdefault("last_error", "")
    st.session_state.setdefault("rejected_text", "")


def reset_conversation() -> None:
    """Clear messages, summary, counters and all temporary UI state."""
    st.session_state.conv.reset()
    st.session_state.last_events = []
    st.session_state.last_warnings = []
    st.session_state.last_error = ""
    st.session_state.rejected_text = ""


@st.dialog("Reset conversation?")
def confirm_reset_dialog() -> None:
    st.write(
        "This permanently clears **all messages, the memory summary, token statistics "
        "and compression counters**, and starts a fresh conversation."
    )
    c1, c2 = st.columns(2)
    if c1.button("Yes, reset", type="primary", use_container_width=True):
        reset_conversation()
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.rerun()


def load_app_settings():
    try:
        return load_settings(secrets=st.secrets)
    except Exception:  # no secrets file -> environment only
        return load_settings()


init_state()
settings = load_app_settings()
conv: Conversation = st.session_state.conv

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Reset Conversation", type="primary", use_container_width=True):
        confirm_reset_dialog()

    st.subheader("Token budget")
    budget = st.slider(
        "Context token budget",
        min_value=MIN_TOKEN_BUDGET,
        max_value=MAX_TOKEN_BUDGET,
        value=settings.token_budget,
        step=100,
        help="Maximum estimated INPUT tokens sent to the model per request (system prompt + "
        "memory summary + recent messages). Older turns are summarized to stay under it.",
    )
    keep_recent = st.slider(
        "Recent turns kept verbatim",
        min_value=0,
        max_value=6,
        value=settings.keep_recent_turns,
        help="Most recent user/assistant turns that are kept word-for-word whenever possible.",
    )

    st.subheader("AI provider")
    forced_offline = not settings.has_credentials
    offline = st.toggle(
        "Offline demo mode",
        value=forced_offline or settings.offline_mode,
        disabled=forced_offline,
        help="Uses a rule-based stand-in instead of an AI model. Token budgeting and "
        "memory compression still run for real.",
    )
    effective = replace(settings, offline_mode=offline)
    counter = get_counter(effective.encoding)
    provider, status = create_provider(effective, counter)

    if status.mode == "live":
        st.success(f"🟢 {status.label}")
    else:
        st.warning(f"🟡 {status.label}")
    st.caption(status.detail)
    st.caption(
        "Configure `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `LLM_MODEL` in a `.env` file or "
        "Streamlit secrets. The same provider handles chat and summarization."
    )

    st.subheader("Token counting")
    st.caption(f"Method: **{counter.method_label}**")
    if not counter.using_tiktoken:
        st.caption("tiktoken could not be loaded; using the heuristic fallback estimator.")
    st.caption(
        "ℹ️ Local counts are **estimates**. Real usage depends on the provider's model, "
        "tokenizer, message formatting and API overhead."
    )
    with st.expander("System prompt"):
        st.code(DEFAULT_SYSTEM_PROMPT, language=None)

manager = MemoryManager(counter, provider, budget, DEFAULT_SYSTEM_PROMPT, keep_recent)

# --------------------------------------------------------------------------- header
st.markdown('<p class="app-title">🧠 Memory-Aware Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Multi-turn chat that stays inside a token budget by <b>summarizing older '
    "turns</b> into memory instead of silently deleting them.</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- dashboard
usage = manager.measure(conv)

with st.container(border=True):
    st.markdown(
        '**📊 Token usage (input context)** <span class="tag-est">LOCAL ESTIMATE</span>',
        unsafe_allow_html=True,
    )
    r1 = st.columns(3)
    r1[0].metric("Token budget", f"{usage.budget:,}")
    r1[1].metric("Total input context", f"{usage.total:,}")
    r1[2].metric("Tokens remaining", f"{usage.remaining:,}")
    r2 = st.columns(3)
    r2[0].metric("System prompt", f"{usage.system_tokens:,}")
    r2[1].metric("Memory summary", f"{usage.summary_tokens:,}")
    r2[2].metric("Recent history", f"{usage.history_tokens:,}")

    st.progress(
        min(max(usage.utilization, 0.0), 1.0),
        text=f"Budget utilization: {usage.utilization:.0%} ({usage.total:,} / {usage.budget:,})",
    )
    st.caption(
        f"Includes ~{usage.overhead_tokens} tokens of message-format overhead. "
        f"Counting method: {usage.method}."
    )

    r3 = st.columns(3)
    api = conv.last_api_usage
    r3[0].metric(
        "Last response: input tokens",
        f"{api.input_tokens:,}" if api and api.input_tokens is not None else "n/a",
        help="Reported by the API (exact). 'n/a' in offline mode or if the provider reports no usage.",
    )
    r3[1].metric(
        "Last response: output tokens",
        f"{api.output_tokens:,}" if api and api.output_tokens is not None else "n/a",
        help="Reported by the API (exact). Output tokens are NOT part of the input budget above.",
    )
    r3[2].metric(
        "Compressions / turns summarized",
        f"{conv.stats.compression_events} / {conv.stats.turns_summarized}",
    )
    st.markdown(
        '<span class="tag-exact">API-REPORTED</span> applies only to the two "Last response" '
        "metrics; everything else above is a local estimate.",
        unsafe_allow_html=True,
    )

    # memory-management status
    if conv.is_empty:
        st.info("🧠 Memory status: empty conversation - nothing to manage yet.")
    elif not usage.within_budget:
        st.error("🔴 Memory status: over budget - older turns will be summarized before the next request.")
    elif usage.utilization >= 0.8:
        st.warning("🟡 Memory status: near the budget - compression will trigger soon.")
    else:
        st.success("🟢 Memory status: within budget.")

# notices from the last request
if st.session_state.last_events:
    ev = st.session_state.last_events
    turns = sum(e.turns_summarized for e in ev)
    st.info(
        f"🗜️ **Memory compressed** during the last request: {turns} older turn(s) summarized "
        f"({ev[0].tokens_before:,} → {ev[-1].tokens_after:,} estimated tokens)."
    )
for w in st.session_state.last_warnings:
    st.warning(w)
if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.rejected_text:
        with st.expander("Your unsent message"):
            st.code(st.session_state.rejected_text, language=None)

# --------------------------------------------------------------------------- memory panel
with st.expander(
    f"🧠 Memory summary ({usage.summary_tokens:,} est. tokens)", expanded=bool(conv.summary)
):
    if conv.summary:
        st.markdown(conv.summary)
        st.caption(
            "This compressed memory replaces the summarized messages in the model's context. "
            "The original messages stay visible in the transcript below."
        )
    else:
        st.caption("No summary yet. It is created automatically when the conversation exceeds the budget.")

# --------------------------------------------------------------------------- transcript
st.subheader("💬 Conversation")
older = conv.summarized_messages()
if older:
    with st.expander(f"📦 Earlier messages ({len(older)}) - already summarized into memory"):
        for m in older:
            with st.chat_message(m.role):
                st.markdown(m.content)
    st.caption("⬇️ Recent messages - sent to the model verbatim")

if not conv.messages:
    st.caption("Say hello! Tip: tell the bot your name and preferences, then keep chatting "
               "until compression kicks in. Lower the budget in the sidebar to see it sooner.")

for m in conv.active_messages():
    with st.chat_message(m.role):
        st.markdown(m.content)

# --------------------------------------------------------------------------- handle input
prompt = st.chat_input("Type your message...")
if prompt and prompt.strip():
    st.session_state.last_events = []
    st.session_state.last_warnings = []
    st.session_state.last_error = ""
    st.session_state.rejected_text = ""

    conv.add_user_message(prompt.strip())
    with st.chat_message("user"):
        st.markdown(prompt.strip())

    with st.chat_message("assistant"):
        with st.spinner("Checking token budget and preparing context..."):
            result = manager.prepare_context(conv)

        st.session_state.last_events = result.events
        st.session_state.last_warnings = result.warnings

        if not result.ok:
            # Nothing was sent; the unanswered message is handed back to the user.
            st.session_state.rejected_text = conv.discard_pending_user_message() or ""
            st.session_state.last_error = result.error or "The context could not be prepared."
        else:
            try:
                with st.spinner("Thinking..."):
                    reply = provider.chat(result.messages, settings.max_response_tokens)
            except ProviderError as exc:
                st.session_state.rejected_text = conv.discard_pending_user_message() or ""
                st.session_state.last_error = (
                    f"The AI provider returned an error: {exc}. Your message was not added to the "
                    "conversation; please try again."
                )
            else:
                conv.add_assistant_message(reply.text)
                conv.last_api_usage = (
                    ApiUsage(reply.input_tokens, reply.output_tokens) if reply.exact_usage else None
                )
    st.rerun()  # refresh dashboard, summary and transcript with the new state
