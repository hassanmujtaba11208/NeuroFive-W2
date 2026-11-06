from src.config import DEFAULT_SYSTEM_PROMPT, load_settings
from src.conversation import Conversation
from src.llm_provider import OfflineProvider, create_provider
from src.memory_manager import SUMMARY_HEADER

from .conftest import FailingProvider, add_turns, make_manager


# ---------------------------------------------------------------- empty / basic
def test_empty_conversation_context(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=1000)
    conv = Conversation()
    result = mgr.prepare_context(conv)
    assert result.ok and not result.compressed
    assert [m["role"] for m in result.messages] == ["system"]
    assert result.usage.history_tokens == 0 and result.usage.summary_tokens == 0
    assert result.usage.total == result.usage.system_tokens + result.usage.overhead_tokens
    assert fake_provider.summarize_calls == []


def test_no_compression_when_under_budget(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=4000)
    conv = Conversation()
    add_turns(conv, 3, words=10)
    conv.add_user_message("pending question")
    result = mgr.prepare_context(conv)
    assert result.ok and not result.compressed
    assert fake_provider.summarize_calls == []
    # system message + 6 history messages + the pending message, in order
    assert [m["role"] for m in result.messages] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert result.messages[-1]["content"] == "pending question"


# ---------------------------------------------------------------- compression
def test_older_turns_summarized_when_budget_exceeded(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=400, keep_recent=2)
    conv = Conversation()
    add_turns(conv, 8, words=30)
    conv.add_user_message("latest question")
    before = mgr.measure(conv).total
    assert before > 400

    result = mgr.prepare_context(conv)

    assert result.ok and result.compressed
    assert fake_provider.summarize_calls, "provider must be asked to summarize"
    assert result.usage.total <= 400
    assert result.usage.total < before
    assert conv.stats.compression_events >= 1
    assert conv.stats.turns_summarized >= 1
    assert conv.summary
    assert SUMMARY_HEADER in result.messages[0]["content"]  # summary is a separate block in the system message


def test_recent_turns_are_preserved_during_compression(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=400, keep_recent=2)
    conv = Conversation()
    add_turns(conv, 8, words=30)
    conv.add_user_message("latest question")
    result = mgr.prepare_context(conv)
    assert result.ok

    active = [m.content for m in conv.active_messages()]
    # the two most recent completed turns (6 and 7) and the pending message are still verbatim
    assert any(c.startswith("Topic 6") for c in active)
    assert any(c.startswith("Topic 7") for c in active)
    assert active[-1] == "latest question"
    # the oldest turn was folded into memory
    summarized = [m.content for m in conv.summarized_messages()]
    assert any(c.startswith("Topic 0") for c in summarized)
    # and none of the summarized raw messages are re-sent to the model
    sent = " ".join(m["content"] for m in result.messages[1:])
    assert "Topic 0:" not in sent
    # transcript still complete for the UI
    assert len(conv.messages) == 8 * 2 + 1


def test_existing_summary_is_merged_with_new_one(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=400, keep_recent=1)
    conv = Conversation()
    conv.summary = "- [Facts] The user's dog is called Rex."
    add_turns(conv, 8, words=30)
    conv.add_user_message("go on")
    result = mgr.prepare_context(conv)
    assert result.ok and result.compressed
    # the provider received the old summary to merge, not just the raw turns
    assert fake_provider.summarize_calls[0]["existing"] == "- [Facts] The user's dog is called Rex."
    assert "dog is called Rex" in conv.summary  # old information survives
    assert "Topic 0" in conv.summary  # new information was added


def test_important_information_is_preserved_by_summary(counter):
    provider = OfflineProvider(counter)
    mgr = make_manager(counter, provider, budget=500, keep_recent=1)
    conv = Conversation()
    conv.add_user_message("My name is Sara. I prefer short answers. The deadline is Friday. Remind me to email Omar.")
    conv.add_assistant_message("Noted. " + "filler " * 60)
    add_turns(conv, 6, words=35, prefix="Filler")
    conv.add_user_message("What next?")
    result = mgr.prepare_context(conv)
    assert result.ok and result.compressed
    s = conv.summary.lower()
    assert "sara" in s  # fact
    assert "short answers" in s  # preference
    assert "friday" in s  # constraint
    assert "email omar" in s  # unfinished task


def test_offline_summary_merges_old_and_new_bullets(counter):
    provider = OfflineProvider(counter)
    old = "- [Facts] My name is Sara."
    new_turns = [
        {"role": "user", "content": "I live in Lahore."},
        {"role": "assistant", "content": "Nice city."},
    ]
    merged = provider.summarize(old, new_turns, max_tokens=300)
    assert "My name is Sara." in merged and "I live in Lahore." in merged
    # re-running with the same turns must not duplicate information
    again = provider.summarize(merged, new_turns, max_tokens=300)
    assert again.count("I live in Lahore.") == 1


def test_repeated_compression_until_within_budget(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=350, keep_recent=1)
    conv = Conversation()
    add_turns(conv, 15, words=40)
    conv.add_user_message("final")
    result = mgr.prepare_context(conv)
    assert result.ok
    assert result.usage.total <= 350
    assert conv.stats.turns_summarized >= 10


# ---------------------------------------------------------------- failure handling
def test_failed_summarization_never_discards_turns(counter):
    mgr = make_manager(counter, FailingProvider(), budget=400, keep_recent=2)
    conv = Conversation()
    add_turns(conv, 8, words=30)
    conv.add_user_message("latest question")
    snapshot = [(m.role, m.content, m.summarized) for m in conv.messages]

    result = mgr.prepare_context(conv)

    assert not result.ok
    assert result.messages == []  # nothing may be sent to the model
    assert "NOT sent" in result.error and "summarization failed" in result.error.lower()
    assert [(m.role, m.content, m.summarized) for m in conv.messages] == snapshot
    assert conv.summary == ""
    assert conv.stats.compression_events == 0


def test_empty_summary_from_provider_is_treated_as_failure(counter, fake_provider):
    fake_provider.summarize = lambda *a, **k: "   "
    mgr = make_manager(counter, fake_provider, budget=400)
    conv = Conversation()
    add_turns(conv, 8, words=30)
    conv.add_user_message("q")
    result = mgr.prepare_context(conv)
    assert not result.ok
    assert not conv.summarized_messages() and conv.summary == ""


# ---------------------------------------------------------------- budget safety
def test_context_stays_within_budget(counter, fake_provider):
    budget = 600
    mgr = make_manager(counter, fake_provider, budget=budget, keep_recent=2)
    conv = Conversation()
    for i in range(20):
        conv.add_user_message(f"Question {i}: " + "word " * 25)
        result = mgr.prepare_context(conv)
        assert result.ok, result.error
        assert result.usage.total <= budget
        assert mgr.measure(conv).total == result.usage.total
        # independent re-count of what is actually sent (allow tiny rounding from the estimator)
        assert counter.count_messages(result.messages) <= budget + 3
        conv.add_assistant_message("Reply " + "word " * 25)


def test_single_oversized_message_is_rejected_safely(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=500)
    conv = Conversation()
    add_turns(conv, 2, words=10)
    conv.add_user_message("huge " * 5000)  # ~5000+ tokens, can never fit in 500
    result = mgr.prepare_context(conv)
    assert not result.ok
    assert result.messages == []
    assert "NOT sent" in result.error and "budget" in result.error.lower()
    # earlier turns were compressed (not lost) while trying; every message is still in the transcript
    assert len(conv.messages) == 2 * 2 + 1
    # the UI can hand the rejected text back
    assert conv.discard_pending_user_message().startswith("huge")


def test_impossible_budget_is_rejected_safely(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=10)  # smaller than the system prompt itself
    conv = Conversation()
    conv.add_user_message("hi")
    result = mgr.prepare_context(conv)
    assert not result.ok and result.messages == []
    assert fake_provider.summarize_calls == []  # nothing to summarize
    assert str(10) in result.error


# ---------------------------------------------------------------- offline workflow
def test_offline_mode_exercises_memory_workflow_without_credentials(counter):
    settings = load_settings(env={})  # no API key anywhere
    assert not settings.has_credentials
    provider, status = create_provider(settings, counter)
    assert provider.is_offline and status.mode == "offline"
    assert "not" in OfflineProvider.name.lower() or "no ai" in OfflineProvider.name.lower()

    mgr = make_manager(counter, provider, budget=500, keep_recent=2)
    conv = Conversation()
    conv.add_user_message("My name is Sara. I prefer short answers.")
    assert mgr.prepare_context(conv).ok
    conv.add_assistant_message(provider.chat(mgr.build_messages(conv), 100).text)

    for i in range(10):
        conv.add_user_message(f"Question {i}: please explain subject {i} in detail? " + "more " * 30)
        result = mgr.prepare_context(conv)
        assert result.ok, result.error
        assert result.usage.total <= 500
        reply = provider.chat(result.messages, 100)
        assert "Offline demo mode" in reply.text  # honestly labelled
        assert reply.input_tokens is None and not reply.exact_usage  # no fake "exact" usage
        conv.add_assistant_message(reply.text)

    assert conv.stats.compression_events >= 1  # real compression happened
    assert conv.summarized_messages()
    assert "sara" in conv.summary.lower()

    # memory really works: the name is recalled although the original message was summarized away
    conv.add_user_message("What is my name?")
    result = mgr.prepare_context(conv)
    assert result.ok
    answer = provider.chat(result.messages, 100).text
    assert "Sara" in answer


def test_system_prompt_constant_is_used(counter, fake_provider):
    mgr = make_manager(counter, fake_provider, budget=1000)
    conv = Conversation()
    conv.add_user_message("hi")
    assert mgr.prepare_context(conv).messages[0]["content"].startswith(DEFAULT_SYSTEM_PROMPT)
