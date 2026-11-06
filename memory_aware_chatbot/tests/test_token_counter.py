from src.token_counter import (
    PER_MESSAGE_OVERHEAD,
    REPLY_PRIMING_TOKENS,
    TokenCounter,
    estimate_tokens_fallback,
)


def test_counts_are_sensible_and_nonnegative(counter):
    assert counter.count_text("") == 0
    assert counter.count_text("hello") >= 1
    short = counter.count_text("hello world")
    long = counter.count_text("hello world " * 50)
    assert 0 < short < long
    assert counter.count_message({"role": "user", "content": ""}) >= PER_MESSAGE_OVERHEAD


def test_empty_message_list_only_costs_priming(counter):
    assert counter.count_messages([]) == REPLY_PRIMING_TOKENS


def test_message_overhead_is_added(counter):
    msg = {"role": "user", "content": "hi there"}
    assert counter.count_message(msg) > counter.count_text("hi there")
    assert counter.count_messages([msg, msg]) == 2 * counter.count_message(msg) + REPLY_PRIMING_TOKENS


def test_fallback_handles_non_ascii_conservatively():
    # non-ASCII chars cost about one token each, so the estimate must not collapse to chars/4
    assert estimate_tokens_fallback("日本語のテキスト") >= 8
    assert estimate_tokens_fallback("a" * 40) == 10


def test_fallback_label_never_claims_exactness(counter):
    assert not counter.using_tiktoken
    assert "estimate" in counter.method_label.lower()


def test_tiktoken_path_used_when_encoding_loads(monkeypatch):
    """Exercise the tiktoken branch with a stub encoding (no network needed)."""

    class StubEncoding:
        def encode(self, text, disallowed_special=()):
            return text.split()  # one 'token' per word

    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: StubEncoding())
    c = TokenCounter("cl100k_base")
    assert c.using_tiktoken
    assert c.count_text("one two three") == 3
    assert "cl100k_base" in c.method_label and "estimate" in c.method_label.lower()


def test_unknown_encoding_falls_back_gracefully(monkeypatch):
    import tiktoken

    def boom(name):
        raise ValueError("unknown encoding")

    monkeypatch.setattr(tiktoken, "get_encoding", boom)
    c = TokenCounter("does-not-exist")
    assert not c.using_tiktoken
    assert c.load_error
    assert c.count_text("some text here") > 0
