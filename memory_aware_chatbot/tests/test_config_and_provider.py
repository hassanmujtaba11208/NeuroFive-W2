import pytest

from src.config import (
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    Settings,
    clamp_budget,
    load_settings,
)
from src.llm_provider import (
    OfflineProvider,
    OpenAICompatibleProvider,
    ProviderError,
    create_provider,
)


def test_default_budget_is_4000():
    assert DEFAULT_TOKEN_BUDGET == 4000
    assert load_settings(env={}).token_budget == 4000


@pytest.mark.parametrize("raw,expected", [
    ("abc", DEFAULT_TOKEN_BUDGET), (None, DEFAULT_TOKEN_BUDGET),
    (10, MIN_TOKEN_BUDGET), (10**6, MAX_TOKEN_BUDGET), ("2500", 2500),
])
def test_clamp_budget(raw, expected):
    assert clamp_budget(raw) == expected


def test_placeholder_key_is_not_a_credential():
    assert not Settings(api_key="your-api-key-here").has_credentials
    assert not Settings(api_key="").has_credentials
    assert Settings(api_key="sk-real-looking-key-123456").has_credentials


def test_masked_key_never_reveals_full_secret():
    s = Settings(api_key="sk-abcdefghijklmnop")
    assert "abcdefghijkl" not in s.masked_key


def test_secrets_take_priority_over_environment():
    s = load_settings(env={"LLM_MODEL": "from-env", "OPENAI_API_KEY": "env-key-123456"},
                      secrets={"LLM_MODEL": "from-secrets"})
    assert s.model == "from-secrets"
    assert s.api_key == "env-key-123456"


def test_missing_key_falls_back_to_labelled_offline_provider(counter):
    provider, status = create_provider(load_settings(env={}), counter)
    assert isinstance(provider, OfflineProvider)
    assert status.mode == "offline" and "no API key" in status.label


def test_forced_offline_even_with_key(counter):
    provider, status = create_provider(
        load_settings(env={"OPENAI_API_KEY": "sk-real-looking-key-123456", "OFFLINE_MODE": "true"}), counter)
    assert provider.is_offline and status.mode == "offline"


def test_key_selects_live_provider(counter):
    provider, status = create_provider(
        load_settings(env={"OPENAI_API_KEY": "sk-real-looking-key-123456"}), counter)
    assert isinstance(provider, OpenAICompatibleProvider) and status.mode == "live"
    assert "sk-real-looking-key-123456" not in status.detail  # key is masked in the UI


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _client(completions):
    class C:
        class chat:  # noqa: N801
            pass
    c = C()
    c.chat.completions = completions
    return c


def _response(text="hi", pt=11, ct=3):
    class R:
        class choices_item:  # noqa: N801
            class message:  # noqa: N801
                content = text
        choices = [choices_item]

        class usage:  # noqa: N801
            prompt_tokens = pt
            completion_tokens = ct
    return R


def test_openai_provider_reports_api_usage_as_exact():
    p = OpenAICompatibleProvider("k", "http://x", "m")
    comp = _FakeCompletions(_response("hello", 42, 7))
    p._client = _client(comp)
    result = p.chat([{"role": "user", "content": "q"}], 100)
    assert result.text == "hello"
    assert (result.input_tokens, result.output_tokens, result.exact_usage) == (42, 7, True)
    assert comp.kwargs["max_tokens"] == 100


def test_openai_provider_wraps_errors():
    p = OpenAICompatibleProvider("k", "http://x", "m")
    p._client = _client(_FakeCompletions(error=RuntimeError("network down")))
    with pytest.raises(ProviderError):
        p.chat([{"role": "user", "content": "q"}], 100)


def test_live_summarize_uses_same_provider_and_rejects_empty():
    p = OpenAICompatibleProvider("k", "http://x", "m")
    comp = _FakeCompletions(_response("- [Facts] x"))
    p._client = _client(comp)
    out = p.summarize("old", [{"role": "user", "content": "new info"}], 200)
    assert out == "- [Facts] x"
    prompt = comp.kwargs["messages"][1]["content"]
    assert "old" in prompt and "new info" in prompt  # existing summary + turns are both sent
    p._client = _client(_FakeCompletions(_response("")))
    with pytest.raises(ProviderError):
        p.summarize("old", [{"role": "user", "content": "x"}], 200)
