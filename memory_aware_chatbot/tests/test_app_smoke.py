"""End-to-end smoke test of the real Streamlit script using Streamlit's AppTest (offline mode)."""

from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    for var in ("OPENAI_API_KEY", "OFFLINE_MODE", "TOKEN_BUDGET"):
        monkeypatch.delenv(var, raising=False)
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def _metrics(at):
    return {m.label: m.value for m in at.metric}


def test_app_starts_in_offline_mode_without_key(app):
    assert not app.exception
    assert _metrics(app)["Token budget"] == "4,000"
    assert any("Offline demo mode" in w.value for w in app.warning)


def test_dashboard_updates_compression_and_reset(app):
    app.slider[0].set_value(500).run()
    for i in range(7):
        app.chat_input[0].set_value(f"My name is Sara. Question {i}: explain topic {i}? " + "detail " * 40).run()
        assert not app.exception

    conv = app.session_state["conv"]
    m = _metrics(app)
    assert conv.stats.compression_events >= 1
    assert conv.summarized_messages()
    assert int(m["Total input context"].replace(",", "")) <= 500
    assert m["Compressions / turns summarized"].split(" / ")[0] != "0"
    assert m["Last response: input tokens"] == "n/a"  # offline: nothing is passed off as API-exact

    # reset logic (the dialog button calls exactly this function's effects)
    conv.reset()
    app.run()
    assert not app.exception
    assert _metrics(app)["Compressions / turns summarized"] == "0 / 0"
    assert _metrics(app)["Memory summary"] == "0"
