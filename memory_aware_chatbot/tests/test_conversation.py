from src.conversation import Conversation


def test_empty_conversation_is_handled():
    conv = Conversation()
    assert conv.is_empty
    assert conv.active_messages() == []
    assert conv.completed_turns() == []
    assert conv.summary == ""
    assert conv.stats.compression_events == 0


def test_messages_are_stored_in_order():
    conv = Conversation()
    conv.add_user_message("one")
    conv.add_assistant_message("two")
    conv.add_user_message("three")
    assert [(m.role, m.content) for m in conv.messages] == [
        ("user", "one"), ("assistant", "two"), ("user", "three"),
    ]
    assert not conv.is_empty


def test_completed_turns_exclude_pending_user_message():
    conv = Conversation()
    conv.add_user_message("q1")
    conv.add_assistant_message("a1")
    conv.add_user_message("q2")  # unanswered
    turns = conv.completed_turns()
    assert len(turns) == 1
    assert [m.content for m in turns[0]] == ["q1", "a1"]


def test_apply_summary_flags_messages_but_keeps_transcript():
    conv = Conversation()
    conv.add_user_message("q1")
    conv.add_assistant_message("a1")
    conv.add_user_message("q2")
    conv.add_assistant_message("a2")
    first_turn = conv.completed_turns()[0]
    conv.apply_summary("- q1 happened", first_turn)
    assert conv.summary == "- q1 happened"
    assert len(conv.messages) == 4  # nothing deleted from the transcript
    assert [m.content for m in conv.summarized_messages()] == ["q1", "a1"]
    assert [m.content for m in conv.active_messages()] == ["q2", "a2"]


def test_discard_pending_user_message():
    conv = Conversation()
    conv.add_user_message("hello")
    assert conv.discard_pending_user_message() == "hello"
    assert conv.messages == []
    conv.add_user_message("q")
    conv.add_assistant_message("a")
    assert conv.discard_pending_user_message() is None  # only drops an unanswered user message
    assert len(conv.messages) == 2


def test_reset_clears_everything():
    conv = Conversation()
    conv.add_user_message("q1")
    conv.add_assistant_message("a1")
    conv.apply_summary("- memory", conv.completed_turns()[0])
    conv.stats.compression_events = 3
    conv.stats.turns_summarized = 5
    conv.last_error = "oops"
    conv.reset()
    assert conv.messages == []
    assert conv.summary == ""
    assert conv.stats.compression_events == 0 and conv.stats.turns_summarized == 0
    assert conv.stats.log == []
    assert conv.last_api_usage is None and conv.last_error == ""
    assert conv.is_empty
