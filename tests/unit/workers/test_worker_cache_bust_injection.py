# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models.dataset_models import Conversation, Text, Turn
from aiperf.credit.structs import Credit
from aiperf.workers.session_manager import UserSession
from aiperf.workers.worker import (
    _apply_cache_bust,
    _apply_cache_bust_to_system_message,
    _find_first_system_message,
    _find_first_user_turn,
    _inject_marker_into_first_user_text,
    _inject_marker_into_first_user_turn,
    _inject_marker_into_raw_messages,
)

_PREFIX_MARKER = "[rid:abc123def456]\n\n"
_SUFFIX_MARKER = "\n\n[rid:abc123def456]"


def _make_session(
    raw_messages: list[dict] | None, *, num_turns: int = 1
) -> UserSession:
    """Build a UserSession whose ``turn_list[-1].raw_messages`` is the given
    list, simulating the post-``advance_turn`` state on the dispatch path."""
    turn = Turn(raw_messages=raw_messages)
    conversation = Conversation(session_id="conv_test", turns=[turn] * num_turns)
    session = UserSession(
        x_correlation_id="xcorr_test",
        num_turns=num_turns,
        conversation=conversation,
        turn_list=[turn],
    )
    return session


def _make_credit(
    *,
    target: CacheBustTarget,
    marker: str | None,
    turn_index: int = 0,
    num_turns: int = 1,
) -> Credit:
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id="conv_test",
        x_correlation_id="xcorr_test",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        cache_bust_marker=marker,
        cache_bust_target=target,
    )


def test_apply_system_message_none_target_passthrough():
    out = _apply_cache_bust_to_system_message("hello", "", CacheBustTarget.NONE)
    assert out == "hello"


def test_apply_system_message_prefix():
    out = _apply_cache_bust_to_system_message(
        "hello", _PREFIX_MARKER, CacheBustTarget.SYSTEM_PREFIX
    )
    assert out == _PREFIX_MARKER + "hello"


def test_apply_system_message_suffix():
    out = _apply_cache_bust_to_system_message(
        "hello", _SUFFIX_MARKER, CacheBustTarget.SYSTEM_SUFFIX
    )
    assert out == "hello" + _SUFFIX_MARKER


def test_apply_with_none_system_message_returns_none_for_caller_fallback():
    out = _apply_cache_bust_to_system_message(
        None, _PREFIX_MARKER, CacheBustTarget.SYSTEM_PREFIX
    )
    assert out is None


def test_inject_marker_into_raw_messages_prefix():
    raw = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == _PREFIX_MARKER + "you are helpful"


def test_inject_marker_into_raw_messages_suffix():
    raw = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    _inject_marker_into_raw_messages(raw, _SUFFIX_MARKER, is_prefix=False)
    assert raw[0]["content"] == "you are helpful" + _SUFFIX_MARKER


def test_inject_marker_no_system_role_is_noop():
    raw = [{"role": "user", "content": "hi"}]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == "hi"


def test_inject_marker_empty_raw_is_noop():
    raw: list[dict] = []
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw == []


def test_inject_first_user_turn_prefix_with_system_present():
    raw = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[1]["content"] == _PREFIX_MARKER + "hi"


def test_inject_first_user_turn_suffix_user_only():
    raw = [{"role": "user", "content": "hi"}]
    _inject_marker_into_first_user_turn(raw, _SUFFIX_MARKER, is_prefix=False)
    assert raw[0]["content"] == "hi" + _SUFFIX_MARKER


def test_inject_first_user_turn_empty_raw_is_noop():
    raw: list[dict] = []
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw == []


# =============================================================================
# Dispatch tests for _apply_cache_bust — covers the SYSTEM_*-fallback-to-
# FIRST_TURN_* path that fixes the silent-drop bug for traces lacking a
# system message.
# =============================================================================


def test_system_prefix_falls_back_to_first_user_turn_when_no_system():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX, marker=_PREFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"


def test_system_suffix_falls_back_to_first_user_turn_when_no_system():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_SUFFIX, marker=_SUFFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].raw_messages[0]["content"] == "hi" + _SUFFIX_MARKER


def test_system_prefix_uses_existing_raw_system_role_when_no_conversation_system():
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX, marker=_PREFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    msgs = session.turn_list[-1].raw_messages
    assert msgs[0]["content"] == _PREFIX_MARKER + "sys"
    # User turn must be untouched.
    assert msgs[1]["content"] == "hi"


def test_system_prefix_fallback_no_op_on_turn_index_gt_zero():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw, num_turns=2)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].raw_messages[0]["content"] == "hi"


def test_first_turn_prefix_unaffected_by_system_message_presence():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX, marker=_PREFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message="sys")

    # System message returned unchanged.
    assert out == "sys"
    # First user turn carries the marker.
    assert session.turn_list[-1].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"


def test_target_none_passes_through_unchanged():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(target=CacheBustTarget.NONE, marker=None, turn_index=0)

    out = _apply_cache_bust(session, credit, system_message="sys")

    assert out == "sys"
    assert session.turn_list[-1].raw_messages[0]["content"] == "hi"


def test_system_prefix_with_conversation_system_message_returns_modified_string():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX, marker=_PREFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message="sys")

    assert out == _PREFIX_MARKER + "sys"
    # Raw messages must NOT be mutated when conversation system_message exists.
    assert session.turn_list[-1].raw_messages[0]["content"] == "hi"


# =============================================================================
# Synthetic-Turn (raw_messages=None) injection — _inject_marker_into_first_user_text
# =============================================================================


def _make_synthetic_session(turn: Turn, *, num_turns: int = 1) -> UserSession:
    """Build a UserSession whose ``turn_list[-1]`` is a synthetic Turn (no raw_messages)."""
    conversation = Conversation(session_id="conv_test", turns=[turn] * num_turns)
    return UserSession(
        x_correlation_id="xcorr_test",
        num_turns=num_turns,
        conversation=conversation,
        turn_list=[turn],
    )


def test_inject_first_user_text_prefix_mutates_first_content():
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert turn.texts[0].contents[0] == _PREFIX_MARKER + "hello"


def test_inject_first_user_text_suffix_appends():
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    _inject_marker_into_first_user_text(turn, _SUFFIX_MARKER, is_prefix=False)
    assert turn.texts[0].contents[0] == "hello" + _SUFFIX_MARKER


def test_inject_first_user_text_empty_texts_creates_marker_text():
    turn = Turn(raw_messages=None, texts=[])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert len(turn.texts) == 1
    assert turn.texts[0].contents == [_PREFIX_MARKER.strip()]


def test_inject_first_user_text_empty_contents_seeds_marker():
    turn = Turn(raw_messages=None, texts=[Text(contents=[])])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert turn.texts[0].contents == [_PREFIX_MARKER.strip()]


def test_inject_first_user_text_empty_marker_is_noop():
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    _inject_marker_into_first_user_text(turn, "", is_prefix=True)
    assert turn.texts[0].contents[0] == "hello"


def test_first_turn_prefix_synthetic_turn_with_texts_mutated():
    """FIRST_TURN_PREFIX on synthetic Turn (raw_messages=None) mutates Text.contents[0]."""
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    session = _make_synthetic_session(turn)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].texts[0].contents[0] == _PREFIX_MARKER + "hello"


def test_first_turn_suffix_synthetic_turn_appends():
    """FIRST_TURN_SUFFIX on synthetic Turn appends marker to Text.contents[0]."""
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    session = _make_synthetic_session(turn)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_SUFFIX,
        marker=_SUFFIX_MARKER,
        turn_index=0,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].texts[0].contents[0] == "hello" + _SUFFIX_MARKER


def test_first_turn_prefix_synthetic_turn_empty_texts_creates_marker_text():
    """FIRST_TURN_PREFIX on synthetic Turn with no texts seeds a marker-only text entry."""
    turn = Turn(raw_messages=None, texts=[])
    session = _make_synthetic_session(turn)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].texts == [Text(contents=[_PREFIX_MARKER.strip()])]


def test_system_prefix_fallback_to_synthetic_text_when_no_raw_and_no_system_message():
    """SYSTEM_PREFIX fallback path mutates Turn.texts when raw_messages is None."""
    turn = Turn(raw_messages=None, texts=[Text(contents=["hi"])])
    session = _make_synthetic_session(turn)
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[-1].texts[0].contents[0].startswith(_PREFIX_MARKER)
    assert session.turn_list[-1].texts[0].contents[0] == _PREFIX_MARKER + "hi"


# =============================================================================
# Multimodal raw_messages content (list-of-parts)
# =============================================================================
# OpenAI multimodal shape: content=[{"type":"text","text":"..."}, {"type":"image_url",...}].
# Marker becomes a new {"type":"text","text":marker} part at the start (prefix)
# or end (suffix). Pre-fix this silently bailed and dropped the marker.


def test_inject_into_raw_messages_multimodal_prefix():
    raw = [
        {"role": "system", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": "hello"},
    ]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"] == [
        {"type": "text", "text": _PREFIX_MARKER.strip()},
        {"type": "text", "text": "hi"},
    ]


def test_inject_into_raw_messages_multimodal_suffix():
    raw = [
        {"role": "system", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": "hello"},
    ]
    _inject_marker_into_raw_messages(raw, _SUFFIX_MARKER, is_prefix=False)

    assert raw[0]["content"] == [
        {"type": "text", "text": "hi"},
        {"type": "text", "text": _SUFFIX_MARKER.strip()},
    ]


def test_inject_into_raw_messages_multimodal_with_image_part_prefix():
    raw = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        },
    ]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"][0] == {
        "type": "text",
        "text": _PREFIX_MARKER.strip(),
    }
    assert raw[0]["content"][1] == {"type": "text", "text": "describe"}
    assert raw[0]["content"][2]["type"] == "image_url"


def test_inject_into_raw_messages_unexpected_content_type_logs_and_bails(caplog):
    raw = [{"role": "system", "content": 12345}]

    with caplog.at_level("WARNING"):
        _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"] == 12345
    assert any("cache-bust" in rec.message for rec in caplog.records)
    assert any("int" in rec.message for rec in caplog.records)


def test_inject_into_first_user_turn_multimodal_prefix():
    raw = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "what is this"}],
        },
    ]
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"] == [
        {"type": "text", "text": _PREFIX_MARKER.strip()},
        {"type": "text", "text": "what is this"},
    ]


def test_inject_into_first_user_turn_multimodal_suffix():
    raw = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "what is this"}],
        },
    ]
    _inject_marker_into_first_user_turn(raw, _SUFFIX_MARKER, is_prefix=False)

    assert raw[0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "text", "text": _SUFFIX_MARKER.strip()},
    ]


def test_inject_into_first_user_turn_unexpected_content_type_logs_and_bails(caplog):
    raw = [{"role": "user", "content": 99999}]

    with caplog.at_level("WARNING"):
        _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"] == 99999
    assert any("cache-bust" in rec.message for rec in caplog.records)


# =============================================================================
# Delta-mode (DELTAS_WITH_RESPONSES) helper + dispatch coverage
# =============================================================================
# Under DELTAS_WITH_RESPONSES the session_manager appends each turn's delta
# to ``turn_list``. The system role lives in ``turn_list[0].raw_messages[0]``;
# subsequent turns' raw_messages start with the prior assistant response and
# the new user prompt. The lookup must walk forward, NOT index ``[-1]``.


def _make_delta_session(turns_raw: list[list[dict] | None]) -> UserSession:
    """Build a UserSession whose ``turn_list`` is an accumulating delta list.

    Each entry in ``turns_raw`` becomes a Turn's raw_messages. The conversation
    declares ``num_turns == len(turns_raw)`` so this represents the post-
    ``advance_turn`` state at the final turn under DELTAS_WITH_RESPONSES.
    """
    turns = [Turn(raw_messages=raw) for raw in turns_raw]
    conversation = Conversation(session_id="conv_test", turns=list(turns))
    return UserSession(
        x_correlation_id="xcorr_test",
        num_turns=len(turns),
        conversation=conversation,
        turn_list=list(turns),
    )


def test_find_first_system_message_in_delta_turn_list_picks_turn_0():
    """In delta mode, system lives in turn_list[0]; later deltas start with assistant."""
    turn_0 = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    turn_1_delta = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session([turn_0, turn_1_delta])

    raw = _find_first_system_message(session.turn_list)

    assert raw is session.turn_list[0].raw_messages
    assert raw[0]["role"] == "system"


def test_find_first_system_message_no_system_returns_none():
    turn_0 = [{"role": "user", "content": "hi"}]
    turn_1 = [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    session = _make_delta_session([turn_0, turn_1])

    assert _find_first_system_message(session.turn_list) is None


def test_find_first_user_turn_skips_leading_system_only_delta():
    """A leading delta with only a system role must NOT be returned by user-turn lookup."""
    turn_0_system_only = [{"role": "system", "content": "rules"}]
    turn_1_user = [{"role": "user", "content": "hi"}]
    session = _make_delta_session([turn_0_system_only, turn_1_user])

    user_turn = _find_first_user_turn(session.turn_list)

    assert user_turn is session.turn_list[1]


def test_find_first_user_turn_picks_turn_with_user_role():
    turn_0 = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    turn_1 = [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    session = _make_delta_session([turn_0, turn_1])

    assert _find_first_user_turn(session.turn_list) is session.turn_list[0]


def test_apply_system_prefix_under_deltas_injects_into_turn_0_not_last():
    """The bug we are fixing: under deltas, system_prefix must mutate turn_list[0],
    NOT turn_list[-1] (whose raw_messages start with an assistant role)."""
    turn_0 = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    turn_1_delta = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session([turn_0, turn_1_delta])
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    # System message in turn_list[0] is mutated, not turn_list[-1].
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "rules"
    # Turn 1's delta is untouched (still starts with assistant, no marker).
    assert session.turn_list[1].raw_messages[0]["role"] == "assistant"
    assert session.turn_list[1].raw_messages[0]["content"] == "hello"


def test_apply_system_suffix_under_deltas_injects_into_turn_0_system():
    turn_0 = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    turn_1_delta = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session([turn_0, turn_1_delta])
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_SUFFIX,
        marker=_SUFFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == "rules" + _SUFFIX_MARKER


def test_apply_first_turn_prefix_under_deltas_injects_into_turn_0_user_role():
    """FIRST_TURN_PREFIX with turn_index==0 must target turn_list[0]'s user role,
    not the latest delta's user role."""
    turn_0 = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    turn_1_delta = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session([turn_0, turn_1_delta])
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    # First user message in turn 0 mutated; turn 1's user is untouched.
    assert session.turn_list[0].raw_messages[1]["content"] == _PREFIX_MARKER + "hi"
    assert session.turn_list[1].raw_messages[1]["content"] == "follow up"


def test_apply_system_prefix_no_system_under_deltas_falls_back_to_turn_0_user():
    """No system anywhere + delta-mode turn_list -> fallback marks turn 0 user only."""
    turn_0 = [{"role": "user", "content": "hi"}]
    turn_1_delta = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session([turn_0, turn_1_delta])
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"
    assert session.turn_list[1].raw_messages[1]["content"] == "follow up"
