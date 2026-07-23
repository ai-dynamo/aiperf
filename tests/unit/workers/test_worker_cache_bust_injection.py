# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models.dataset_models import Conversation, Text, Turn
from aiperf.credit.structs import Credit
from aiperf.workers.session_manager import UserSession, UserSessionManager
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
    """Build a UserSession whose ``turn_list[-1].raw_messages`` is the given"""
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


def test_inject_marker_into_raw_messages_prefix_is_idempotent():
    raw = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]

    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"] == _PREFIX_MARKER + "you are helpful"


def test_inject_marker_into_raw_messages_suffix():
    raw = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    _inject_marker_into_raw_messages(raw, _SUFFIX_MARKER, is_prefix=False)
    assert raw[0]["content"] == "you are helpful" + _SUFFIX_MARKER


@pytest.mark.parametrize(
    ("marker", "is_prefix", "expected"),
    [
        param(_PREFIX_MARKER, True, _PREFIX_MARKER + "you are helpful", id="prefix"),
        param(_SUFFIX_MARKER, False, "you are helpful" + _SUFFIX_MARKER, id="suffix"),
    ],
)  # fmt: skip
def test_inject_marker_into_raw_messages_idempotent(marker, is_prefix, expected):
    """In DELTAS mode turn_list[0] is a single shared object re-visited every"""
    raw = [{"role": "system", "content": "you are helpful"}]
    _inject_marker_into_raw_messages(raw, marker, is_prefix=is_prefix)
    _inject_marker_into_raw_messages(raw, marker, is_prefix=is_prefix)
    assert raw[0]["content"] == expected


def test_inject_marker_into_raw_messages_multimodal_idempotent():
    raw = [{"role": "system", "content": [{"type": "text", "text": "hi"}]}]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == [
        {"type": "text", "text": _PREFIX_MARKER.strip()},
        {"type": "text", "text": "hi"},
    ]


def test_inject_marker_no_system_role_is_noop():
    raw = [{"role": "user", "content": "hi"}]
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == "hi"


def test_inject_first_user_turn_idempotent_prefix():
    """Injection is unconditional per credit (seeded resume marks turn 0 every"""
    raw = [{"role": "user", "content": "hi"}]
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == _PREFIX_MARKER + "hi"


def test_inject_first_user_turn_idempotent_multimodal():
    raw = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[0]["content"] == [
        {"type": "text", "text": _PREFIX_MARKER.strip()},
        {"type": "text", "text": "hi"},
    ]


def test_inject_first_user_text_idempotent():
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert turn.texts[0].contents[0] == _PREFIX_MARKER + "hello"


def test_inject_marker_empty_raw_is_noop():
    raw: list[dict] = []
    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw == []


def test_inject_first_user_turn_prefix_with_system_present():
    raw = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
    assert raw[1]["content"] == _PREFIX_MARKER + "hi"


def test_inject_first_user_turn_prefix_is_idempotent():
    raw = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)
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
    assert msgs[1]["content"] == "hi"


def test_system_prefix_fallback_marks_first_user_on_turn_index_gt_zero():
    """SYSTEM_PREFIX with no system anywhere falls back to the first user turn,"""
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
    assert session.turn_list[-1].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"


def test_first_turn_prefix_unaffected_by_system_message_presence():
    raw = [{"role": "user", "content": "hi"}]
    session = _make_session(raw)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX, marker=_PREFIX_MARKER, turn_index=0
    )

    out = _apply_cache_bust(session, credit, system_message="sys")

    assert out == "sys"
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
    assert session.turn_list[-1].raw_messages[0]["content"] == "hi"


def _make_synthetic_session(turn: Turn, *, num_turns: int = 1) -> UserSession:
    """Build a UserSession whose ``turn_list[-1]`` is a synthetic Turn (no raw_messages)."""
    conversation = Conversation(session_id="conv_test", turns=[turn] * num_turns)
    return UserSession(
        x_correlation_id="xcorr_test",
        num_turns=num_turns,
        conversation=conversation,
        turn_list=[turn],
    )


@pytest.mark.parametrize(
    ("marker", "is_prefix", "expected"),
    [
        param(_PREFIX_MARKER, True, _PREFIX_MARKER + "hello", id="prefix"),
        param(_SUFFIX_MARKER, False, "hello" + _SUFFIX_MARKER, id="suffix"),
    ],
)  # fmt: skip
def test_inject_first_user_text_mutates_first_content(marker, is_prefix, expected):
    turn = Turn(raw_messages=None, texts=[Text(contents=["hello"])])
    _inject_marker_into_first_user_text(turn, marker, is_prefix=is_prefix)
    assert turn.texts[0].contents[0] == expected


def test_inject_first_user_text_empty_texts_creates_marker_text():
    turn = Turn(raw_messages=None, texts=[])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert len(turn.texts) == 1
    assert turn.texts[0].contents == [_PREFIX_MARKER]


def test_inject_first_user_text_empty_contents_seeds_marker():
    turn = Turn(raw_messages=None, texts=[Text(contents=[])])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert turn.texts[0].contents == [_PREFIX_MARKER]


def test_inject_first_user_text_empty_seed_is_idempotent():
    """Empty-text seed must store the full marker so a second inject does not stack."""
    turn = Turn(raw_messages=None, texts=[])
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    _inject_marker_into_first_user_text(turn, _PREFIX_MARKER, is_prefix=True)
    assert turn.texts[0].contents == [_PREFIX_MARKER]
    assert turn.texts[0].contents[0].count("[rid:") == 1


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
    assert session.turn_list[-1].texts == [Text(contents=[_PREFIX_MARKER])]


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


def _make_delta_session(turns_raw: list[list[dict] | None]) -> UserSession:
    """Build a UserSession whose ``turn_list`` is an accumulating delta list."""
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
    """The bug we are fixing: under deltas, system_prefix must mutate turn_list[0],"""
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
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "rules"
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
    """FIRST_TURN_PREFIX with turn_index==0 must target turn_list[0]'s user role,"""
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

    assert session.turn_list[0].raw_messages[1]["content"] == _PREFIX_MARKER + "hi"
    assert session.turn_list[1].raw_messages[1]["content"] == "follow up"


def test_apply_first_turn_prefix_under_deltas_mid_turn_marks_seeded_turn_0_once():
    """Agentic replay can start at turn_index>0 after seeding turns 0..k-1."""
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
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)
    _apply_cache_bust(session, credit, system_message=None)

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


def _make_delta_session_with_resets(
    turns_raw: list[list[dict] | None], reset_flags: list[bool]
) -> UserSession:
    """Like ``_make_delta_session`` but sets ``reset_context`` per turn."""
    turns = [
        Turn(raw_messages=raw, reset_context=reset)
        for raw, reset in zip(turns_raw, reset_flags, strict=True)
    ]
    conversation = Conversation(session_id="conv_test", turns=list(turns))
    return UserSession(
        x_correlation_id="xcorr_test",
        num_turns=len(turns),
        conversation=conversation,
        turn_list=list(turns),
    )


def test_first_turn_prefix_reapplied_on_reset_context_turn():
    """FIRST_TURN_PREFIX at turn_index > 0 must inject into the reset turn (the"""
    turn_0 = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    turn_1_reset = [
        {"role": "system", "content": "fresh rules"},
        {"role": "user", "content": "new prefix"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset], reset_flags=[False, True]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert (
        session.turn_list[1].raw_messages[1]["content"] == _PREFIX_MARKER + "new prefix"
    )
    assert session.turn_list[0].raw_messages[1]["content"] == "hi"


def test_first_turn_suffix_reapplied_on_reset_context_turn():
    turn_0 = [{"role": "user", "content": "hi"}]
    turn_1_reset = [{"role": "user", "content": "new prefix"}]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset], reset_flags=[False, True]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_SUFFIX,
        marker=_SUFFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert (
        session.turn_list[1].raw_messages[0]["content"] == "new prefix" + _SUFFIX_MARKER
    )


def test_first_turn_prefix_marks_prefix_turn_on_ordinary_later_turn():
    """A non-reset turn at index > 0 re-marks the shared turn-0 prefix"""
    turn_0 = [{"role": "user", "content": "hi"}]
    turn_1 = [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "follow up"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1], reset_flags=[False, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"
    assert session.turn_list[1].raw_messages[1]["content"] == "follow up"


def test_first_turn_prefix_reset_on_turn_zero_uses_turn_zero_path_once():
    """A reset flag on turn 0 still resolves through the turn-0 path and injects"""
    turn_0_reset = [{"role": "user", "content": "hi"}]
    session = _make_delta_session_with_resets([turn_0_reset], reset_flags=[True])
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
        num_turns=1,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "hi"


def test_first_turn_prefix_marks_seeded_turn_zero_on_resume():
    """FIRST_TURN_PREFIX at turn_index > 0 with no reset must mark the seeded"""
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1 = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
    ]
    turn_2 = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1, turn_2], reset_flags=[False, False, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=2,
        num_turns=3,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"
    assert session.turn_list[2].raw_messages[1]["content"] == "u2"


def test_first_turn_prefix_resume_then_next_turn_no_stacking():
    """The seeded turn 0 is shared across the session's turns; processing the"""
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1 = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1], reset_flags=[False, False]
    )

    _apply_cache_bust(
        session,
        _make_credit(
            target=CacheBustTarget.FIRST_TURN_PREFIX,
            marker=_PREFIX_MARKER,
            turn_index=1,
            num_turns=2,
        ),
        system_message=None,
    )
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"

    _apply_cache_bust(
        session,
        _make_credit(
            target=CacheBustTarget.FIRST_TURN_PREFIX,
            marker=_PREFIX_MARKER,
            turn_index=1,
            num_turns=2,
        ),
        system_message=None,
    )
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"


def test_system_prefix_subpath3_marks_seeded_turn_zero_on_resume():
    """SYSTEM_PREFIX with no system anywhere falls back to first-user; under a"""
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1 = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1], reset_flags=[False, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"


def _make_fork_child_session(
    turns: list[Turn], *, num_turns: int | None = None
) -> UserSession:
    conversation = Conversation(session_id="child", turns=list(turns))
    return UserSession(
        x_correlation_id="child_xcorr",
        num_turns=num_turns if num_turns is not None else len(turns),
        conversation=conversation,
        turn_list=list(turns),
        parent_correlation_id="parent_xcorr",
        branch_mode=ConversationBranchMode.FORK,
    )


def test_fork_child_first_turn_is_noop_inherits_parent_marker():
    parent_marked_t0 = Turn(
        raw_messages=[{"role": "user", "content": "[rid:PARENT00000]\n\nu0"}],
        reset_context=False,
    )
    child_turn = Turn(
        raw_messages=[
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
        ],
        reset_context=False,
    )
    session = _make_fork_child_session([parent_marked_t0, child_turn])
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker="[rid:CHILD000000]\n\n",
        turn_index=1,
        num_turns=2,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"


def test_fork_child_system_target_is_noop():
    parent_marked_sys = Turn(
        raw_messages=[
            {"role": "system", "content": "[rid:PARENT00000]\n\nS0"},
            {"role": "user", "content": "u0"},
        ],
        reset_context=False,
    )
    child_turn = Turn(
        raw_messages=[
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
        ],
        reset_context=False,
    )
    session = _make_fork_child_session([parent_marked_sys, child_turn])
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker="[rid:CHILD000000]\n\n",
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nS0"


def test_spawn_child_is_busted_normally():
    """SPAWN children start fresh (no shared parent turns), so they are busted"""
    t0 = Turn(raw_messages=[{"role": "user", "content": "u0"}], reset_context=False)
    conversation = Conversation(session_id="spawn", turns=[t0])
    session = UserSession(
        x_correlation_id="spawn_xcorr",
        num_turns=1,
        conversation=conversation,
        turn_list=[t0],
        parent_correlation_id="parent_xcorr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=0,
        num_turns=1,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"


def test_first_turn_prefix_marks_buried_reset_turn_not_discarded_turn_zero():
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1_reset = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "u1"},
    ]
    turn_2 = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset, turn_2], reset_flags=[False, True, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=2,
        num_turns=3,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[1]["content"] == _PREFIX_MARKER + "u1"
    assert session.turn_list[0].raw_messages[0]["content"] == "u0"
    assert session.turn_list[2].raw_messages[1]["content"] == "u2"


def test_system_prefix_marks_buried_reset_turn_system():
    turn_0 = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "u0"},
    ]
    turn_1_reset = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "u1"},
    ]
    turn_2 = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset, turn_2], reset_flags=[False, True, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=2,
        num_turns=3,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[0]["content"] == _PREFIX_MARKER + "S1"
    assert session.turn_list[0].raw_messages[0]["content"] == "S0"


def test_first_turn_prefix_marks_only_last_of_multiple_resets():
    """With two resets, only the last (the effective prefix) is marked."""
    turn_0_reset = [{"role": "user", "content": "u0"}]
    turn_1_reset = [{"role": "user", "content": "u1"}]
    turn_2 = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0_reset, turn_1_reset, turn_2], reset_flags=[True, True, False]
    )
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=2,
        num_turns=3,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[0]["content"] == _PREFIX_MARKER + "u1"
    assert session.turn_list[0].raw_messages[0]["content"] == "u0"


def test_system_prefix_reapplied_on_reset_turn_with_own_system():
    """Sub-path 2 under reset: the reset turn's own system message (the new wire"""
    turn_0 = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "u0"},
    ]
    turn_1_reset = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "u1"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset], reset_flags=[False, True]
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    out = _apply_cache_bust(session, credit, system_message=None)

    assert out is None
    assert session.turn_list[1].raw_messages[0]["content"] == _PREFIX_MARKER + "S1"
    assert session.turn_list[0].raw_messages[0]["content"] == "S0"


def test_system_suffix_reapplied_on_reset_turn_with_own_system():
    turn_0 = [{"role": "system", "content": "S0"}]
    turn_1_reset = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "u1"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset], reset_flags=[False, True]
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_SUFFIX,
        marker=_SUFFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[0]["content"] == "S1" + _SUFFIX_MARKER
    assert session.turn_list[0].raw_messages[0]["content"] == "S0"


def test_system_prefix_reset_no_system_falls_back_to_reset_turn_user():
    """Sub-path 3 under reset: no system anywhere, so the marker falls back to"""
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1_reset = [{"role": "user", "content": "u1"}]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset], reset_flags=[False, True]
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[0]["content"] == _PREFIX_MARKER + "u1"
    assert session.turn_list[0].raw_messages[0]["content"] == "u0"


def test_system_prefix_subpath2_no_stacking_across_delta_turns():
    """Sub-path 2 dispatch: under DELTAS the shared turn_list[0] system is"""
    turn_0 = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "u0"},
    ]
    turn_1 = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1], reset_flags=[False, False]
    )

    _apply_cache_bust(
        session,
        _make_credit(
            target=CacheBustTarget.SYSTEM_PREFIX,
            marker=_PREFIX_MARKER,
            turn_index=0,
            num_turns=2,
        ),
        system_message=None,
    )
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "S0"

    _apply_cache_bust(
        session,
        _make_credit(
            target=CacheBustTarget.SYSTEM_PREFIX,
            marker=_PREFIX_MARKER,
            turn_index=1,
            num_turns=2,
        ),
        system_message=None,
    )
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "S0"


def test_system_prefix_conversation_message_safe_under_reset():
    """Sub-path 1 regression: a Conversation-level system_message is re-marked"""
    turns = [
        Turn(
            raw_messages=[{"role": "user", "content": "u0"}],
            reset_context=False,
        ),
        Turn(
            raw_messages=[{"role": "user", "content": "u1"}],
            reset_context=True,
        ),
    ]
    conversation = Conversation(
        session_id="conv_test", turns=list(turns), system_message="CONV"
    )
    session = UserSession(
        x_correlation_id="xcorr_test",
        num_turns=2,
        conversation=conversation,
        turn_list=list(turns),
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX,
        marker=_PREFIX_MARKER,
        turn_index=1,
        num_turns=2,
    )

    out = _apply_cache_bust(session, credit, system_message="CONV")

    assert out == _PREFIX_MARKER + "CONV"
    assert session.turn_list[1].raw_messages[0]["content"] == "u1"


_ALL_TARGETS = [
    CacheBustTarget.FIRST_TURN_PREFIX,
    CacheBustTarget.FIRST_TURN_SUFFIX,
    CacheBustTarget.SYSTEM_PREFIX,
    CacheBustTarget.SYSTEM_SUFFIX,
]


def _marker_for(target: CacheBustTarget) -> str:
    """Prefix targets need a trailing-newline marker; suffix targets a leading one"""
    return (
        _SUFFIX_MARKER
        if target in (CacheBustTarget.FIRST_TURN_SUFFIX, CacheBustTarget.SYSTEM_SUFFIX)
        else _PREFIX_MARKER
    )


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_fork_child_is_noop_for_all_targets(target: CacheBustTarget):
    """A FORK child must never re-bust its inherited prefix, regardless of target."""
    parent_marked = Turn(
        raw_messages=[
            {"role": "system", "content": "[rid:PARENT00000]\n\nS0"},
            {"role": "user", "content": "[rid:PARENT00000]\n\nu0"},
        ],
        reset_context=False,
    )
    child_turn = Turn(
        raw_messages=[
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
        ],
        reset_context=False,
    )
    session = _make_fork_child_session([parent_marked, child_turn])
    before_sys = session.turn_list[0].raw_messages[0]["content"]
    before_user = session.turn_list[0].raw_messages[1]["content"]
    credit = _make_credit(
        target=target, marker="[rid:CHILD000000]\n\n", turn_index=1, num_turns=2
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[0].raw_messages[0]["content"] == before_sys
    assert session.turn_list[0].raw_messages[1]["content"] == before_user


def test_fork_child_noop_even_when_conversation_system_message_present():
    """SYSTEM sub-path 1 (Conversation-level system_message) is also skipped for"""
    parent_marked = Turn(
        raw_messages=[{"role": "user", "content": "u0"}], reset_context=False
    )
    conversation = Conversation(
        session_id="child", turns=[parent_marked], system_message="CONV"
    )
    session = UserSession(
        x_correlation_id="child_xcorr",
        num_turns=1,
        conversation=conversation,
        turn_list=[parent_marked],
        parent_correlation_id="parent_xcorr",
        branch_mode=ConversationBranchMode.FORK,
    )
    credit = _make_credit(
        target=CacheBustTarget.SYSTEM_PREFIX, marker=_PREFIX_MARKER, num_turns=1
    )

    out = _apply_cache_bust(session, credit, system_message="CONV")

    assert out == "CONV"


def test_fork_child_multi_turn_prefix_stays_single_marked():
    """Processing several FORK-child credits never stacks onto the shared turn 0."""
    shared_t0 = Turn(
        raw_messages=[{"role": "user", "content": "[rid:PARENT00000]\n\nu0"}],
        reset_context=False,
    )
    later = Turn(
        raw_messages=[
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u_later"},
        ],
        reset_context=False,
    )
    session = _make_fork_child_session([shared_t0, later], num_turns=2)
    for ti in (1, 1, 1):
        _apply_cache_bust(
            session,
            _make_credit(
                target=CacheBustTarget.FIRST_TURN_PREFIX,
                marker="[rid:CHILD000000]\n\n",
                turn_index=ti,
                num_turns=2,
            ),
            system_message=None,
        )
    assert session.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"


def test_fork_child_with_own_reset_is_still_noop():
    """A FORK child carrying its OWN reset_context turn is still a no-op: FORK"""
    shared_t0 = Turn(
        raw_messages=[{"role": "user", "content": "[rid:PARENT00000]\n\nu0"}],
        reset_context=False,
    )
    child_reset = Turn(
        raw_messages=[{"role": "user", "content": "child fresh prefix"}],
        reset_context=True,
    )
    session = _make_fork_child_session([shared_t0, child_reset], num_turns=2)
    credit = _make_credit(
        target=CacheBustTarget.FIRST_TURN_PREFIX,
        marker="[rid:CHILD000000]\n\n",
        turn_index=1,
        num_turns=2,
    )

    _apply_cache_bust(session, credit, system_message=None)

    assert session.turn_list[1].raw_messages[0]["content"] == "child fresh prefix"
    assert session.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"


def test_fork_lifecycle_child_inherits_parents_marked_turn_object():
    """End-to-end at the session layer: a parent marks turn 0 in place, a FORK"""
    mgr = UserSessionManager()
    t0 = Turn(raw_messages=[{"role": "user", "content": "u0"}], reset_context=False)
    parent_conv = Conversation(session_id="root", turns=[t0])
    parent = mgr.create_and_store("P", parent_conv, num_turns=1)
    parent.advance_turn(0)
    _apply_cache_bust(
        parent,
        _make_credit(
            target=CacheBustTarget.FIRST_TURN_PREFIX,
            marker="[rid:PARENT00000]\n\n",
            turn_index=0,
            num_turns=1,
        ),
        system_message=None,
    )
    assert parent.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"

    child = mgr.create_and_store(
        "C",
        parent_conv,
        num_turns=1,
        parent_correlation_id="P",
        branch_mode=ConversationBranchMode.FORK,
    )
    mgr.seed_from_parent("C", "P")
    assert child.turn_list[0] is parent.turn_list[0]

    _apply_cache_bust(
        child,
        _make_credit(
            target=CacheBustTarget.FIRST_TURN_PREFIX,
            marker="[rid:CHILD000000]\n\n",
            turn_index=0,
            num_turns=1,
        ),
        system_message=None,
    )

    assert child.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"
    assert parent.turn_list[0].raw_messages[0]["content"] == "[rid:PARENT00000]\n\nu0"


@pytest.mark.parametrize("target", _ALL_TARGETS)
def test_spawn_child_is_busted_for_all_targets(target: CacheBustTarget):
    marker = _marker_for(target)
    raw = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "u0"},
    ]
    turn = Turn(raw_messages=[dict(m) for m in raw], reset_context=False)
    conversation = Conversation(session_id="spawn", turns=[turn])
    session = UserSession(
        x_correlation_id="spawn_xcorr",
        num_turns=1,
        conversation=conversation,
        turn_list=[turn],
        parent_correlation_id="parent_xcorr",
        branch_mode=ConversationBranchMode.SPAWN,
    )

    _apply_cache_bust(
        session,
        _make_credit(target=target, marker=marker, turn_index=0, num_turns=1),
        system_message=None,
    )

    msgs = session.turn_list[0].raw_messages
    if target in (CacheBustTarget.SYSTEM_PREFIX, CacheBustTarget.SYSTEM_SUFFIX):
        carrier = msgs[0]["content"]
    else:
        carrier = msgs[1]["content"]
    assert _PREFIX_MARKER.strip() in carrier


def test_within_session_many_credits_single_marker_prefix():
    """A root session re-processed across many credits keeps exactly one marker"""
    t0_raw = [{"role": "user", "content": "u0"}]
    rest_raw = [
        [
            {"role": "assistant", "content": f"a{i}"},
            {"role": "user", "content": f"u{i + 1}"},
        ]
        for i in range(4)
    ]
    session = _make_delta_session_with_resets(
        [t0_raw, *rest_raw], reset_flags=[False] * 5
    )
    for ti in range(5):
        _apply_cache_bust(
            session,
            _make_credit(
                target=CacheBustTarget.FIRST_TURN_PREFIX,
                marker=_PREFIX_MARKER,
                turn_index=ti,
                num_turns=5,
            ),
            system_message=None,
        )
    assert session.turn_list[0].raw_messages[0]["content"] == _PREFIX_MARKER + "u0"


def test_within_session_many_credits_single_marker_suffix():
    session = _make_delta_session_with_resets(
        [[{"role": "user", "content": "u0"}]], reset_flags=[False]
    )
    for _ in range(3):
        _apply_cache_bust(
            session,
            _make_credit(
                target=CacheBustTarget.FIRST_TURN_SUFFIX,
                marker=_SUFFIX_MARKER,
                turn_index=0,
                num_turns=1,
            ),
            system_message=None,
        )
    assert session.turn_list[0].raw_messages[0]["content"] == "u0" + _SUFFIX_MARKER


def test_seeded_resume_with_buried_reset_suffix():
    """Buried reset + suffix target on a seeded resume: marker suffixes the reset"""
    turn_0 = [{"role": "user", "content": "u0"}]
    turn_1_reset = [{"role": "user", "content": "u1"}]
    turn_2 = [
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    session = _make_delta_session_with_resets(
        [turn_0, turn_1_reset, turn_2], reset_flags=[False, True, False]
    )
    _apply_cache_bust(
        session,
        _make_credit(
            target=CacheBustTarget.FIRST_TURN_SUFFIX,
            marker=_SUFFIX_MARKER,
            turn_index=2,
            num_turns=3,
        ),
        system_message=None,
    )
    assert session.turn_list[1].raw_messages[0]["content"] == "u1" + _SUFFIX_MARKER
    assert session.turn_list[0].raw_messages[0]["content"] == "u0"
