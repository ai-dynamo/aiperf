# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media uuid suffixing (``cache_bust.media_uuids``) in ``aiperf.workers.worker``."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models.dataset_models import Conversation, Image, Turn
from aiperf.credit.structs import Credit
from aiperf.timing.strategies.cache_bust import (
    RID_TARGETS,
    WARMUP_ISOLATION_MARKER,
    build_cache_bust_marker,
    rid_digest,
)
from aiperf.workers.session_manager import UserSession
from aiperf.workers.worker import _apply_cache_bust, _suffix_media_uuids

_SCREEN_A = "data:image/png;base64,QUFB"
_SCREEN_B = "data:image/png;base64,QkJC"


def _image_part(url: str, uuid: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}, "uuid": uuid}


def _agent_messages() -> list[dict]:
    """One recorded agent request: a screenshot in a user and in a tool message, each with its cache uuid."""
    return [
        {"role": "system", "content": "You are a computer-use agent."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Open the settings."},
                _image_part(_SCREEN_A, "1111"),
            ],
        },
        {"role": "assistant", "content": "click(10, 20)"},
        {
            "role": "tool",
            "tool_call_id": "call-0",
            "content": [_image_part(_SCREEN_B, "2222")],
        },
    ]


def _session(*turns: Turn, fork: bool = False) -> UserSession:
    """A session whose ``turn_list`` holds ``turns``, as after ``advance_turn``."""
    turn_list = list(turns) or [Turn(raw_messages=_agent_messages())]
    return UserSession(
        x_correlation_id="xcorr",
        num_turns=len(turn_list),
        conversation=Conversation(session_id="traj-0", turns=turn_list),
        turn_list=turn_list,
        parent_correlation_id="xcorr-parent" if fork else None,
        branch_mode=ConversationBranchMode.FORK if fork else None,
    )


def _marker(
    recycle_pass: int = 0, target: CacheBustTarget = CacheBustTarget.SYSTEM_SUFFIX
) -> str:
    marker = build_cache_bust_marker("bench", recycle_pass, 0, "traj-0", target=target)
    assert marker is not None
    return marker


def _credit(marker: str | None, target: CacheBustTarget) -> Credit:
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id="traj-0",
        x_correlation_id="xcorr",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        cache_bust_marker=marker,
        cache_bust_target=target,
    )


def _apply(
    session: UserSession,
    marker: str,
    target: CacheBustTarget = CacheBustTarget.SYSTEM_SUFFIX,
    *,
    media_uuids: bool = True,
) -> None:
    _apply_cache_bust(session, _credit(marker, target), None, media_uuids=media_uuids)


def _uuids_by_url(session: UserSession) -> dict[str, set[str]]:
    """Every uuid the session sends, grouped by the media it names."""
    by_url: dict[str, set[str]] = {}
    for turn in session.turn_list:
        for message in turn.raw_messages or []:
            content = message.get("content")
            parts = content if isinstance(content, list) else []
            for part in (p for p in parts if "uuid" in p):
                by_url.setdefault(part["image_url"]["url"], set()).add(part["uuid"])
    return by_url


def test_apply_cache_bust_media_uuids_false_leaves_uuids_unchanged():
    session = _session()

    _apply(session, _marker(), media_uuids=False)

    assert _uuids_by_url(session) == {_SCREEN_A: {"1111"}, _SCREEN_B: {"2222"}}


@pytest.mark.parametrize("target", [param(t, id=str(t)) for t in RID_TARGETS])
def test_apply_cache_bust_media_uuids_suffix_is_marker_digest(target: CacheBustTarget):
    session = _session()
    marker = _marker(target=target)

    _apply(session, marker, target)

    digest = rid_digest(marker)
    assert _uuids_by_url(session) == {
        _SCREEN_A: {f"1111-{digest}"},
        _SCREEN_B: {f"2222-{digest}"},
    }


def test_apply_cache_bust_media_uuids_same_suffix_across_turns():
    """The screenshot window repeats a screenshot over several turns; all of them must name the same cache entry."""
    later = [
        *_agent_messages(),
        {"role": "assistant", "content": "type('wifi')"},
        {"role": "user", "content": [_image_part(_SCREEN_A, "1111")]},
    ]
    session = _session(Turn(raw_messages=_agent_messages()), Turn(raw_messages=later))
    marker = _marker()

    for _ in range(3):  # one call per credit; must not stack the suffix
        _apply(session, marker)

    digest = rid_digest(marker)
    assert _uuids_by_url(session) == {
        _SCREEN_A: {f"1111-{digest}"},
        _SCREEN_B: {f"2222-{digest}"},
    }


def test_apply_cache_bust_media_uuids_replay_gets_new_suffix():
    """Recycle passes of one trajectory decode fresh copies of it and get different markers."""
    first_play, replay = _session(), _session()

    _apply(first_play, _marker(recycle_pass=0))
    _apply(replay, _marker(recycle_pass=1))

    for url in (_SCREEN_A, _SCREEN_B):
        assert _uuids_by_url(first_play)[url].isdisjoint(_uuids_by_url(replay)[url])


def test_apply_cache_bust_media_uuids_replaces_message_dicts():
    """Like marker injection, never edit a message or part dict in place."""
    session = _session()
    raw_messages = session.turn_list[0].raw_messages
    assert raw_messages is not None
    user_message = raw_messages[1]
    image_part = user_message["content"][1]

    _apply(session, _marker())

    assert image_part["uuid"] == "1111"
    assert user_message["content"][1] is image_part
    assert raw_messages[1] is not user_message


def test_suffix_media_uuids_part_without_uuid_unchanged():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look."},
            {"type": "image_url", "image_url": {"url": _SCREEN_A}},
        ],
    }
    turn = Turn(raw_messages=[message])
    assert turn.raw_messages is not None
    stored = turn.raw_messages[0]

    _suffix_media_uuids([turn], "abc123def456")

    assert turn.raw_messages == [message]
    assert turn.raw_messages[0] is stored


def test_suffix_media_uuids_is_idempotent():
    turn = Turn(raw_messages=_agent_messages())
    _suffix_media_uuids([turn], "abc123def456")
    assert turn.raw_messages is not None
    first = list(turn.raw_messages)

    _suffix_media_uuids([turn], "abc123def456")

    assert all(a is b for a, b in zip(turn.raw_messages, first, strict=True))


def test_apply_cache_bust_media_uuids_suffixes_image_uuids():
    turn = Turn(
        images=[
            Image(contents=[_SCREEN_A, _SCREEN_B], uuids=["1111", "2222"]),
            Image(contents=[_SCREEN_A]),
        ]
    )
    marker = _marker()

    _apply(_session(turn), marker)

    digest = rid_digest(marker)
    assert turn.images[0].uuids == [f"1111-{digest}", f"2222-{digest}"]
    assert turn.images[0].contents == [_SCREEN_A, _SCREEN_B]
    assert turn.images[1].uuids == []


def test_apply_cache_bust_media_uuids_fork_child_suffixes_own_media():
    """A FORK child shares its parent's digest: inherited turns stay as the parent sent them."""
    marker = _marker()
    parent = _session()
    _apply(parent, marker)
    inherited = parent.turn_list[0]
    assert inherited.raw_messages is not None
    sent_by_parent = list(inherited.raw_messages)
    own = Turn(
        raw_messages=[{"role": "user", "content": [_image_part(_SCREEN_B, "3333")]}]
    )
    child = _session(inherited, own, fork=True)

    _apply(child, marker)

    assert all(
        now is before
        for now, before in zip(inherited.raw_messages, sent_by_parent, strict=True)
    )
    assert own.raw_messages is not None
    assert own.raw_messages[0]["content"][0]["uuid"] == f"3333-{rid_digest(marker)}"


def test_apply_cache_bust_media_uuids_warmup_isolation_marker_unchanged():
    session = _session()

    _apply(session, WARMUP_ISOLATION_MARKER, CacheBustTarget.WARMUP_ISOLATION_SYSTEM)

    assert _uuids_by_url(session) == {_SCREEN_A: {"1111"}, _SCREEN_B: {"2222"}}
