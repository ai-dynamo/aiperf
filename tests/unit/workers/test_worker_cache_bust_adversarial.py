# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial edge-case coverage for cache-bust injection helpers in ``aiperf.workers.worker``."""

from __future__ import annotations

from aiperf.common.enums import CacheBustTarget
from aiperf.workers.worker import (
    _apply_cache_bust_to_system_message,
    _inject_marker_into_first_user_turn,
    _inject_marker_into_raw_messages,
)


def test_apply_to_system_message_empty_string_marker_is_noop():
    """marker="" must short-circuit via ``not marker`` and return system_message unchanged."""
    out = _apply_cache_bust_to_system_message(
        "hello", "", CacheBustTarget.SYSTEM_PREFIX
    )
    assert out == "hello"


def test_apply_to_system_message_empty_string_system_with_marker_returns_marker():
    """An empty-string (not None) system_message passes the ``is None`` guard and gets prefixed in place, producing the marker alone."""
    out = _apply_cache_bust_to_system_message(
        "", "[rid:abc]\n\n", CacheBustTarget.SYSTEM_PREFIX
    )
    assert out == "[rid:abc]\n\n"


def test_apply_to_system_message_unknown_target_is_passthrough():
    """A non-SYSTEM target (e.g. FIRST_TURN_PREFIX) falls through both branches and returns the input string unchanged."""
    out = _apply_cache_bust_to_system_message(
        "hello", "marker-x", CacheBustTarget.FIRST_TURN_PREFIX
    )
    assert out == "hello"


def test_inject_into_raw_messages_multimodal_content_list_injects_text_part():
    """A list (multimodal) system content gets a new ``{"type":"text","text":marker}`` part inserted at the start on prefix."""
    raw: list[dict] = [{"role": "system", "content": [{"type": "text", "text": "hi"}]}]

    _inject_marker_into_raw_messages(raw, "MARKER", is_prefix=True)

    assert raw == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "MARKER"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]


def test_inject_into_raw_messages_with_extra_keys_preserves_them():
    """The spread-then-overwrite rewrite preserves every original key; only ``content`` changes."""
    raw: list[dict] = [
        {
            "role": "system",
            "content": "hi",
            "name": "sys_v1",
            "metadata": {"x": 1},
        }
    ]

    _inject_marker_into_raw_messages(raw, "m", is_prefix=True)

    assert raw[0]["name"] == "sys_v1"
    assert raw[0]["metadata"] == {"x": 1}
    assert raw[0]["content"] == "m" + "hi"
    assert raw[0]["role"] == "system"


def test_inject_into_raw_messages_first_message_not_dict_is_noop():
    """A non-dict first element (e.g. a stray string from a malformed trace) is skipped cleanly without raising."""
    raw: list = ["not a dict"]
    snapshot = list(raw)

    _inject_marker_into_raw_messages(raw, "M", is_prefix=True)

    assert raw == snapshot


def test_inject_into_first_user_turn_only_first_user_mutated():
    """Only the FIRST user-role message is mutated; subsequent user-role messages remain untouched."""
    raw: list[dict] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]

    _inject_marker_into_first_user_turn(raw, "M", is_prefix=True)

    assert raw[1]["content"] == "M" + "u1"
    assert raw[3]["content"] == "u2"  # second user untouched
    assert raw[0]["content"] == "s"
    assert raw[2]["content"] == "a"


def test_inject_into_first_user_turn_no_user_role_is_noop():
    """No user-role message anywhere leaves the list untouched (system + assistant only)."""
    raw: list[dict] = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a"},
    ]
    snapshot = [dict(msg) for msg in raw]

    _inject_marker_into_first_user_turn(raw, "M", is_prefix=True)

    assert raw == snapshot


def test_inject_into_first_user_turn_multimodal_content_injects_text_part():
    """Multimodal content on the first user turn injects the marker as a new text part (same as the system-message path)."""
    raw: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    _inject_marker_into_first_user_turn(raw, "MARKER", is_prefix=True)

    assert raw == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "MARKER"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]
