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
    """An empty-string (NOT None) system_message falls past the early-return guard"""
    out = _apply_cache_bust_to_system_message(
        "", "[rid:abc]\n\n", CacheBustTarget.SYSTEM_PREFIX
    )
    assert out == "[rid:abc]\n\n"


def test_apply_to_system_message_unknown_target_is_passthrough():
    """The helper only handles SYSTEM_PREFIX/SUFFIX/NONE. Any other target"""
    out = _apply_cache_bust_to_system_message(
        "hello", "marker-x", CacheBustTarget.FIRST_TURN_PREFIX
    )
    assert out == "hello"


def test_inject_into_raw_messages_multimodal_content_list_injects_text_part():
    """When the system message's content is a list (multimodal blocks) rather"""
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
    """Locks the spread-then-overwrite pattern (``{**first, "content": ...}``):"""
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
    """If the first element is anything other than a dict (e.g. a stray string"""
    raw: list = ["not a dict"]
    snapshot = list(raw)

    _inject_marker_into_raw_messages(raw, "M", is_prefix=True)

    assert raw == snapshot


def test_inject_into_first_user_turn_only_first_user_mutated():
    """The helper iterates and mutates the FIRST user-role message, then returns."""
    raw: list[dict] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]

    _inject_marker_into_first_user_turn(raw, "M", is_prefix=True)

    assert raw[1]["content"] == "M" + "u1"
    assert raw[3]["content"] == "u2"
    assert raw[0]["content"] == "s"
    assert raw[2]["content"] == "a"


def test_inject_into_first_user_turn_no_user_role_is_noop():
    """No user-role message anywhere in the list -> helper iterates and exits"""
    raw: list[dict] = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a"},
    ]
    snapshot = [dict(msg) for msg in raw]

    _inject_marker_into_first_user_turn(raw, "M", is_prefix=True)

    assert raw == snapshot


def test_inject_into_first_user_turn_multimodal_content_injects_text_part():
    """Multimodal content list on the first user turn -> inject marker as a new"""
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
