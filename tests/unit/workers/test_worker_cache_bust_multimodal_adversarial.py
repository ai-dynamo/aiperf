# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive adversarial coverage for multimodal cache-bust marker injection in ``aiperf.workers.worker``."""

from __future__ import annotations

import pytest

from aiperf.workers.worker import (
    _inject_marker_into_first_user_turn,
    _inject_marker_into_raw_messages,
)

# Marker shape parity with the worker hot path (see ``_apply_cache_bust``).
_PREFIX_MARKER = "[rid:abc123def456]\n\n"
_SUFFIX_MARKER = "\n\n[rid:abc123def456]"

# As-injected text part body — the helpers call ``marker.strip()`` before
# building the new ``{"type": "text", "text": ...}`` dict.
_PREFIX_PART_TEXT = _PREFIX_MARKER.strip()
_SUFFIX_PART_TEXT = _SUFFIX_MARKER.strip()


def test_inject_marker_into_text_only_multimodal_prefix():
    """A pure text-multimodal system message + prefix marker -> a new text
    part is prepended; the original text part survives unchanged at index 1."""
    raw: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": "hello"}]}
    ]

    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": _PREFIX_PART_TEXT},
                {"type": "text", "text": "hello"},
            ],
        }
    ]


def test_inject_marker_into_text_only_multimodal_suffix():
    """Same setup with suffix orientation -> marker text part is appended."""
    raw: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": "hello"}]}
    ]

    _inject_marker_into_raw_messages(raw, _SUFFIX_MARKER, is_prefix=False)

    assert raw == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": _SUFFIX_PART_TEXT},
            ],
        }
    ]


def test_inject_marker_into_image_first_multimodal_prefix():
    """When the original content opens with an image_url part (no leading
    text), the marker still goes at index 0 — token-0 cache-bust semantics
    require the marker to be the literal first token of the wire payload."""
    raw: list[dict] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw=="},
                },
                {"type": "text", "text": "caption"},
            ],
        }
    ]

    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw[0]["content"][0] == {"type": "text", "text": _PREFIX_PART_TEXT}
    # Original parts shift right one slot, in original order.
    assert raw[0]["content"][1]["type"] == "image_url"
    assert raw[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
    assert raw[0]["content"][2] == {"type": "text", "text": "caption"}
    assert len(raw[0]["content"]) == 3


@pytest.mark.parametrize(
    "is_prefix, expected_marker_index",
    [
        pytest.param(True, 0, id="prefix-marker-at-index-0"),
        pytest.param(False, -1, id="suffix-marker-at-index--1"),
    ],
)
def test_inject_marker_into_audio_video_mixed_content(
    is_prefix: bool, expected_marker_index: int
):
    """Mixed audio + image + video + text parts — marker preserves the original
    parts' order, only adding one new text part at the marker end of the list.

    The helper does NOT inspect part types; it just prepends/appends. Locks
    that behavior so a future change cannot start dropping non-text parts.
    """
    original_parts: list[dict] = [
        {"type": "text", "text": "describe these"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        },
        {
            "type": "input_audio",
            "input_audio": {"data": "AUDIO_B64", "format": "wav"},
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/clip.mp4"},
        },
    ]
    raw: list[dict] = [
        {"role": "system", "content": [dict(p) for p in original_parts]},
    ]

    marker = _PREFIX_MARKER if is_prefix else _SUFFIX_MARKER
    expected_text = _PREFIX_PART_TEXT if is_prefix else _SUFFIX_PART_TEXT

    _inject_marker_into_raw_messages(raw, marker, is_prefix=is_prefix)

    new_content = raw[0]["content"]
    assert len(new_content) == len(original_parts) + 1
    # Marker landed in the right place.
    assert new_content[expected_marker_index] == {
        "type": "text",
        "text": expected_text,
    }
    # All original parts present, in original order, with original values.
    remaining = new_content[1:] if is_prefix else new_content[:-1]
    assert remaining == original_parts


def test_inject_marker_preserves_extra_keys_on_message_dict_multimodal():
    """The spread-then-overwrite rewrite (``{**first, "content": new_content}``)
    must preserve every non-content key on the original message dict —
    metadata, name, tool_call_id, anything. Locks that the multimodal branch
    of the helper (the ``isinstance(content, list)`` arm) uses the same
    rewrite shape as the string branch.
    """
    raw: list[dict] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "hi"}],
            "name": "sys-v3",
            "metadata": {"trace_id": "abc", "tags": ["x", "y"]},
            "tool_call_id": "call_42",
            "extra_field": object(),
        }
    ]
    sentinel_obj = raw[0]["extra_field"]

    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    out = raw[0]
    # Content was rewritten correctly.
    assert out["content"] == [
        {"type": "text", "text": _PREFIX_PART_TEXT},
        {"type": "text", "text": "hi"},
    ]
    # Every original non-content key survived.
    assert out["role"] == "system"
    assert out["name"] == "sys-v3"
    assert out["metadata"] == {"trace_id": "abc", "tags": ["x", "y"]}
    assert out["tool_call_id"] == "call_42"
    # Object identity preserved (no deep-copy, just spread).
    assert out["extra_field"] is sentinel_obj


def test_inject_marker_into_raw_messages_unexpected_content_int(caplog):
    """``content = 12345`` (int) -> not str, not list -> helper logs WARNING
    and leaves the message untouched (marker dropped, but loudly)."""
    raw: list[dict] = [{"role": "system", "content": 12345}]

    with caplog.at_level("WARNING"):
        _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw == [{"role": "system", "content": 12345}]
    assert any("cache-bust" in rec.message for rec in caplog.records), (
        "expected at least one cache-bust warning"
    )
    assert any("int" in rec.message for rec in caplog.records), (
        "warning should name the offending type (int)"
    )


def test_inject_marker_into_raw_messages_unexpected_content_dict(caplog):
    """``content = {"foo": "bar"}`` (dict, NOT list of parts) -> helper logs
    WARNING and leaves the message untouched. Locks the strict
    ``isinstance(content, list)`` check — a dict-shaped content is not
    promoted to a single-element list."""
    raw: list[dict] = [{"role": "system", "content": {"foo": "bar"}}]

    with caplog.at_level("WARNING"):
        _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw == [{"role": "system", "content": {"foo": "bar"}}]
    assert any("cache-bust" in rec.message for rec in caplog.records)
    assert any("dict" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "is_prefix, marker, expected_text",
    [
        pytest.param(True, _PREFIX_MARKER, _PREFIX_PART_TEXT, id="prefix"),
        pytest.param(False, _SUFFIX_MARKER, _SUFFIX_PART_TEXT, id="suffix"),
    ],
)
@pytest.mark.parametrize(
    "user_content, original_parts_id",
    [
        pytest.param(
            [{"type": "text", "text": "hi"}],
            "text-only",
            id="text-only",
        ),
        pytest.param(
            [
                {"type": "image_url", "image_url": {"url": "img.png"}},
                {"type": "text", "text": "caption"},
            ],
            "image-first",
            id="image-first",
        ),
        pytest.param(
            [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "i.png"}},
                {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "v.mp4"}},
            ],
            "mixed",
            id="mixed",
        ),
    ],
)
def test_inject_marker_into_first_user_turn_multimodal_user_message(
    is_prefix: bool,
    marker: str,
    expected_text: str,
    user_content: list[dict],
    original_parts_id: str,
):
    """Same parametrization sweep as the system-role variant, but on a
    user-role message via ``_inject_marker_into_first_user_turn``. Both
    helpers share the multimodal injection logic; this locks parity."""
    original = [dict(p) for p in user_content]
    raw: list[dict] = [{"role": "user", "content": [dict(p) for p in user_content]}]

    _inject_marker_into_first_user_turn(raw, marker, is_prefix=is_prefix)

    new_content = raw[0]["content"]
    assert len(new_content) == len(original) + 1
    if is_prefix:
        assert new_content[0] == {"type": "text", "text": expected_text}
        assert new_content[1:] == original
    else:
        assert new_content[-1] == {"type": "text", "text": expected_text}
        assert new_content[:-1] == original


def test_inject_marker_multimodal_first_user_after_system():
    """raw_messages = [system_dict, user_multimodal_dict]. Calling the
    first-user-turn helper must:

    - leave the system message at index 0 completely untouched (different
      content type, different role);
    - inject the marker as a new text part on the user message at index 1;
    - find the user message via the role==``user`` filter (skip system).
    """
    system_msg: dict = {
        "role": "system",
        "content": "you are a helpful assistant",
        "metadata": {"v": 1},
    }
    user_multimodal: dict = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://x/img.jpg"}},
            {"type": "text", "text": "what's in this picture?"},
        ],
        "name": "alice",
    }
    raw: list[dict] = [
        dict(system_msg),
        {**user_multimodal, "content": [dict(p) for p in user_multimodal["content"]]},
    ]

    _inject_marker_into_first_user_turn(raw, _PREFIX_MARKER, is_prefix=True)

    # System unchanged, key-for-key.
    assert raw[0] == system_msg
    # User message: marker prepended, original parts in original order, extra
    # keys (name, role) preserved.
    assert raw[1]["role"] == "user"
    assert raw[1]["name"] == "alice"
    assert raw[1]["content"][0] == {"type": "text", "text": _PREFIX_PART_TEXT}
    assert raw[1]["content"][1]["type"] == "image_url"
    assert raw[1]["content"][1]["image_url"]["url"] == "https://x/img.jpg"
    assert raw[1]["content"][2] == {"type": "text", "text": "what's in this picture?"}
    assert len(raw[1]["content"]) == 3


def test_inject_marker_into_empty_list_content_system_role():
    """``content = []`` (empty list) -> ``isinstance(content, list)`` is True,
    so the helper takes the multimodal path. Result: a single text part
    containing only the stripped marker. Locks that the empty-list shape is
    NOT treated as "missing content" / a no-op."""
    raw: list[dict] = [{"role": "system", "content": []}]

    _inject_marker_into_raw_messages(raw, _PREFIX_MARKER, is_prefix=True)

    assert raw == [
        {
            "role": "system",
            "content": [{"type": "text", "text": _PREFIX_PART_TEXT}],
        }
    ]


def test_inject_marker_into_empty_list_content_first_user():
    """Same edge case on the first-user-turn variant."""
    raw: list[dict] = [{"role": "user", "content": []}]

    _inject_marker_into_first_user_turn(raw, _SUFFIX_MARKER, is_prefix=False)

    assert raw == [
        {
            "role": "user",
            "content": [{"type": "text", "text": _SUFFIX_PART_TEXT}],
        }
    ]
