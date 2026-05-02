# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ChatEndpoint multimodal UUID cache emission (vLLM extension).

ChatEndpoint is stateless. Dedup happens at dataset load time
(`SingleTurnDatasetLoader._dedup_repeated_images_inplace`): repeats
within a conversation arrive with empty `image.contents[i]` but their
`image.uuids[i]` set. The endpoint just translates the pre-deduped data
into the wire format.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_request_info


def _make_endpoint(uuid_and_strip: bool) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_url="http://localhost:8000",
            uuid_and_strip=uuid_and_strip,
        ),
    )


def _turn_with_images(contents: list[str], uuids: list[str] | None = None) -> Turn:
    image = Image(
        name="img",
        contents=contents,
        uuids=uuids if uuids is not None else [],
    )
    return Turn(texts=[Text(contents=["describe"])], images=[image])


def _image_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract image_url content parts from a chat-completions payload."""
    parts: list[dict[str, Any]] = []
    for msg in payload["messages"]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                parts.append(part)
    return parts


class TestUuidAndStripOff:
    """`uuid_and_strip=False` produces wire output identical to today: no `uuid` keys."""

    def test_no_uuid_emitted_even_when_image_uuids_present(self):
        endpoint = ChatEndpoint(_make_endpoint(False))
        turn = _turn_with_images(
            contents=["http://example.com/a.png", "http://example.com/b.png"],
            uuids=["uuid-a", "uuid-b"],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
        payload = endpoint.format_payload(req)
        parts = _image_parts(payload)
        assert len(parts) == 2
        assert all("uuid" not in p for p in parts)
        assert parts[0]["image_url"] == {"url": "http://example.com/a.png"}
        assert parts[1]["image_url"] == {"url": "http://example.com/b.png"}


class TestUuidAndStripOn:
    """`uuid_and_strip=True`: emits `uuid` per part; empty content → strip wire shape."""

    def test_first_occurrence_ships_bytes_with_uuid(self):
        """Non-empty content (first occurrence post-dedup) → ship bytes + uuid."""
        endpoint = ChatEndpoint(_make_endpoint(True))
        turn = _turn_with_images(
            contents=["http://example.com/a.png"], uuids=["uuid-a"]
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
        parts = _image_parts(endpoint.format_payload(req))
        assert parts[0] == {
            "type": "image_url",
            "image_url": {"url": "http://example.com/a.png"},
            "uuid": "uuid-a",
        }

    def test_empty_content_with_uuid_ships_strip_wire_shape(self):
        """Empty content (load-time dedup stripped bytes) → `{url:"", uuid}`."""
        endpoint = ChatEndpoint(_make_endpoint(True))
        turn = _turn_with_images(contents=[""], uuids=["uuid-a"])
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
        parts = _image_parts(endpoint.format_payload(req))
        assert parts[0] == {
            "type": "image_url",
            "image_url": {"url": ""},
            "uuid": "uuid-a",
        }

    def test_sliding_window_pattern(self):
        """Mixed bytes + stripped within one turn — typical post-dedup shape."""
        endpoint = ChatEndpoint(_make_endpoint(True))
        # After load-time dedup: repeats stripped to "", new entry kept.
        turn = _turn_with_images(
            contents=["", "", "", "", "http://example.com/img6.png"],
            uuids=["u2", "u3", "u4", "u5", "u6"],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
        parts = _image_parts(endpoint.format_payload(req))
        assert [p["image_url"] for p in parts] == [
            {"url": ""},
            {"url": ""},
            {"url": ""},
            {"url": ""},
            {"url": "http://example.com/img6.png"},
        ]
        assert [p["uuid"] for p in parts] == ["u2", "u3", "u4", "u5", "u6"]


class TestRawMessagesBypass:
    """raw_messages bypasses UUID injection silently."""

    def test_raw_messages_pass_through(self):
        endpoint = ChatEndpoint(_make_endpoint(True))
        turn = Turn(
            raw_messages=[{"role": "user", "content": "hello"}],
            texts=[Text(contents=["unused"])],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
        payload = endpoint.format_payload(req)
        assert payload["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "uuid_and_strip",
    [
        param(False, id="off"),
        param(True, id="on"),
    ],
)  # fmt: skip
def test_image_without_uuids_never_emits_uuid_key(uuid_and_strip: bool) -> None:
    """When `Image.uuids` is empty, neither mode emits a `uuid` key."""
    endpoint = ChatEndpoint(_make_endpoint(uuid_and_strip))
    turn = _turn_with_images(contents=["http://example.com/a.png"])
    req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
    parts = _image_parts(endpoint.format_payload(req))
    assert "uuid" not in parts[0]
