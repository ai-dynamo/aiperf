# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Chat Completions image UUID cache emission."""

from __future__ import annotations

from typing import Any

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Audio, Image, Text, Turn, Video
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_request_info


def _make_endpoint(uuid_and_strip: bool) -> ChatEndpoint:
    model_endpoint = ModelEndpointInfo(
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
    return ChatEndpoint(model_endpoint)


def _image_parts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload["messages"][0]["content"]
    return [part for part in content if part["type"] == "image_url"]


def test_authored_uuids_pass_through_when_strip_disabled() -> None:
    endpoint = _make_endpoint(False)
    turn = Turn(
        texts=[Text(contents=["describe"])],
        images=[
            Image(contents=["a.png"], uuids=["uuid-a"]),
            Image(uuids=["uuid-b"]),
            Image(contents=["without-uuid.png"]),
        ],
    )
    request = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

    assert _image_parts(endpoint.format_payload(request)) == [
        {"type": "image_url", "image_url": {"url": "a.png"}, "uuid": "uuid-a"},
        {"type": "image_url", "image_url": {"url": ""}, "uuid": "uuid-b"},
        {"type": "image_url", "image_url": {"url": "without-uuid.png"}},
    ]


def test_uuid_and_strip_emits_cached_and_uncached_image_parts() -> None:
    endpoint = _make_endpoint(True)
    turn = Turn(
        texts=[Text(contents=["describe"])],
        images=[
            Image(contents=["a.png"], uuids=["uuid-a"]),
            Image(uuids=["uuid-b"]),
            Image(contents=["without-uuid.png"]),
        ],
    )
    request = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])
    payload = endpoint.format_payload(request)

    assert _image_parts(payload) == [
        {"type": "image_url", "image_url": {"url": "a.png"}, "uuid": "uuid-a"},
        {"type": "image_url", "image_url": {"url": ""}, "uuid": "uuid-b"},
        {"type": "image_url", "image_url": {"url": "without-uuid.png"}},
    ]
    assert endpoint.extract_payload_inputs(payload).image_count == 3


def test_uuid_and_strip_keeps_shared_audio_and_video_rendering() -> None:
    endpoint = _make_endpoint(True)
    turn = Turn(
        texts=[Text(contents=["inspect"])],
        audios=[Audio(contents=["data:audio/wav;base64,QUJD"])],
        videos=[Video(contents=["video.mp4"])],
    )
    request = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

    content = endpoint.format_payload(request)["messages"][0]["content"]
    assert content[1:] == [
        {
            "type": "input_audio",
            "input_audio": {"format": "wav", "data": "QUJD"},
        },
        {"type": "video_url", "video_url": {"url": "video.mp4"}},
    ]


def test_uuid_and_strip_preserves_raw_messages() -> None:
    endpoint = _make_endpoint(True)
    turn = Turn(raw_messages=[{"role": "user", "content": "hello"}])
    request = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

    assert endpoint.format_payload(request)["messages"] == [
        {"role": "user", "content": "hello"}
    ]
