# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAISpeechEndpoint (text-to-speech)."""

import base64

import orjson
import pytest

from aiperf.common.models import AudioResponseData, Text, Turn
from aiperf.common.models.record_models import BinaryResponse, SSEMessage, TextResponse
from aiperf.endpoints.openai_speech import OpenAISpeechEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


def _sse(obj: dict, perf_ns: int = 222) -> SSEMessage:
    return SSEMessage.parse(b"data: " + orjson.dumps(obj), perf_ns)


class TestSpeechEndpointFormatPayload:
    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.SPEECH, model_name="tts-1")

    @pytest.fixture
    def streaming_model_endpoint(self):
        return create_model_endpoint(
            EndpointType.SPEECH, model_name="tts-1", streaming=True
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(OpenAISpeechEndpoint, model_endpoint)

    def test_format_payload_defaults(self, endpoint, model_endpoint):
        request_info = create_request_info(
            model_endpoint=model_endpoint, texts=["Hello world"], model="tts-1"
        )
        payload = endpoint.format_payload(request_info)
        assert payload["input"] == "Hello world"
        assert payload["model"] == "tts-1"
        assert payload["voice"] == "alloy"
        assert payload["response_format"] == "mp3"
        assert "stream_format" not in payload

    def test_format_payload_streaming_sets_stream_format(
        self, streaming_model_endpoint
    ):
        endpoint = create_endpoint_with_mock_transport(
            OpenAISpeechEndpoint, streaming_model_endpoint
        )
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, texts=["Hi"]
        )
        payload = endpoint.format_payload(request_info)
        assert payload["stream_format"] == "sse"

    def test_format_payload_extra_inputs_override_defaults(self):
        model_endpoint = create_model_endpoint(
            EndpointType.SPEECH,
            model_name="tts-1",
            extra=[("voice", "echo"), ("response_format", "wav"), ("speed", 1.5)],
        )
        endpoint = create_endpoint_with_mock_transport(
            OpenAISpeechEndpoint, model_endpoint
        )
        request_info = create_request_info(model_endpoint=model_endpoint, texts=["Hi"])
        payload = endpoint.format_payload(request_info)
        assert payload["voice"] == "echo"
        assert payload["response_format"] == "wav"
        assert payload["speed"] == 1.5

    def test_format_payload_turn_extra_body_overrides(self, model_endpoint, endpoint):
        turn = Turn(texts=[Text(contents=["Hi"])], extra_body={"voice": "fable"})
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)
        assert payload["voice"] == "fable"

    def test_format_payload_no_text_raises(self, model_endpoint, endpoint):
        turn = Turn(texts=[])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        with pytest.raises(ValueError, match="text input"):
            endpoint.format_payload(request_info)


class TestSpeechEndpointParseResponse:
    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.SPEECH, model_name="tts-1")
        return create_endpoint_with_mock_transport(OpenAISpeechEndpoint, model_endpoint)

    def test_parse_binary_clip(self, endpoint):
        response = BinaryResponse(
            perf_ns=999, raw_bytes=b"RIFFxxxx", content_type="audio/wav"
        )
        parsed = endpoint.parse_response(response)
        assert parsed is not None
        assert parsed.perf_ns == 999
        assert isinstance(parsed.data, AudioResponseData)
        assert parsed.data.audio_bytes == b"RIFFxxxx"
        assert parsed.data.format == "wav"

    def test_parse_binary_mpeg_format(self, endpoint):
        response = BinaryResponse(
            perf_ns=1, raw_bytes=b"\xff\xfb", content_type="audio/mpeg"
        )
        parsed = endpoint.parse_response(response)
        assert parsed.data.format == "mp3"

    def test_parse_empty_binary_returns_none(self, endpoint):
        response = BinaryResponse(perf_ns=1, raw_bytes=b"", content_type="audio/wav")
        assert endpoint.parse_response(response) is None

    def test_parse_sse_audio_delta(self, endpoint):
        audio = b"\x01\x02\x03\x04"
        msg = _sse(
            {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(audio).decode("utf-8"),
            },
            perf_ns=555,
        )
        parsed = endpoint.parse_response(msg)
        assert parsed is not None
        assert parsed.perf_ns == 555
        assert isinstance(parsed.data, AudioResponseData)
        assert parsed.data.audio_bytes == audio

    def test_parse_sse_done_is_usage_only(self, endpoint):
        msg = _sse(
            {"type": "speech.audio.done", "usage": {"input_tokens": 5}}, perf_ns=777
        )
        parsed = endpoint.parse_response(msg)
        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None

    def test_parse_sse_done_marker_returns_none(self, endpoint):
        msg = SSEMessage.parse(b"data: [DONE]", 888)
        assert endpoint.parse_response(msg) is None

    def test_parse_non_audio_text_returns_none(self, endpoint):
        response = TextResponse(
            perf_ns=1, text='{"foo": "bar"}', content_type="application/json"
        )
        assert endpoint.parse_response(response) is None
