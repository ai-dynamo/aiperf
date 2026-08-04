# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AudioTranscriptionEndpoint."""

import base64

import orjson
import pytest
from pydantic import TypeAdapter

from aiperf.common.models import Audio, Text, Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.endpoints.openai_audio_transcription import (
    AudioTranscriptionEndpoint,
    _build_audio_field,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

_WAV_B64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode("ascii")
_WAV_CONTENT = f"wav,{_WAV_B64}"
_MP3_B64 = base64.b64encode(b"\xff\xfb\x90\x00" + b"\x00" * 16).decode("ascii")
_MP3_CONTENT = f"mp3,{_MP3_B64}"


class TestAudioTranscriptionEndpoint:
    """Tests for AudioTranscriptionEndpoint format_payload + parse_response."""

    @pytest.fixture
    def model_endpoint(self) -> ModelEndpointInfo:
        return create_model_endpoint(
            EndpointType.AUDIO_TRANSCRIPTION, model_name="openai/whisper-large-v3"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint: ModelEndpointInfo) -> AudioTranscriptionEndpoint:
        return create_endpoint_with_mock_transport(
            AudioTranscriptionEndpoint, model_endpoint
        )

    # ===== format_payload =====

    def test_format_payload_wav_produces_correct_file_field(
        self, endpoint: AudioTranscriptionEndpoint, model_endpoint: ModelEndpointInfo
    ) -> None:
        turn = Turn(
            audios=[Audio(contents=[_WAV_CONTENT])], model="openai/whisper-large-v3"
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "openai/whisper-large-v3"
        file_field = payload["file"]
        assert file_field["b64_data"] == _WAV_B64
        assert file_field["filename"] == "audio.wav"
        assert file_field["content_type"] == "audio/wav"

    def test_format_payload_mp3_mime_and_filename(
        self, endpoint: AudioTranscriptionEndpoint, model_endpoint: ModelEndpointInfo
    ) -> None:
        turn = Turn(audios=[Audio(contents=[_MP3_CONTENT])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["file"]["filename"] == "audio.mp3"
        assert payload["file"]["content_type"] == "audio/mpeg"

    def test_format_payload_is_json_serialisable(
        self, endpoint: AudioTranscriptionEndpoint, model_endpoint: ModelEndpointInfo
    ) -> None:
        """Payload must survive model_dump(mode='json') + orjson round-trip."""
        turn = Turn(audios=[Audio(contents=[_WAV_CONTENT])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        TypeAdapter(dict).dump_python(payload, mode="json")
        orjson.dumps(payload)

    def test_format_payload_extra_inputs_forwarded(
        self, model_endpoint: ModelEndpointInfo
    ) -> None:
        me = create_model_endpoint(
            EndpointType.AUDIO_TRANSCRIPTION,
            extra=[("language", "en"), ("temperature", "0.0")],
        )
        endpoint = create_endpoint_with_mock_transport(AudioTranscriptionEndpoint, me)
        turn = Turn(audios=[Audio(contents=[_WAV_CONTENT])])
        request_info = create_request_info(model_endpoint=me, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["language"] == "en"
        assert payload["temperature"] == "0.0"

    def test_format_payload_reserved_key_in_extra_inputs_ignored(
        self, model_endpoint: ModelEndpointInfo, caplog: pytest.LogCaptureFixture
    ) -> None:
        me = create_model_endpoint(
            EndpointType.AUDIO_TRANSCRIPTION,
            extra=[("file", "should-be-ignored")],
        )
        endpoint = create_endpoint_with_mock_transport(AudioTranscriptionEndpoint, me)
        turn = Turn(audios=[Audio(contents=[_WAV_CONTENT])])
        request_info = create_request_info(model_endpoint=me, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert isinstance(payload["file"], dict)
        assert "b64_data" in payload["file"]

    def test_format_payload_no_turns_raises(
        self, endpoint: AudioTranscriptionEndpoint, model_endpoint: ModelEndpointInfo
    ) -> None:
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_audio_raises(
        self, endpoint: AudioTranscriptionEndpoint, model_endpoint: ModelEndpointInfo
    ) -> None:
        turn = Turn(texts=[Text(contents=["transcribe this"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires audio"):
            endpoint.format_payload(request_info)

    # ===== parse_response =====

    def test_parse_response_extracts_text(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        response = create_mock_response(
            json_data={"text": "Hello world", "usage": {"prompt_tokens": 10}}
        )

        result = endpoint.parse_response(response)

        assert result is not None
        assert result.usage == {"prompt_tokens": 10}

    def test_parse_response_no_json_returns_none(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        response = create_mock_response(json_data=None)
        assert endpoint.parse_response(response) is None

    def test_parse_response_missing_text_field_returns_none(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        response = create_mock_response(json_data={"error": "bad request"})
        assert endpoint.parse_response(response) is None

    def test_parse_response_no_usage_is_fine(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        response = create_mock_response(json_data={"text": "Transcript here."})
        result = endpoint.parse_response(response)
        assert result is not None
        assert result.usage is None

    def test_parse_response_plain_text_body(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        """response_format text/srt/vtt returns a non-JSON body; the whole body
        is the transcript and must not be dropped."""
        response = create_mock_response(json_data=None, text="a plain transcript")
        result = endpoint.parse_response(response)
        assert result is not None
        assert result.usage is None

    def test_parse_response_empty_body_returns_none(
        self, endpoint: AudioTranscriptionEndpoint
    ) -> None:
        response = create_mock_response(json_data=None, text="")
        assert endpoint.parse_response(response) is None

    # ===== _build_audio_field =====

    @pytest.mark.parametrize(
        ("fmt", "expected_mime"),
        [
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("mpga", "audio/mpeg"),
            ("mpeg", "audio/mpeg"),
            ("flac", "audio/flac"),
            ("ogg", "audio/ogg"),
            ("m4a", "audio/mp4"),
            ("mp4", "audio/mp4"),
            ("webm", "audio/webm"),
        ],
    )
    def test_build_audio_field_known_formats(
        self, fmt: str, expected_mime: str
    ) -> None:
        b64 = base64.b64encode(b"fake").decode("ascii")
        field = _build_audio_field(f"{fmt},{b64}")
        assert field["content_type"] == expected_mime
        assert field["filename"] == f"audio.{fmt}"
        assert field["b64_data"] == b64

    def test_build_audio_field_unknown_format_falls_back_to_audio_slash_fmt(
        self,
    ) -> None:
        b64 = base64.b64encode(b"fake").decode("ascii")
        field = _build_audio_field(f"aiff,{b64}")
        assert field["content_type"] == "audio/aiff"

    def test_build_audio_field_missing_comma_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            _build_audio_field("nob64herenocomma")
