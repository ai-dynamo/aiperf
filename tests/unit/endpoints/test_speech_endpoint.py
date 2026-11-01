# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speech payloads and audio duration must not depend on a particular server."""

import io
import wave

import pytest
from pydantic import TypeAdapter, ValidationError
from pytest import param

from aiperf.common.models import (
    AudioResponseData,
    BinaryResponse,
    Text,
    TextResponse,
    Turn,
)
from aiperf.config.endpoint import EndpointConfig, PCMOutputConfig
from aiperf.endpoints.openai_speech import SpeechEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


def wav_body(frames: int = 320, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Construct a finite PCM WAV, including non-UTF-8 samples."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((channels, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\xff\x80" * frames * channels)
    return output.getvalue()


@pytest.fixture
def endpoint() -> SpeechEndpoint:
    """Configure explicit raw PCM metadata independently of server parameters."""
    model = create_model_endpoint(EndpointType.SPEECH, extra=[("voice", "test-voice")])
    model.endpoint.pcm_output = PCMOutputConfig(sample_rate=24000, channels=1)
    return SpeechEndpoint(model_endpoint=model)


def test_payload_extra_precedence_and_metadata_not_sent(
    endpoint: SpeechEndpoint,
) -> None:
    """Dataset extras override global extras without leaking benchmark metadata."""
    request = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        turns=[
            Turn(
                texts=[Text(contents=["Hello."])],
                extra_body={"voice": "second", "response_format": "pcm", "speed": 1.1},
            )
        ],
    )
    assert endpoint.format_payload(request) == {
        "input": "Hello.",
        "model": "test-model",
        "voice": "second",
        "response_format": "pcm",
        "speed": 1.1,
    }


@pytest.mark.parametrize(
    "extra, message",
    [
        param({"voice": ""}, "requires voice", id="missing-voice"),
        param({"input": " "}, "nonempty input", id="empty-input"),
        param({"response_format": "mp3"}, "only pcm and wav", id="unsupported-codec"),
        param({"stream_format": "sse"}, "complete audio", id="sse"),
        param({"stream": True}, "complete audio", id="stream"),
    ],
)  # fmt: skip
def test_payload_rejects_unsupported_requests(
    endpoint: SpeechEndpoint, extra: dict, message: str
) -> None:
    """Unsupported output modes fail before a request can be mismeasured."""
    request = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        turns=[Turn(texts=[Text(contents=["Hello."])], extra_body=extra)],
    )
    with pytest.raises(ValueError, match=message):
        endpoint.format_payload(request)


def test_pcm_requires_explicit_metadata(endpoint: SpeechEndpoint) -> None:
    """Headerless bytes cannot determine their own sample rate or channel count."""
    endpoint.model_endpoint.endpoint.pcm_output = None
    request = create_request_info(
        model_endpoint=endpoint.model_endpoint,
        turns=[
            Turn(
                texts=[Text(contents=["Hello."])], extra_body={"response_format": "pcm"}
            )
        ],
    )
    with pytest.raises(ValueError, match="pcm_output"):
        endpoint.format_payload(request)
    assert (
        endpoint.parse_response(
            BinaryResponse(perf_ns=10, raw_bytes=b"\xff\xff", content_type="audio/pcm")
        )
        is None
    )


@pytest.mark.parametrize(
    "sample_rate, channels", [param(16000, 1, id="mono"), param(48000, 2, id="stereo")]
)  # fmt: skip
def test_wav_uses_header_metadata(
    endpoint: SpeechEndpoint, sample_rate: int, channels: int
) -> None:
    """WAV metadata overrides raw PCM declarations and survives metadata JSON."""
    result = endpoint.parse_response(
        BinaryResponse(
            perf_ns=123,
            raw_bytes=wav_body(320, sample_rate, channels),
            content_type="audio/wav",
        )
    )
    assert result is not None
    assert result.perf_ns == 123
    assert result.data == AudioResponseData(
        frames=320, sample_rate=sample_rate, channels=channels
    )
    adapter = TypeAdapter(AudioResponseData)
    assert adapter.validate_json(adapter.dump_json(result.data)) == result.data


def test_pcm_counts_complete_stereo_frames(endpoint: SpeechEndpoint) -> None:
    """Two channels double bytes per frame, not the duration."""
    endpoint.model_endpoint.endpoint.pcm_output = PCMOutputConfig(
        sample_rate=48000, channels=2
    )
    result = endpoint.parse_response(
        BinaryResponse(
            perf_ns=123,
            raw_bytes=b"\xff\x80" * 960,
            content_type="audio/pcm; rate=48000",
        )
    )
    assert result is not None
    assert result.data == AudioResponseData(frames=480, sample_rate=48000, channels=2)
    assert (
        endpoint.parse_response(
            BinaryResponse(perf_ns=123, raw_bytes=b"\xff\x80", content_type="audio/pcm")
        )
        is None
    )


@pytest.mark.parametrize(
    "body, content_type",
    [
        param(b"", "audio/pcm", id="empty"),
        param(b"x", "audio/pcm", id="partial-sample"),
        param(b"not a wave", "audio/wav", id="malformed-wave"),
        param(wav_body()[:-2], "audio/wav", id="truncated-wave"),
        param(wav_body(0), "audio/wav", id="empty-wave"),
        param(wav_body()[:4] + b"\xff" * 4 + wav_body()[8:], "audio/wav", id="unknown-length-wave"),
        param(b"\xff\xfe", "audio/mpeg", id="compressed"),
    ],
)  # fmt: skip
def test_invalid_audio_is_not_success(
    endpoint: SpeechEndpoint, body: bytes, content_type: str
) -> None:
    """Invalid output must reach request error accounting instead of audio throughput."""
    assert (
        endpoint.parse_response(
            BinaryResponse(perf_ns=123, raw_bytes=body, content_type=content_type)
        )
        is None
    )


def test_json_error_is_not_audio(endpoint: SpeechEndpoint) -> None:
    """A 200 response containing JSON does not become a successful speech record."""
    assert (
        endpoint.parse_response(
            TextResponse(
                perf_ns=123, text='{"error":"failed"}', content_type="application/json"
            )
        )
        is None
    )


def test_pcm_config_roundtrip() -> None:
    """YAML-style aliases retain PCM format through endpoint serialization."""
    config = EndpointConfig.model_validate(
        {
            "url": "localhost:8000",
            "type": "speech",
            "pcmOutput": {"sampleRate": 48000, "channels": 2},
        }
    )
    assert config.pcm_output == PCMOutputConfig(sample_rate=48000, channels=2)
    assert (
        EndpointConfig.model_validate_json(config.model_dump_json()).pcm_output
        == config.pcm_output
    )
    with pytest.raises(ValidationError):
        PCMOutputConfig(sample_rate=0, channels=1)
    with pytest.raises(ValidationError):
        PCMOutputConfig(sample_rate=24000, channels=1, sample_format="float32")
