# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible complete-response PCM/WAV speech generation."""

import io
import wave
from typing import Any

from aiperf.common.models import (
    AudioResponseData,
    BinaryResponse,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint

_WAV_TYPES = {"audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"}


class SpeechEndpoint(BaseEndpoint):
    """Measure complete audio responses; no first-audio timestamp is inferred."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format one text input with explicit voice and optional server parameters."""
        if not request_info.turns:
            raise ValueError("Speech endpoint requires at least one turn")
        turn = request_info.turns[-1]
        if len(turn.texts) != 1 or len(turn.texts[0].contents) != 1:
            raise ValueError("Speech endpoint requires exactly one text input")
        payload = {
            "model": turn.model or request_info.model_endpoint.primary_model_name,
            "input": turn.texts[0].contents[0],
            "response_format": "wav",
        }
        payload.update(request_info.model_endpoint.endpoint.extra)
        payload.update(turn.extra_body or {})
        if not isinstance(payload.get("input"), str) or not payload["input"].strip():
            raise ValueError("Speech endpoint requires nonempty input text")
        if not payload.get("voice"):
            raise ValueError(
                "Speech endpoint requires voice via --extra-inputs or dataset extra"
            )
        if payload["response_format"] not in {"pcm", "wav"}:
            raise ValueError(
                "Speech endpoint supports only pcm and wav response_format"
            )
        if payload.get("stream") or payload.get("stream_format", "audio") != "audio":
            raise ValueError(
                "Speech endpoint currently supports complete audio responses only"
            )
        if (
            payload["response_format"] == "pcm"
            and self.model_endpoint.endpoint.pcm_output is None
        ):
            raise ValueError(
                "PCM responses require benchmark.endpoint.pcm_output in --config"
            )
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Accept only complete, nonempty PCM sample frames or finite-length PCM WAV."""
        if not isinstance(response, BinaryResponse):
            return None
        content_type = (response.content_type or "").split(";", 1)[0].strip().lower()
        body = response.raw_bytes
        if not body:
            return None
        is_wav = body[:4] == b"RIFF" and body[8:12] == b"WAVE"
        if content_type in _WAV_TYPES or (
            content_type == "application/octet-stream" and is_wav
        ):
            audio = self._parse_wav(body)
        elif content_type in {"audio/pcm", "application/octet-stream"}:
            pcm = self.model_endpoint.endpoint.pcm_output
            if pcm is None or len(body) % (2 * pcm.channels):
                return None
            audio = AudioResponseData(
                frames=len(body) // (2 * pcm.channels),
                sample_rate=pcm.sample_rate,
                channels=pcm.channels,
            )
        else:
            return None
        return ParsedResponse(perf_ns=response.perf_ns, data=audio) if audio else None

    @staticmethod
    def _parse_wav(body: bytes) -> AudioResponseData | None:
        """Check the declared container and sample counts before trusting duration."""
        if len(body) < 12 or int.from_bytes(body[4:8], "little") + 8 != len(body):
            return None
        try:
            with wave.open(io.BytesIO(body), "rb") as wav:
                frames = wav.getnframes()
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                expected_bytes = frames * channels * wav.getsampwidth()
                if (
                    frames <= 0
                    or sample_rate <= 0
                    or len(wav.readframes(frames + 1)) != expected_bytes
                ):
                    return None
                return AudioResponseData(
                    frames=frames, sample_rate=sample_rate, channels=channels
                )
        except (wave.Error, EOFError):
            return None
