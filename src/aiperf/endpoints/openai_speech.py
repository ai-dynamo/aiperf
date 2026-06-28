# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
from typing import Any

from aiperf.common.models import (
    AudioResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint

# Short format tags for the audio MIME types TTS servers commonly return.
_AUDIO_FORMAT_BY_MIME: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/opus": "opus",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
    "audio/pcm": "pcm",
}


def _format_from_content_type(content_type: str | None) -> str | None:
    """Map an ``audio/*`` content type to a short format tag (e.g. mp3, wav)."""
    if not content_type:
        return None
    base = content_type.split(";", 1)[0].strip().lower()
    if base in _AUDIO_FORMAT_BY_MIME:
        return _AUDIO_FORMAT_BY_MIME[base]
    if base.startswith("audio/"):
        return base[len("audio/") :] or None
    return None


class OpenAISpeechEndpoint(BaseEndpoint):
    """OpenAI-compatible text-to-speech endpoint (/v1/audio/speech).

    Sends text and returns synthesized audio - either a full clip
    (non-streaming binary body) or a stream of audio chunks. When streaming
    is enabled the payload requests Server-Sent Events (``stream_format:
    sse``) so per-chunk timing (time-to-first-audio) is captured; servers
    that instead stream a raw chunked audio body are handled by the
    transport's streamed-binary path and also parsed here as binary chunks.

    Voice, response format, and speed are passed via ``--extra-inputs``
    (e.g. ``--extra-inputs voice:alloy response_format:wav speed:1.0``),
    mirroring how the image-generation endpoint takes ``size``/``quality``.

    See: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    DEFAULT_VOICE = "alloy"
    DEFAULT_RESPONSE_FORMAT = "mp3"

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format an OpenAI /v1/audio/speech request payload from RequestInfo.

        - input (required): text to synthesize, from ``turn.texts[0]``
        - model (optional): from ``turn.model`` or the endpoint's primary model
        - voice / response_format: defaulted; override via ``--extra-inputs``
        - stream_format: set to ``sse`` when streaming is enabled
        - speed and any other server tunables: pass via ``--extra-inputs``
        """
        if not request_info.turns:
            raise ValueError("Speech endpoint requires at least one turn.")

        turn = request_info.turns[-1]
        model_endpoint = request_info.model_endpoint

        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("Speech endpoint requires a text input to synthesize.")

        text = turn.texts[0].contents[0]

        payload: dict[str, Any] = {
            "input": text,
            "model": turn.model or model_endpoint.primary_model_name,
            "voice": self.DEFAULT_VOICE,
            "response_format": self.DEFAULT_RESPONSE_FORMAT,
        }

        if model_endpoint.endpoint.streaming:
            # SSE audio deltas give per-chunk timing for TTFA plus a final
            # usage event. Overridable via --extra-inputs stream_format:audio.
            payload["stream_format"] = "sse"

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if turn.extra_body:
            payload.update(turn.extra_body)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse one speech response unit into an AudioResponseData chunk.

        Handles three shapes: a non-streaming binary clip, a streamed binary
        chunk (both arrive as raw bytes), and an SSE ``speech.audio.delta``
        (base64 audio). The final ``speech.audio.done`` event carries usage
        only and is returned with no data so it is excluded from content
        responses but still contributes token usage.
        """
        # Non-streaming clip or a single streamed-binary chunk: raw audio bytes.
        raw = response.get_raw()
        if isinstance(raw, bytes):
            if not raw:
                return None
            fmt = _format_from_content_type(getattr(response, "content_type", None))
            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=AudioResponseData(audio_bytes=raw, format=fmt),
            )

        # Streaming SSE: speech.audio.delta (base64 audio) / speech.audio.done (usage).
        json_obj = response.get_json()
        if not json_obj:
            return None

        audio_b64 = json_obj.get("audio")
        if audio_b64:
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except (ValueError, TypeError):
                self.debug(
                    lambda: "Failed to base64-decode audio delta; skipping chunk."
                )
                return None
            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=AudioResponseData(audio_bytes=audio_bytes),
            )

        # Usage-only event (e.g. speech.audio.done) - no audio data.
        usage = json_obj.get("usage") or None
        if usage:
            return ParsedResponse(perf_ns=response.perf_ns, usage=usage)

        return None
