# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint

_MIME_BY_FORMAT: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "webm": "audio/webm",
}

# File field is managed by the endpoint; everything else can come from --extra-inputs.
_RESERVED_PAYLOAD_KEYS: frozenset[str] = frozenset({"file"})


class AudioTranscriptionEndpoint(BaseEndpoint):
    """OpenAI Audio Transcription endpoint (/v1/audio/transcriptions).

    Sends audio as a multipart file upload and returns a plain-text transcript.
    Compatible with Whisper-style servers (OpenAI, NIM, SGLang).

    Audio is sourced from ``turn.audios`` in the ``"<fmt>,<b64>"`` format
    produced by HFASRDatasetLoader (e.g. ``"wav,<base64>..."``).

    Use ``--endpoint-type audio_transcription`` with an ASR dataset loader
    (``--dataset librispeech``, ``--dataset voxpopuli``, etc.).
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        if not request_info.turns:
            raise ValueError("Audio transcription endpoint requires at least one turn.")

        turn = request_info.turns[-1]
        model_endpoint = request_info.model_endpoint

        if not turn.audios or not turn.audios[0].contents:
            raise ValueError(
                "Audio transcription endpoint requires audio in turn.audios[0]."
            )

        audio_content = turn.audios[0].contents[0]
        if not audio_content:
            raise ValueError("Audio content is empty.")

        payload: dict[str, Any] = {
            "file": _build_audio_field(audio_content),
            "model": turn.model or model_endpoint.primary_model_name,
        }

        for key, value in model_endpoint.endpoint.extra or []:
            if key in _RESERVED_PAYLOAD_KEYS:
                self.warning(
                    f"--extra-inputs {key!r} is managed by the endpoint and was ignored."
                )
                continue
            payload[key] = value

        for key, value in (turn.extra_body or {}).items():
            if key in _RESERVED_PAYLOAD_KEYS:
                self.warning(
                    f"extra_body {key!r} is managed by the endpoint and was ignored."
                )
                continue
            payload[key] = value

        self.trace(
            lambda: f"Formatted audio transcription payload keys: {list(payload)}"
        )
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        json_obj = response.get_json()
        if not json_obj:
            return None
        text = json_obj.get("text")
        if text is None:
            return None
        usage = json_obj.get("usage") or None
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(text),
            usage=usage,
        )


def _build_audio_field(format_and_b64: str) -> dict[str, Any]:
    """Convert a ``"<fmt>,<b64>"`` audio string into a multipart file descriptor."""
    if "," not in format_and_b64:
        raise ValueError(
            f"audio content must be in the format '<fmt>,<b64>' "
            f"(got {format_and_b64[:40]!r}); "
            f"expected e.g. 'wav,<base64>'"
        )
    fmt, b64 = format_and_b64.split(",", 1)
    mime = _MIME_BY_FORMAT.get(fmt.lower(), f"audio/{fmt.lower()}")
    return {
        "b64_data": b64,
        "filename": f"audio.{fmt.lower()}",
        "content_type": mime,
    }
