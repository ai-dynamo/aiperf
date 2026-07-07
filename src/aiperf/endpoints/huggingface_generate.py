# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import InferenceServerResponse, ParsedResponse, RequestInfo
from aiperf.endpoints.base_endpoint import BaseEndpoint


class HuggingFaceGenerateEndpoint(BaseEndpoint):
    """Hugging Face TGI (Text Generation Inference) endpoint.

    Supports both non-streaming (/ or /generate) and streaming (/generate_stream)
    endpoints automatically, based on the model endpoint's `streaming` flag.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for Hugging Face TGI request."""
        if len(request_info.turns) != 1:
            raise ValueError("TGI endpoint supports a single turn per request.")

        turn = request_info.turns[0]

        inputs = " ".join(
            [content for text in turn.texts for content in text.contents if content]
        )

        parameters: dict[str, Any] = {}
        if turn.max_tokens is not None:
            parameters["max_new_tokens"] = turn.max_tokens

        if self.run.cfg.endpoint.extra:
            parameters.update(self.run.cfg.endpoint.extra)

        payload: dict[str, Any] = {
            "inputs": inputs,
            "parameters": parameters,
        }

        if turn.extra_body:
            payload.update(turn.extra_body)

        self.trace(lambda: f"Formatted TGI payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse TGI response into ParsedResponse.

        Handles both streaming and non-streaming modes.
        """
        if self.run.cfg.endpoint.streaming:
            return self._parse_streaming(response)
        return self._parse_non_streaming(response)

    def _parse_non_streaming(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Handle standard (non-streaming) JSON response."""
        json_obj = response.get_json()
        if not json_obj:
            return None

        # TGI returns either ``{"generated_text": ...}`` or ``[{"generated_text":
        # ...}]``. A malformed first list element (``[None]``, ``['x']``, ``[5]``)
        # or a non-dict top-level body would crash ``.get(...)`` on the worker's
        # unconditional post-response parse; degrade to None instead (the
        # streaming path already has a try/except; this one did not).
        if isinstance(json_obj, list):
            first = json_obj[0] if json_obj else None
            text = first.get("generated_text") if isinstance(first, dict) else None
        elif isinstance(json_obj, dict):
            text = json_obj.get("generated_text")
        else:
            text = None

        if not text:
            self.debug(lambda: f"No generated_text in response: {json_obj}")
            return None

        data = self.make_text_response_data(text)
        return ParsedResponse(perf_ns=response.perf_ns, data=data)

    def _parse_streaming(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Hugging Face TGI streaming response (single SSE event).

        Each event is parsed independently; the caller accumulates across events.
        Always use token.text (the incremental token), never generated_text
        (which is the full accumulated text on the final event and would duplicate
        all prior tokens).
        """
        try:
            json_obj = response.get_json()
            if not json_obj:
                self.debug("Empty or invalid streaming JSON response.")
                return None

            token_obj = json_obj.get("token")
            text = token_obj.get("text") if token_obj else None

            if not text:
                self.debug("No token text in TGI stream event.")
                return None

            data = self.make_text_response_data(text)
            return ParsedResponse(perf_ns=response.perf_ns, data=data)

        except Exception as e:  # noqa: BLE001 - TGI event parsing must never crash the stream; malformed events are logged and skipped
            self.debug(lambda e=e: f"Error parsing TGI stream: {e!r}")
            return None
