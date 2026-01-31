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
        model_endpoint = request_info.model_endpoint

        inputs = " ".join(
            [content for text in turn.texts for content in text.contents if content]
        )

        parameters: dict[str, Any] = {}
        if turn.max_tokens is not None:
            parameters["max_new_tokens"] = turn.max_tokens

        if model_endpoint.endpoint.extra:
            parameters.update(model_endpoint.endpoint.extra)

        payload: dict[str, Any] = {
            "inputs": inputs,
            "parameters": parameters,
        }

        self.trace(lambda: f"Formatted TGI payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse TGI response into ParsedResponse.

        Handles both streaming and non-streaming modes.
        """
        if self.model_endpoint.endpoint.streaming:
            return self._parse_streaming(response)
        return self._parse_non_streaming(response)

    def _parse_non_streaming(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Handle standard (non-streaming) JSON response."""
        json_obj = response.get_json()
        if not json_obj:
            return None

        if isinstance(json_obj, list) and json_obj:
            text = json_obj[0].get("generated_text")
        else:
            text = json_obj.get("generated_text")

        if not text:
            self.debug(lambda: f"No generated_text in response: {json_obj}")
            return None

        data = self.make_text_response_data(text)
        return ParsedResponse(perf_ns=response.perf_ns, data=data)

    def _parse_streaming(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Hugging Face TGI streaming response (single-packet version)."""
        try:
            chunks: list[str] = []

            json_obj = response.get_json()
            if not json_obj:
                self.debug("Empty or invalid streaming JSON response.")
                return None

            token_obj = json_obj.get("token")
            if token_obj and (text := token_obj.get("text")):
                chunks.append(text)

            if text := json_obj.get("generated_text"):
                chunks.append(text)

            if not chunks:
                self.debug(lambda: "No text chunks collected from TGI stream.")
                return None

            data = self.make_text_response_data("".join(chunks))
            return ParsedResponse(perf_ns=response.perf_ns, data=data)

        except Exception as e:
            self.debug(lambda e=e: f"Error parsing TGI stream: {e!r}")
            return None
