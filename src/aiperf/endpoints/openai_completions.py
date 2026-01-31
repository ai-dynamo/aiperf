# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.common.types import JsonObject, RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


class CompletionsEndpoint(BaseEndpoint):
    """OpenAI Completions endpoint.

    Supports text completions with streaming.
    """

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for a completions request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Completions API payload
        """
        if len(request_info.turns) != 1:
            raise ValueError("Completions endpoint only supports one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]

        extra = model_endpoint.endpoint.extra or []

        payload = {
            "prompt": prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens:
            payload["max_tokens"] = turn.max_tokens

        if extra:
            payload.update(extra)

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
            # Automatically set stream_options to include usage when using server token counts
            if "stream_options" not in payload:
                payload["stream_options"] = {"include_usage": True}
            elif (
                isinstance(payload["stream_options"], dict)
                and "include_usage" not in payload["stream_options"]
            ):
                payload["stream_options"]["include_usage"] = True

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        data = self.extract_completions_response_data(json_obj)
        usage = json_obj.get("usage") or None

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def extract_completions_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI Completions JSON response.

        Handles both text_completion and completion object types.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted text data or None if no content
        """
        match json_obj.get("object"):
            case "completion" | "text_completion":
                choices = json_obj.get("choices")
                if not choices:
                    self.debug(lambda: f"No choices found in response: {json_obj}")
                    return None
                return self.make_text_response_data(choices[0].get("text"))
            case _:
                object_type = json_obj.get("object")
                raise ValueError(f"Unsupported OpenAI object type: {object_type!r}")
