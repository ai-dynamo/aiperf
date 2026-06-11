# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.exceptions import InferenceClientError
from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    ToolCallResponseData,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint


class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses. Message-array construction
    uses the generic ``BaseEndpoint.build_messages`` flow — the default
    ``_render_*_part`` hooks already emit OpenAI chat shape, so nothing
    needs overriding here.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo."""
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        # Prepend the shared system + per-conversation user-context prompts
        # (both live on RequestInfo), then flatten turns via the generic
        # build_messages skeleton.
        messages: list[dict[str, Any]] = []
        if request_info.system_message:
            messages.append({"role": "system", "content": request_info.system_message})
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(self.build_messages(turns))

        payload: dict[str, Any] = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].raw_tools is not None:
            payload["tools"] = turns[-1].raw_tools

        if turns[-1].max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if turns[-1].extra_body:
            payload.update(turns[-1].extra_body)

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
        """Parse OpenAI Chat Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        if error := json_obj.get("error"):
            if isinstance(error, dict):
                message = error.get("message") or str(error)
                error_type = error.get("type")
                error_code = error.get("code")
                details = ", ".join(
                    str(value)
                    for value in (error_type, error_code)
                    if value is not None
                )
                if details:
                    message = f"{message} ({details})"
            else:
                message = str(error)
            raise InferenceClientError(f"Inference server error: {message}")

        usage = json_obj.get("usage") or None
        data = (
            self.extract_chat_response_data(json_obj)
            if json_obj.get("choices")
            else None
        )

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def extract_chat_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted response data or None if no content
        """
        match json_obj.get("object"):
            case "chat.completion":
                data_key = "message"
            case "chat.completion.chunk":
                data_key = "delta"
            case _:
                object_type = json_obj.get("object")
                raise ValueError(f"Unsupported OpenAI object type: {object_type!r}")

        choices = json_obj.get("choices")
        if not choices:
            self.debug(lambda: f"No choices found in response: {json_obj}")
            return None

        data = choices[0].get(data_key)
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")

        if reasoning:
            return ReasoningResponseData(content=content, reasoning=reasoning)

        if content:
            return self.make_text_response_data(content)

        tool_calls = data.get("tool_calls") or []
        tool_call_parts: list[str] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "")
            if name:
                tool_call_parts.append(name)
            if arguments:
                tool_call_parts.append(arguments)
        tool_call_text = "".join(tool_call_parts)
        if tool_call_text:
            return ToolCallResponseData(text=tool_call_text)

        return None
