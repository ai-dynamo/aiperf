# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    RequestRecord,
    TokenIdsResponseData,
    Turn,
    Usage,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class SGLangGenerateEndpoint(BaseEndpoint):
    """Native SGLang-compatible streaming ``/generate`` token endpoint."""

    @staticmethod
    def _collect_input_ids(request_info: RequestInfo) -> list[int]:
        """Collect exact cumulative context while honoring reset boundaries."""
        input_ids: list[int] = []
        for turn in request_info.turns:
            if turn.reset_context:
                input_ids.clear()
            if turn.token_ids is None:
                raise ValueError(
                    "The sglang_generate endpoint requires token_ids on every "
                    "request and captured assistant turn."
                )
            input_ids.extend(turn.token_ids)
        if not input_ids:
            raise ValueError(
                "The sglang_generate endpoint requires non-empty input_ids."
            )
        return input_ids

    @staticmethod
    def _merge_sampling_params(target: dict[str, Any], value: Any, source: str) -> None:
        """Validate and merge sampling parameters from one override source."""
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError(f"{source} sampling_params must be a JSON object.")
        target.update(value)

    def _build_sampling_params(
        self,
        request_info: RequestInfo,
        endpoint_extra: dict[str, Any],
        turn_extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge endpoint and turn sampling configuration in precedence order."""
        sampling_params: dict[str, Any] = {"ignore_eos": True}
        self._merge_sampling_params(
            sampling_params,
            endpoint_extra.pop("sampling_params", None),
            "Endpoint",
        )
        current_turn = request_info.turns[-1]
        if current_turn.max_tokens is not None:
            sampling_params["max_new_tokens"] = current_turn.max_tokens
        self._merge_sampling_params(
            sampling_params,
            turn_extra.pop("sampling_params", None),
            "Turn",
        )
        return sampling_params

    @staticmethod
    def _extract_priority(
        endpoint_extra: dict[str, Any], turn_extra: dict[str, Any]
    ) -> Any:
        """Translate Mooncake strict priority into SGLang's top-level field."""
        nvext = turn_extra.pop("nvext", None)
        priority = turn_extra.pop("priority", endpoint_extra.pop("priority", None))
        if isinstance(nvext, dict):
            agent_hints = nvext.get("agent_hints")
            if isinstance(agent_hints, dict) and "strict_priority" in agent_hints:
                priority = agent_hints["strict_priority"]
        return priority

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Build one cumulative token-input request for the active session."""
        if not request_info.model_endpoint.endpoint.streaming:
            raise ValueError(
                "The sglang_generate endpoint requires streaming. Add --streaming."
            )
        if request_info.system_message or request_info.user_context_message:
            raise ValueError(
                "The sglang_generate endpoint cannot tokenize conversation-level "
                "system or user-context messages. Put their tokenized content in "
                "the dataset turns instead."
            )

        input_ids = self._collect_input_ids(request_info)
        endpoint_extra = dict(request_info.model_endpoint.endpoint.extra or [])
        turn_extra = dict(request_info.turns[-1].extra_body or {})
        sampling_params = self._build_sampling_params(
            request_info, endpoint_extra, turn_extra
        )
        priority = self._extract_priority(endpoint_extra, turn_extra)

        payload = endpoint_extra
        payload.update(turn_extra)
        payload.update(
            {
                "rid": request_info.x_request_id,
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "stream": True,
            }
        )
        if priority is not None:
            payload["priority"] = priority

        self.trace(lambda: f"Formatted SGLang /generate payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse one native SGLang SSE event containing incremental token IDs."""
        json_obj = response.get_json()
        if not isinstance(json_obj, dict):
            return None

        output_ids = json_obj.get("output_ids", [])
        if not isinstance(output_ids, list) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in output_ids
        ):
            self.debug(lambda: f"Invalid SGLang output_ids: {output_ids!r}")
            return None

        meta_info = json_obj.get("meta_info")
        usage = None
        metadata: dict[str, Any] = {}
        if isinstance(meta_info, dict):
            prompt_tokens = meta_info.get("prompt_tokens")
            completion_tokens = meta_info.get("completion_tokens")
            usage_data: dict[str, int] = {}
            if isinstance(prompt_tokens, int):
                usage_data["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                usage_data["completion_tokens"] = completion_tokens
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                usage_data["total_tokens"] = prompt_tokens + completion_tokens
            usage = Usage(usage_data) if usage_data else None
            metadata = {
                key: meta_info[key]
                for key in ("id", "finish_reason")
                if key in meta_info
            }

        data = TokenIdsResponseData(token_ids=output_ids) if output_ids else None
        if data is None and usage is None:
            return None
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=data,
            usage=usage,
            metadata=metadata,
        )

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture raw output IDs so the next user turn replays exact context."""
        output_ids = [
            token_id
            for response in self.extract_response_data(record)
            if isinstance(response.data, TokenIdsResponseData)
            for token_id in response.data.token_ids
        ]
        if not output_ids:
            return None
        return Turn(role="assistant", token_ids=output_ids)

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        """Report exact pre-tokenized ISL without invoking a tokenizer."""
        input_ids = payload.get("input_ids")
        return ExtractedPayload(
            pretokenised_token_count=len(input_ids)
            if isinstance(input_ids, list)
            else 0
        )
