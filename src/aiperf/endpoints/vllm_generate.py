# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""vLLM token-in/token-out ``/inference/v1/generate`` endpoint."""

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    BaseResponseData,
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
    RequestRecord,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class VllmGenerateEndpoint(BaseEndpoint):
    """Send and measure vLLM ``GenerateRequest`` token arrays.

    The endpoint is intentionally non-streaming so the same payload works with
    vLLM and Dynamo's vLLM-compatible engine API. Use ``--endpoint-path`` only
    when the server mounts the API at a custom path.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        if len(request_info.turns) != 1:
            raise ValueError(
                "vLLM generate endpoint requires one token payload per turn"
            )

        turn = request_info.turns[0]
        extra_body = dict(turn.extra_body or {})
        token_ids = extra_body.pop("token_ids", None)
        if not self._valid_token_ids(token_ids):
            raise ValueError(
                "turn.extra_body.token_ids must be a non-empty list of integers"
            )

        sampling_params = dict(extra_body.pop("sampling_params", {}) or {})
        if turn.max_tokens is not None:
            sampling_params.setdefault("max_tokens", turn.max_tokens)
        if extra_body.pop("stream", False):
            raise ValueError("vLLM generate endpoint does not support streaming")

        payload: dict[str, Any] = {
            "model": turn.model or request_info.model_endpoint.primary_model_name,
            "token_ids": token_ids,
            "sampling_params": sampling_params,
            "stream": False,
        }
        if request_info.x_request_id:
            payload["request_id"] = request_info.x_request_id
        payload.update(dict(self.model_endpoint.endpoint.extra or []))
        payload.update(extra_body)
        if payload.get("stream") is not False:
            raise ValueError("vLLM generate endpoint requires stream=false")
        return payload

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        result = ExtractedPayload()
        token_ids = payload.get("token_ids")
        if self._valid_token_ids(token_ids):
            result.pretokenised_token_count = len(token_ids)
        return result

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        return self._parse_response(response, prompt_tokens=None)

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        prompt_tokens = self._prompt_token_count(record)
        return [
            parsed
            for response in record.responses or []
            if (parsed := self._parse_response(response, prompt_tokens)) is not None
        ]

    def _parse_response(
        self,
        response: InferenceServerResponse,
        prompt_tokens: int | None,
    ) -> ParsedResponse | None:
        payload = response.get_json()
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            return None
        completion_ids = choices[0].get("token_ids")
        if not isinstance(completion_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in completion_ids
        ):
            return None

        completion_tokens = len(completion_ids)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if prompt_tokens is not None else None
            ),
        }
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=BaseResponseData(),
            usage=usage,
            metadata={
                "request_id": payload.get("request_id"),
                "finish_reason": choices[0].get("finish_reason"),
                "completion_token_ids": completion_ids,
            },
        )

    @classmethod
    def _prompt_token_count(cls, record: RequestRecord) -> int | None:
        if not record.turns:
            return None
        turn = record.turns[-1]
        payload = turn.raw_payload or turn.extra_body or {}
        token_ids = payload.get("token_ids") if isinstance(payload, dict) else None
        return len(token_ids) if cls._valid_token_ids(token_ids) else None

    @staticmethod
    def _valid_token_ids(value: Any) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in value
            )
        )
