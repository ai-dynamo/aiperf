# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class KServeV2InferEndpoint(BaseEndpoint):
    """KServe V2 Open Inference Protocol endpoint for Triton/TRT-LLM.

    Wraps text as BYTES tensors per the V2 inference protocol spec.
    Tensor names are configurable via --extra v2_input_name:X --extra v2_output_name:Y.
    """

    DEFAULT_INPUT_NAME = "text_input"
    DEFAULT_OUTPUT_NAME = "text_output"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = (
            dict(self.model_endpoint.endpoint.extra)
            if self.model_endpoint.endpoint.extra
            else {}
        )
        self._input_name: str = extra.pop("v2_input_name", self.DEFAULT_INPUT_NAME)
        self._output_name: str = extra.pop("v2_output_name", self.DEFAULT_OUTPUT_NAME)
        self._extra_params: dict[str, Any] = extra

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format V2 inference request with BYTES tensor inputs.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V2 Open Inference Protocol payload
        """
        if not request_info.turns:
            raise ValueError("KServe V2 endpoint requires at least one turn.")

        turn = request_info.turns[0]

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        prompt = " ".join(prompts) if prompts else ""

        inputs: list[dict[str, Any]] = [
            {
                "name": self._input_name,
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt],
            }
        ]

        if turn.max_tokens is not None:
            inputs.append(
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [turn.max_tokens],
                }
            )

        payload: dict[str, Any] = {"inputs": inputs}

        if self._extra_params:
            payload["parameters"] = dict(self._extra_params)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse V2 inference response, extracting text from BYTES output tensor.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content, or None if no content
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        outputs = json_obj.get("outputs")
        if not outputs:
            return None

        for output in outputs:
            if output.get("name") == self._output_name:
                data = output.get("data")
                if isinstance(data, list) and len(data) > 0 and data[0] is not None:
                    text = str(data[0])
                    return ParsedResponse(
                        perf_ns=response.perf_ns,
                        data=self.make_text_response_data(text),
                    )

        # Fallback: try first output with data
        for output in outputs:
            data = output.get("data")
            if isinstance(data, list) and len(data) > 0 and data[0] is not None:
                text = str(data[0])
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=self.make_text_response_data(text),
                )

        return None
