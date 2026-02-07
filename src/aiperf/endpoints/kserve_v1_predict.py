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


class KServeV1PredictEndpoint(BaseEndpoint):
    """KServe V1 Inference Protocol (TensorFlow Serving style) endpoint.

    Uses instances/predictions format. Field names are configurable via
    --extra v1_input_field:X --extra v1_output_field:Y.
    """

    DEFAULT_INPUT_FIELD = "text"
    DEFAULT_OUTPUT_FIELD = "output"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = (
            dict(self.model_endpoint.endpoint.extra)
            if self.model_endpoint.endpoint.extra
            else {}
        )
        self._input_field: str = extra.pop("v1_input_field", self.DEFAULT_INPUT_FIELD)
        self._output_field: str = extra.pop(
            "v1_output_field", self.DEFAULT_OUTPUT_FIELD
        )

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format V1 predict request with instances format.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V1 predict payload: {"instances": [{input_field: "text"}]}
        """
        if not request_info.turns:
            raise ValueError("KServe V1 endpoint requires at least one turn.")

        turn = request_info.turns[0]

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        prompt = " ".join(prompts) if prompts else ""

        payload: dict[str, Any] = {
            "instances": [{self._input_field: prompt}],
        }

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse V1 predict response, extracting text from predictions.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text content, or None if no content
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        predictions = json_obj.get("predictions")
        if not predictions or not isinstance(predictions, list):
            # Fallback to auto-detection for flexible response handling
            data = self.auto_detect_and_extract(json_obj)
            if data:
                return ParsedResponse(perf_ns=response.perf_ns, data=data)
            return None

        first = predictions[0]

        # Handle dict predictions: {"predictions": [{"output": "text"}]}
        if isinstance(first, dict):
            text = first.get(self._output_field)
            if text and isinstance(text, str):
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=self.make_text_response_data(text),
                )
            # Try auto-detection on the prediction dict
            data = self.auto_detect_and_extract(first)
            if data:
                return ParsedResponse(perf_ns=response.perf_ns, data=data)

        # Handle scalar predictions: {"predictions": ["text"]}
        if isinstance(first, str) and first:
            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=self.make_text_response_data(first),
            )

        return None
