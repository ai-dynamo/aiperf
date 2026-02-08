# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    EmbeddingResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint


class KServeV2EmbeddingsEndpoint(BaseEndpoint):
    """KServe V2 Open Inference Protocol endpoint for embedding models.

    Wraps text as BYTES tensors per the V2 inference protocol spec.
    Tensor names are configurable via --extra v2_input_name:X --extra v2_output_name:Y.
    """

    DEFAULT_INPUT_NAME = "text_input"
    DEFAULT_OUTPUT_NAME = "embedding_output"

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
        """Format V2 inference request with BYTES tensor inputs for embeddings.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V2 Open Inference Protocol payload

        Raises:
            ValueError: If request doesn't contain exactly one turn
        """
        if len(request_info.turns) != 1:
            raise ValueError("Embeddings endpoint only supports one turn.")

        turn = request_info.turns[0]

        if turn.max_tokens is not None:
            self.warning("Max_tokens is provided but is not supported for embeddings.")

        texts = [content for text in turn.texts for content in text.contents if content]

        inputs: list[dict[str, Any]] = [
            {
                "name": self._input_name,
                "shape": [len(texts)],
                "datatype": "BYTES",
                "data": texts,
            }
        ]

        payload: dict[str, Any] = {"inputs": inputs}

        if self._extra_params:
            payload["parameters"] = dict(self._extra_params)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse V2 inference response, extracting embeddings from FP32 output tensor.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted embeddings, or None if no content
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        outputs = json_obj.get("outputs")
        if not outputs:
            return None

        # Find output tensor by name, fallback to first
        output = None
        for o in outputs:
            if o.get("name") == self._output_name:
                output = o
                break
        if output is None:
            output = outputs[0]

        data = output.get("data")
        if not data:
            return None

        shape = output.get("shape", [])
        embeddings = self._reshape_embeddings(data, shape)

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=EmbeddingResponseData(embeddings=embeddings),
        )

    @staticmethod
    def _reshape_embeddings(
        flat_data: list[float], shape: list[int]
    ) -> list[list[float]]:
        """Reshape flat float array into embedding vectors using shape metadata.

        Args:
            flat_data: Flat array of float values from the output tensor
            shape: Tensor shape, e.g. [N, D] for N embeddings of dimension D

        Returns:
            List of embedding vectors
        """
        if len(shape) == 2:
            n, d = shape
            return [flat_data[i * d : (i + 1) * d] for i in range(n)]
        # Single-dim or unknown shape: treat entire array as one embedding
        return [flat_data]
