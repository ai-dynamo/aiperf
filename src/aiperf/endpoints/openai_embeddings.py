# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import (
    EmbeddingResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_embeddings_endpoint import BaseEmbeddingsEndpoint


class EmbeddingsEndpoint(BaseEmbeddingsEndpoint):
    """OpenAI Embeddings endpoint.

    Generates vector embeddings for text inputs.
    """

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for an embeddings request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Embeddings API payload
        """
        turn = self._validate_and_get_turn(request_info)
        inputs = self._extract_texts(turn)

        return self._build_payload(turn, input=inputs)

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Embeddings response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted embeddings
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(
                lambda: f"No JSON object found in response: {response.get_raw()}"
            )
            return None

        data = json_obj.get("data", [])
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        if all(
            isinstance(item, dict) and item.get("object") == "embedding"
            for item in data
        ):
            embeddings = [
                item.get("embedding")
                for item in data
                if item.get("embedding") is not None
            ]
            if not embeddings:
                return None
            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=EmbeddingResponseData(embeddings=embeddings),
            )

        else:
            raise ValueError(f"Received invalid list in response: {json_obj}")
