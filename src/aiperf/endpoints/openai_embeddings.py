# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    EmbeddingResponseData,
    InferenceServerResponse,
    ModelEndpointInfo,
    ParsedResponse,
    RequestInfo,
    Turn,
)
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


class EmbeddingsEndpoint(BaseEndpoint):
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

        # Extract text contents
        inputs = [
            content for text in turn.texts for content in text.contents if content
        ]

        return self._build_payload(turn, request_info.model_endpoint, inputs)

    def _validate_and_get_turn(self, request_info: RequestInfo):
        """Validate request and return the single turn.

        Args:
            request_info: Request context including turns

        Returns:
            The single turn from the request

        Raises:
            ValueError: If request doesn't contain exactly one turn
        """
        if len(request_info.turns) != 1:
            raise ValueError("Embeddings endpoint only supports one turn.")

        turn = request_info.turns[0]

        if turn.max_tokens:
            self.error("Max_tokens is provided but is not supported for embeddings.")

        return turn

    def _build_payload(
        self, turn: Turn, model_endpoint: ModelEndpointInfo, inputs: list[str]
    ) -> dict[str, Any]:
        """Build the final payload dictionary.

        Args:
            turn: The validated turn containing model override
            model_endpoint: Model endpoint info with primary model name and extra config
            inputs: List of input strings to embed

        Returns:
            OpenAI Embeddings API payload
        """
        payload: dict[str, Any] = {
            "model": turn.model or model_endpoint.primary_model_name,
            "input": inputs,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

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
