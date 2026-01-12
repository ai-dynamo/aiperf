# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import (
    ParsedResponse,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import EmbeddingResponseData, RequestInfo
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.EMBEDDINGS)
class EmbeddingsEndpoint(BaseEndpoint):
    """OpenAI Embeddings endpoint.

    Generates vector embeddings for text and/or image inputs.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Embeddings endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/embeddings",
            supports_streaming=False,
            produces_tokens=False,
            tokenizes_input=True,
            metrics_title="Embeddings Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format payload for an embeddings request.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Embeddings API payload
        """
        if len(request_info.turns) != 1:
            raise ValueError("Embeddings endpoint only supports one turn.")

        # Use first turn (hardcoded for now)
        turn = request_info.turns[0]

        if turn.max_tokens:
            self.error("Max_tokens is provided but is not supported for embeddings.")

        # Extract text contents
        texts = [
            content for text in turn.texts for content in text.contents if content
        ]

        # Extract images (list of data URL strings)
        images = [
            image_content 
            for image in turn.images for image_content in image.contents if image_content
        ]

        # Build input based on what's provided
        if texts and images:
            # Both text and images provided - must be same length
            if len(texts) != len(images):
                raise ValueError(
                    f"When both texts and images are provided, they must have the same length. "
                    f"Got {len(texts)} texts and {len(images)} images."
                )
            # Combine as "text <img src='data_url'/>"
            inputs = [
                f"{text} <img src='{image}'/>" for text, image in zip(texts, images)
            ]
        elif images:
            # Only images provided - wrap each in an img tag
            inputs = [f"<img src='{image}'/>" for image in images]
        else:
            # Only text provided (or nothing)
            inputs = texts

        model_endpoint = request_info.model_endpoint

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
