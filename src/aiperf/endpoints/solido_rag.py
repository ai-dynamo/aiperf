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
from aiperf.common.models.record_models import RAGSources, RequestInfo, TextResponseData
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.common.types import JsonObject, RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.SOLIDO_RAG)
class SolidoEndpoint(BaseEndpoint):
    """SOLIDO RAG endpoint.

    SOLIDO is a RAG (Retrieval-Augmented Generation) endpoint that processes
    queries with filters and inference model specifications. Supports streaming
    responses.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return SOLIDO endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/rag/api/prompt",
            supports_streaming=True,
            produces_tokens=True,
            tokenizes_input=True,
            metrics_title="SOLIDO RAG Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format SOLIDO RAG request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            SOLIDO API payload with query, filters, and inference_model fields
        """
        if not request_info.turns:
            raise ValueError("SOLIDO endpoint requires at least one turn.")

        turn = request_info.turns[-1]
        model_endpoint = request_info.model_endpoint

        # Extract query text from turn
        query = [content for text in turn.texts for content in text.contents if content]

        # Default filters for SOLIDO
        filters = {"family": "Solido", "tool": "SDE"}

        # Use the model name from the turn or model endpoint
        inference_model = turn.model or model_endpoint.primary_model_name

        payload: dict[str, Any] = {
            "query": query,
            "filters": filters,
            "inference_model": inference_model,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.trace(lambda: f"Formatted SOLIDO payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse SOLIDO API response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted content or None if parsing fails
        """
        json_obj = response.get_json()
        if not json_obj:
            self.debug(lambda: f"No JSON in response: {response.get_raw()}")
            return None

        data, sources = self._extract_solido_response_data(json_obj)
        return (
            ParsedResponse(perf_ns=response.perf_ns, data=data, sources=sources)
            if data
            else None
        )

    def _extract_solido_response_data(
        self, json_obj: JsonObject
    ) -> tuple[TextResponseData, RAGSources | None]:
        """Extract content from SOLIDO JSON response.

        Args:
            json_obj: Deserialized SOLIDO response

        Returns:
            Extracted response data or None if no content
        """
        # SOLIDO responses contain a "content" field with the generated text
        content = json_obj.get("content")
        if not content:
            self.debug(lambda: f"No content found in SOLIDO response: {json_obj}")
            return None, None

        sources = json_obj.get("sources")
        if not sources:
            self.debug(lambda: f"No sources found in SOLIDO response: {json_obj}")

        return self.make_text_response_data(content), sources
