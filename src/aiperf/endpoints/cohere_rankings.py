# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.protocols import EndpointProtocol
from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.COHERE_RANKINGS)
class CohereRankingsEndpoint(BaseRankingsEndpoint):
    """Cohere Rankings Endpoint."""

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        return EndpointMetadata(
            endpoint_path="/v2/rerank",
            supports_streaming=False,
            produces_tokens=False,
            tokenizes_input=True,
            metrics_title="Ranking Metrics",
        )

    def build_payload(
        self, query_text: str, passages: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build payload to match Cohere Rankings API schema."""
        payload = {
            "model": model_name,
            "query": query_text,
            "documents": passages,
        }
        return payload

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking results from Cohere Rankings API response."""
        results = json_obj.get("results", [])
        rankings = [
            {"index": r.get("index"), "score": r.get("relevance_score")}
            for r in results
        ]
        return rankings
