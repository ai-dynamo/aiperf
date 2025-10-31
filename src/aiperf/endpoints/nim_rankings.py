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
@EndpointFactory.register(EndpointType.NIM_RANKINGS)
class NIMRankingsEndpoint(BaseRankingsEndpoint):
    """NIM Rankings endpoint.

    Processes ranking requests by taking a query and a set of passages,
    returning their relevance scores."""

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Rankings endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/ranking",
            supports_streaming=False,
            produces_tokens=False,
            tokenizes_input=True,
            metrics_title="Rankings Metrics",
        )

    def build_payload(
        self, query_text: str, passages: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build payload to match NIM rankings API schema."""
        payload = {
            "model": model_name,
            "query": {"text": query_text},
            "passages": [{"text": p} for p in passages],
        }
        return payload

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking results from NIM rankings API response."""
        return json_obj.get("rankings", [])
