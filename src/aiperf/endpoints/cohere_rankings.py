# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


class CohereRankingsEndpoint(BaseRankingsEndpoint):
    """Cohere Rankings Endpoint."""

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
