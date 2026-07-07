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
        # A bare-list/string/int 200-OK body would crash ``json_obj.get(...)`` on
        # the worker's unconditional post-response parse; degrade to an empty
        # ranking list (BaseRankingsEndpoint.parse_response then returns None).
        # The guard lives here, not in the shared parse_response, because
        # HFTeiRankingsEndpoint legitimately expects a top-level list body.
        if not isinstance(json_obj, dict):
            return []
        results = json_obj.get("results", [])
        if not isinstance(results, list):
            return []
        # Skip non-dict result items (``[None]``, ``['x']``, ``[5]``) so a
        # malformed 200 body degrades to an empty ranking list rather than
        # crashing ``r.get(...)`` on the worker's unconditional post-response
        # parse.
        return [
            {"index": r.get("index"), "score": r.get("relevance_score")}
            for r in results
            if isinstance(r, dict)
        ]
