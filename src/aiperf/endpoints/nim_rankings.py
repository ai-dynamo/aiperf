# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any

from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


class NIMRankingsEndpoint(BaseRankingsEndpoint):
    """NIM Rankings endpoint.

    Processes ranking requests by taking a query and a set of passages,
    returning their relevance scores."""

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
        # A bare-list/string/int 200-OK body would crash ``json_obj.get(...)`` on
        # the worker's unconditional post-response parse; degrade to an empty
        # ranking list (BaseRankingsEndpoint.parse_response then returns None).
        # The guard lives here, not in the shared parse_response, because
        # HFTeiRankingsEndpoint legitimately expects a top-level list body.
        if not isinstance(json_obj, dict):
            return []
        return json_obj.get("rankings", [])
