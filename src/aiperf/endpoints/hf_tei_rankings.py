# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


class HFTeiRankingsEndpoint(BaseRankingsEndpoint):
    """HuggingFace TEI Rankings Endpoint."""

    def build_payload(
        self, query_text: str, passages: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build payload to match Huggingface TEI Rankings API schema."""
        payload = {
            "query": query_text,
            "texts": passages,
        }
        return payload

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking results from Huggingface TEI Rankings API response."""
        return json_obj if isinstance(json_obj, list) else json_obj.get("results", [])
