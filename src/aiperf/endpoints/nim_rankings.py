# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from typing import Any

from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


class NIMRankingsEndpoint(BaseRankingsEndpoint):
    """NIM Rankings endpoint.

    Processes ranking requests by taking a query and a set of passages,
    returning their relevance scores."""

    def build_payload(
        self,
        query_text: str,
        passages: Sequence[str],
        model_name: str,
        *,
        images: Sequence[str] = (),
        videos: Sequence[str] = (),
        audios: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Build payload to match NIM rankings API schema."""
        _ = images, videos, audios
        payload = {
            "model": model_name,
            "query": {"text": query_text},
            "passages": [{"text": p} for p in passages],
        }
        return payload

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking results from NIM rankings API response."""
        return json_obj.get("rankings", [])
