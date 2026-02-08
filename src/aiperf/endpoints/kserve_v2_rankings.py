# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import RequestInfo
from aiperf.endpoints.base_rankings_endpoint import BaseRankingsEndpoint


class KServeV2RankingsEndpoint(BaseRankingsEndpoint):
    """KServe V2 Open Inference Protocol endpoint for reranking models.

    Wraps query and passages as BYTES tensors per the V2 inference protocol spec.
    Tensor names are configurable via --extra v2_query_name:X --extra v2_passages_name:Y
    --extra v2_output_name:Z.
    """

    DEFAULT_QUERY_NAME = "query"
    DEFAULT_PASSAGES_NAME = "passages"
    DEFAULT_OUTPUT_NAME = "scores"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        extra = (
            dict(self.model_endpoint.endpoint.extra)
            if self.model_endpoint.endpoint.extra
            else {}
        )
        self._query_name: str = extra.pop("v2_query_name", self.DEFAULT_QUERY_NAME)
        self._passages_name: str = extra.pop(
            "v2_passages_name", self.DEFAULT_PASSAGES_NAME
        )
        self._output_name: str = extra.pop("v2_output_name", self.DEFAULT_OUTPUT_NAME)
        self._extra_params: dict[str, Any] = extra

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format V2 inference request with BYTES tensors for ranking.

        Overrides base class because base does payload.update(extra) which is
        incompatible with the V2 tensor format that needs extras under "parameters".

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            V2 Open Inference Protocol payload

        Raises:
            ValueError: If request doesn't contain exactly one turn or query is missing
        """
        if len(request_info.turns) != 1:
            raise ValueError("Rankings endpoint only supports one turn.")

        turn = request_info.turns[0]

        if turn.max_tokens is not None:
            self.warning("Max_tokens is provided but is not supported for rankings.")

        query_text, passage_texts = self._extract_query_and_passages(turn)

        payload = self.build_payload(query_text, passage_texts, "")

        if self._extra_params:
            payload["parameters"] = dict(self._extra_params)

        self.trace(lambda: f"Formatted rankings payload: {payload}")
        return payload

    def build_payload(
        self, query_text: str, passages: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build V2 inference payload with query and passages as BYTES tensors.

        Args:
            query_text: The search query
            passages: List of passages to rank
            model_name: Unused for V2 (model in URL path)

        Returns:
            V2 Open Inference Protocol payload with tensor inputs
        """
        return {
            "inputs": [
                {
                    "name": self._query_name,
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [query_text],
                },
                {
                    "name": self._passages_name,
                    "shape": [len(passages)],
                    "datatype": "BYTES",
                    "data": passages,
                },
            ]
        }

    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ranking scores from V2 inference response.

        Finds the output tensor by name (fallback to first), converts flat
        float scores to [{index: i, score: s}, ...].

        Args:
            json_obj: V2 inference response JSON

        Returns:
            List of ranking dicts with index and score
        """
        outputs = json_obj.get("outputs")
        if not outputs:
            return []

        output = None
        for o in outputs:
            if o.get("name") == self._output_name:
                output = o
                break
        if output is None:
            output = outputs[0]

        data = output.get("data")
        if not data:
            return []

        return [{"index": i, "score": float(s)} for i, s in enumerate(data)]
