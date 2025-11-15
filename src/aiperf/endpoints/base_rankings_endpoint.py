# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod
from typing import Any

from aiperf.common.models import ParsedResponse
from aiperf.common.models.record_models import RankingsResponseData, RequestInfo
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


class BaseRankingsEndpoint(BaseEndpoint):
    """Base class for rankings endpoint"""

    @abstractmethod
    def build_payload(
        self, query_text: str, passages: list[str], model_name: str
    ) -> dict[str, Any]:
        """Build payload based on the endpoint"""

    @abstractmethod
    def extract_rankings(self, json_obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse ranking results into a list."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for a rankings request.

        Accepts texts with specific names:
        - 'query' or 'queries': Single text containing the query to rank against
        - 'passages': Multiple texts containing passages to be ranked

        Note: Both 'query' and 'queries' are accepted for backward compatibility
        with genai-perf datasets.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Rankings API payload

        Raises:
            ValueError: If query is missing
        """
        if len(request_info.turns) != 1:
            raise ValueError("Rankings endpoint only supports one turn.")

        turn = request_info.turns[0]
        model_endpoint = request_info.model_endpoint

        if turn.max_tokens:
            self.warning("Max_tokens is provided but is not supported for rankings.")
        query_texts = []
        passage_texts = []

        for text in turn.texts:
            match text.name:
                case "passages":
                    passage_texts.extend(text.contents)
                case "query" | "queries":
                    query_texts.extend(text.contents)
                case _:
                    self.warning(
                        f"Ignoring text with name '{text.name}' - rankings expects 'query'/'queries' and 'passages'"
                    )

        if not query_texts:
            raise ValueError(
                "Rankings request requires a text with name 'query' or 'queries'. "
                "Provide a Text object with name='query' or name='queries' containing the search query."
            )

        if len(query_texts) > 1:
            self.warning(
                f"Multiple query texts found, using the first one. Found {len(query_texts)} queries."
            )

        query_text = query_texts[0]

        if not passage_texts:
            self.warning(
                "Rankings request has query but no passages to rank. "
                "Consider adding a Text object with name='passages' containing texts to rank."
            )

        extra = model_endpoint.endpoint.extra or []
        model_name = turn.model or model_endpoint.primary_model_name

        payload = self.build_payload(query_text, passage_texts, model_name)
        if extra:
            payload.update(extra)

        self.debug(lambda: f"Formatted rankings payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Rankings response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted rankings
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        # Use the subclass's implementation to extract ranking data
        # according to its specific API response format.
        rankings = self.extract_rankings(json_obj)
        if not rankings:
            self.debug(lambda: f"No rankings found in response: {json_obj}")
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns, data=RankingsResponseData(rankings=rankings)
        )
