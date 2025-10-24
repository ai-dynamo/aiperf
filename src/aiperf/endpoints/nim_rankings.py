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
from aiperf.common.models.record_models import RankingsResponseData, RequestInfo
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.RANKINGS)
class RankingsEndpoint(BaseEndpoint):
    """NIM Rankings endpoint.

    Ranks passages against a query.

    Expected input format:
    - 'query': Text object containing the query to rank against
    - 'passages': Text object containing passages to be ranked
    """

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

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format payload for a rankings request.

        Expects texts with specific names:
        - 'query': Single text containing the query to rank against
        - 'passages': Multiple texts containing passages to be ranked

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            NIM Rankings API payload

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
            if text.name == "query":
                query_texts.extend(text.contents)
            elif text.name == "passages":
                passage_texts.extend(text.contents)
            else:
                self.warning(
                    f"Ignoring text with name '{text.name}' - rankings expects 'query' and 'passages'"
                )

        if not query_texts:
            raise ValueError(
                "Rankings request requires a text with name 'query'. "
                "Provide a Text object with name='query' containing the search query."
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

        payload = {
            "model": turn.model or model_endpoint.primary_model_name,
            "query": {"text": query_text},
            "passages": [{"text": passage} for passage in passage_texts],
        }

        if extra:
            payload.update(extra)

        self.debug(lambda: f"Formatted rankings payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse NIM Rankings response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted rankings
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        rankings = json_obj.get("rankings", [])
        if not rankings:
            self.debug(lambda: f"No rankings found in response: {json_obj}")
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns, data=RankingsResponseData(rankings=rankings)
        )
