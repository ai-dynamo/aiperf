# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

from aiperf.common.decorators import implements_protocol
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import (
    ParsedResponse,
    RequestInfo,
    RequestRecord,
    TextResponseData,
)
from aiperf.common.protocols import (
    EndpointProtocol,
    InferenceServerResponse,
    RequestOutputT,
)


@implements_protocol(EndpointProtocol)
class BaseEndpoint(AIPerfLoggerMixin, ABC):
    """Base for all endpoints.

    Endpoints handle API-specific formatting and parsing.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint

    @classmethod
    @abstractmethod
    def metadata(cls) -> EndpointMetadata:
        """Return endpoint metadata."""

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint headers (auth + user custom). Override to customize."""
        headers = {}

        cfg = self.model_endpoint.endpoint
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        if cfg.headers:
            headers.update(dict(cfg.headers))

        return headers

    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]:
        """Get endpoint URL query params (e.g., api-version). Override to customize."""
        params = {}

        cfg = self.model_endpoint.endpoint
        if cfg.url_params:
            params.update(cfg.url_params)

        return params

    @abstractmethod
    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        """Format request payload from RequestInfo.

        Uses request_info.turns[0] as the turn data (currently hardcoded to first turn).
        """

    @abstractmethod
    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        """Extract parsed data from record.

        Args:
            record: Request record containing responses to parse

        Returns:
            List of successfully parsed responses
        """
        results: list[ParsedResponse] = []
        for response in record.responses:
            if parsed := self.parse_response(response):
                results.append(parsed)
        return results

    @staticmethod
    def make_text_response_data(text: str | None) -> TextResponseData | None:
        """Make a TextResponseData object from a string or return None if the text is empty."""
        return TextResponseData(text=text) if text else None
