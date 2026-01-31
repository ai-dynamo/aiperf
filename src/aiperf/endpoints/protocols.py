# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.models import (
        EndpointMetadata,
        InferenceServerResponse,
        ModelEndpointInfo,
        ParsedResponse,
        RequestInfo,
        RequestRecord,
    )
    from aiperf.common.types import RequestOutputT


@runtime_checkable
class EndpointProtocol(Protocol):
    """Protocol for an endpoint."""

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None: ...

    @classmethod
    def metadata(cls) -> EndpointMetadata: ...

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT: ...
    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]: ...
    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]: ...
    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]: ...

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse response. Return None to skip."""
