# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.protocols import AIPerfLifecycleProtocol


@dataclass(slots=True)
class SignedRequest:
    """Result of signing a request.

    Most signers only set headers. url and body are optionally set by signers
    that modify the request URL (presigned URLs) or body (encryption).
    """

    headers: dict[str, str]
    url: str | None = None
    body: bytes | None = None


@runtime_checkable
class RequestSignerProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for request signers that add authentication signatures.

    Signers are created once per transport and called for every request.
    The sign() method is async to support signers that need I/O for
    credential/token refresh (OAuth2, GCP IAM, etc.).
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None: ...

    async def sign(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> SignedRequest: ...
