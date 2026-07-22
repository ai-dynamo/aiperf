# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auth-header construction for readiness probes and control-plane hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.endpoint import EndpointConfig

# Anthropic Messages authenticates with x-api-key plus a required
# anthropic-version header. Bearer auth or a missing version returns a 4xx,
# which the readiness probe's "status < 500 == ready" rule would misread as
# ready. Mirrors MessagesEndpoint.get_endpoint_headers().
_ANTHROPIC_VERSION = "2023-06-01"


def auth_headers_for_endpoint(cfg: EndpointConfig) -> dict[str, str]:
    """Build auth headers matching the endpoint's scheme (Bearer or Anthropic)."""
    headers = dict(cfg.headers or {})
    if str(cfg.type) == "messages":
        headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        if cfg.api_key:
            # Hard-assign so --api-key overrides any preconfigured x-api-key,
            # matching MessagesEndpoint.get_endpoint_headers(); otherwise
            # preflight would probe a different key than real requests use.
            headers["x-api-key"] = cfg.api_key
    elif cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers
