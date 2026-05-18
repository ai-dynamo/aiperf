# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bearer-token guard for results-server routes that mutate cluster state."""

from __future__ import annotations

import os
import secrets
from collections.abc import Sequence

from fastapi import Depends, HTTPException, status
from fastapi.params import Depends as DependsParam
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aiperf.operator.environment import OperatorEnvironment

_bearer_scheme = HTTPBearer(auto_error=False)
_bearer_dependency = Depends(_bearer_scheme)
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _mutating_routes_enabled() -> bool:
    raw = os.environ.get("AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED")
    if raw is None:
        return OperatorEnvironment.MUTATING_ROUTES_ENABLED
    return raw.strip().lower() in _TRUE_VALUES


def _mutating_routes_token() -> str:
    return os.environ.get(
        "AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN",
        OperatorEnvironment.MUTATING_ROUTES_TOKEN,
    )


async def require_mutating_route_token(
    credentials: HTTPAuthorizationCredentials | None = _bearer_dependency,
) -> None:
    """Require the explicit results-server mutating-route bearer token."""
    if not _mutating_routes_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Results-server mutating routes are disabled by default; set AIPERF_OPERATOR_MUTATING_ROUTES_ENABLED=true and configure AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN to enable them.",
        )

    token = _mutating_routes_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Results-server mutating routes are enabled but no bearer token is configured.",
        )

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required for results-server mutating routes.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token for results-server mutating routes.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def mutating_route_dependencies() -> Sequence[DependsParam]:
    """Return dependencies applied only to results-server mutating routes."""
    return [Depends(require_mutating_route_token)]
