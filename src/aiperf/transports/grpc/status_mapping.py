# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mapping of gRPC status codes to HTTP status equivalents."""

from __future__ import annotations

import grpc

GRPC_TO_HTTP_STATUS: dict[int, int] = {
    grpc.StatusCode.OK.value[0]: 200,
    grpc.StatusCode.CANCELLED.value[0]: 499,
    grpc.StatusCode.INVALID_ARGUMENT.value[0]: 400,
    grpc.StatusCode.DEADLINE_EXCEEDED.value[0]: 504,
    grpc.StatusCode.NOT_FOUND.value[0]: 404,
    grpc.StatusCode.PERMISSION_DENIED.value[0]: 403,
    grpc.StatusCode.RESOURCE_EXHAUSTED.value[0]: 429,
    grpc.StatusCode.UNIMPLEMENTED.value[0]: 501,
    grpc.StatusCode.INTERNAL.value[0]: 500,
    grpc.StatusCode.UNAVAILABLE.value[0]: 503,
    grpc.StatusCode.UNAUTHENTICATED.value[0]: 401,
}


def grpc_status_to_http(code: int) -> int:
    """Convert a gRPC status code to its HTTP equivalent.

    Args:
        code: gRPC status code integer value.

    Returns:
        HTTP status code. Defaults to 500 for unknown gRPC codes.
    """
    return GRPC_TO_HTTP_STATUS.get(code, 500)
