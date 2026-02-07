# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gRPC status code to HTTP status mapping."""

import pytest

from aiperf.transports.grpc.status_mapping import (
    GRPC_TO_HTTP_STATUS,
    grpc_status_to_http,
)


@pytest.mark.parametrize(
    ("grpc_code", "expected_http"),
    [
        (0, 200),  # OK
        (1, 499),  # CANCELLED
        (3, 400),  # INVALID_ARGUMENT
        (4, 504),  # DEADLINE_EXCEEDED
        (5, 404),  # NOT_FOUND
        (7, 403),  # PERMISSION_DENIED
        (8, 429),  # RESOURCE_EXHAUSTED
        (12, 501),  # UNIMPLEMENTED
        (13, 500),  # INTERNAL
        (14, 503),  # UNAVAILABLE
        (16, 401),  # UNAUTHENTICATED
    ],
)
def test_grpc_to_http_status_mapping(grpc_code: int, expected_http: int) -> None:
    """Verify all known gRPC status codes map to the correct HTTP equivalents."""
    assert grpc_status_to_http(grpc_code) == expected_http


def test_unknown_grpc_code_defaults_to_500() -> None:
    """Unknown gRPC codes should map to HTTP 500."""
    assert grpc_status_to_http(999) == 500
    assert grpc_status_to_http(-1) == 500


def test_mapping_dict_has_expected_entries() -> None:
    """Verify the mapping dict has all expected gRPC code entries."""
    assert len(GRPC_TO_HTTP_STATUS) == 11
