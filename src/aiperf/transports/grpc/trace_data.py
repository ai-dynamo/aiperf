# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace data models for gRPC transport."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from aiperf.common.models.trace_models import BaseTraceData, TraceDataExport


class GrpcTraceDataExport(TraceDataExport):
    """Export model for gRPC trace data with wall-clock timestamps."""

    trace_type: Literal["grpc"] = "grpc"

    grpc_status_code: int | None = Field(
        default=None,
        description="gRPC status code of the response.",
    )
    grpc_status_message: str | None = Field(
        default=None,
        description="gRPC status message of the response.",
    )


class GrpcTraceData(BaseTraceData):
    """Trace data for gRPC requests.

    Extends BaseTraceData with gRPC-specific status information.
    Inherits all base timing fields (reference timestamps, request/response chunks, etc.).
    """

    trace_type: str = "grpc"

    grpc_status_code: int | None = Field(
        default=None,
        description="gRPC status code of the response.",
    )
    grpc_status_message: str | None = Field(
        default=None,
        description="gRPC status message of the response.",
    )
