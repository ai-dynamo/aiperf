# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gRPC trace data models."""

from aiperf.transports.grpc.trace_data import GrpcTraceData, GrpcTraceDataExport


class TestGrpcTraceData:
    """Tests for GrpcTraceData model."""

    def test_creation_defaults(self) -> None:
        """GrpcTraceData should be created with correct defaults."""
        trace = GrpcTraceData()

        assert trace.trace_type == "grpc"
        assert trace.grpc_status_code is None
        assert trace.grpc_status_message is None
        assert trace.reference_time_ns is not None
        assert trace.reference_perf_ns is not None

    def test_creation_with_grpc_fields(self) -> None:
        """GrpcTraceData should accept gRPC-specific fields."""
        trace = GrpcTraceData(
            grpc_status_code=0,
            grpc_status_message="OK",
        )

        assert trace.grpc_status_code == 0
        assert trace.grpc_status_message == "OK"

    def test_to_export_produces_grpc_export(self) -> None:
        """to_export() should produce a GrpcTraceDataExport instance."""
        trace = GrpcTraceData(
            grpc_status_code=0,
            grpc_status_message="OK",
        )
        # Set some base timing fields
        trace.request_send_start_perf_ns = trace.reference_perf_ns
        trace.response_receive_start_perf_ns = trace.reference_perf_ns + 1000
        trace.response_receive_end_perf_ns = trace.reference_perf_ns + 2000

        export = trace.to_export()

        assert isinstance(export, GrpcTraceDataExport)
        assert export.trace_type == "grpc"
        assert export.grpc_status_code == 0
        assert export.grpc_status_message == "OK"
        # Verify wall-clock conversion happened
        assert export.request_send_start_ns == trace.reference_time_ns
        assert export.response_receive_start_ns == trace.reference_time_ns + 1000
        assert export.response_receive_end_ns == trace.reference_time_ns + 2000

    def test_inherits_base_fields(self) -> None:
        """GrpcTraceData should have all BaseTraceData fields."""
        trace = GrpcTraceData()

        # Check base fields exist
        assert hasattr(trace, "request_send_start_perf_ns")
        assert hasattr(trace, "request_chunks")
        assert hasattr(trace, "response_chunks")
        assert hasattr(trace, "response_receive_start_perf_ns")
        assert hasattr(trace, "response_receive_end_perf_ns")
        assert hasattr(trace, "error_timestamp_perf_ns")


class TestGrpcTraceDataExport:
    """Tests for GrpcTraceDataExport model."""

    def test_creation(self) -> None:
        """GrpcTraceDataExport should be created with correct trace_type."""
        export = GrpcTraceDataExport(
            trace_type="grpc",
            grpc_status_code=13,
            grpc_status_message="INTERNAL",
        )

        assert export.trace_type == "grpc"
        assert export.grpc_status_code == 13
        assert export.grpc_status_message == "INTERNAL"
