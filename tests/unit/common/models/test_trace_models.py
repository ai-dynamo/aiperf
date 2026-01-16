# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for trace data models."""

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.models import (
    AioHttpTraceData,
    AioHttpTraceDataExport,
    BaseTraceData,
    TraceDataExport,
)
from tests.unit.common.models.conftest import (
    create_aiohttp_trace_data,
    create_base_trace_data,
)


class TestBaseTraceData:
    """Tests for BaseTraceData model."""

    def test_creation_with_minimal_fields(self):
        """Trace data can be created with only trace_type."""
        trace = create_base_trace_data()

        assert trace.trace_type == "base"
        assert trace.reference_time_ns is not None
        assert trace.reference_perf_ns is not None

    def test_reference_timestamps_auto_initialize(self):
        """Reference timestamps are automatically initialized if not provided."""
        trace = create_base_trace_data()

        # Both should be set
        assert trace.reference_time_ns is not None
        assert trace.reference_perf_ns is not None
        # time_ns should be much larger than perf_counter_ns
        assert trace.reference_time_ns > trace.reference_perf_ns

    def test_reference_timestamps_preserved_when_provided(self):
        """Reference timestamps are preserved when explicitly provided."""
        ref_time = 1732000000000000000
        ref_perf = 1000000000

        trace = create_base_trace_data(
            reference_time_ns=ref_time, reference_perf_ns=ref_perf
        )

        assert trace.reference_time_ns == ref_time
        assert trace.reference_perf_ns == ref_perf

    def test_request_phase_timestamps(self, base_trace_timestamps, sample_headers):
        """Request phase timestamps are properly stored."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_headers=sample_headers,
            request_headers_sent_perf_ns=ts["request_headers_sent"],
            request_send_end_perf_ns=ts["request_end"],
        )

        assert trace.request_send_start_perf_ns == ts["request_start"]
        assert trace.request_headers == sample_headers
        assert trace.request_headers_sent_perf_ns == ts["request_headers_sent"]
        assert trace.request_send_end_perf_ns == ts["request_end"]

    def test_response_phase_timestamps(self, base_trace_timestamps, sample_headers):
        """Response phase timestamps are properly stored."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            response_status_code=200,
            response_receive_start_perf_ns=ts["response_start"],
            response_headers=sample_headers,
            response_headers_received_perf_ns=ts["response_headers"],
            response_receive_end_perf_ns=ts["response_end"],
        )

        assert trace.response_status_code == 200
        assert trace.response_receive_start_perf_ns == ts["response_start"]
        assert trace.response_headers == sample_headers
        assert trace.response_headers_received_perf_ns == ts["response_headers"]
        assert trace.response_receive_end_perf_ns == ts["response_end"]

    def test_request_chunks_tracking(self, base_trace_timestamps):
        """Request chunks (timestamp, size tuples) are tracked correctly."""
        ts = base_trace_timestamps
        request_chunks = [
            (ts["request_start"] + 1_000_000, 100),
            (ts["request_start"] + 2_000_000, 200),
            (ts["request_start"] + 3_000_000, 150),
        ]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_chunks=request_chunks,
        )

        assert trace.request_chunks == request_chunks

    def test_response_chunks_tracking(self, base_trace_timestamps):
        """Response chunks (timestamp, size tuples) are tracked correctly."""
        ts = base_trace_timestamps
        response_chunks = [
            (ts["response_start"] + 1_000_000, 512),
            (ts["response_start"] + 5_000_000, 1024),
        ]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            response_chunks=response_chunks,
        )

        assert trace.response_chunks == response_chunks

    def test_error_timestamp(self, base_trace_timestamps):
        """Error timestamp is properly stored."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            error_timestamp_perf_ns=ts["error"],
        )

        assert trace.error_timestamp_perf_ns == ts["error"]

    def test_convert_perf_to_wall(self, base_trace_timestamps):
        """Perf counter timestamps are correctly converted to wall-clock time."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
        )

        # Convert a timestamp 10ms after reference
        perf_ts = ts["reference_perf"] + 10_000_000
        wall_ts = trace._convert_perf_to_wall(perf_ts)

        assert wall_ts == ts["reference_time"] + 10_000_000

    def test_convert_perf_to_wall_none(self):
        """Converting None perf timestamp returns None."""
        trace = create_base_trace_data()
        assert trace._convert_perf_to_wall(None) is None

    def test_convert_perf_to_wall_without_reference_raises_error(self):
        """Converting without reference timestamps raises ValueError."""
        trace = create_base_trace_data()

        # Manually clear reference timestamps (bypass model_post_init)
        trace.reference_time_ns = None
        trace.reference_perf_ns = None

        with pytest.raises(ValueError, match="Cannot convert without reference"):
            trace._convert_perf_to_wall(1000000000)

    def test_serialization_roundtrip(self, base_trace_timestamps, sample_headers):
        """Trace data can be serialized and deserialized."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_headers=sample_headers,
            response_status_code=200,
        )

        # Serialize to JSON
        json_data = orjson.dumps(trace.model_dump())

        # Deserialize
        loaded_data = orjson.loads(json_data)
        loaded_trace = BaseTraceData(**loaded_data)

        assert loaded_trace.trace_type == trace.trace_type
        assert loaded_trace.reference_time_ns == trace.reference_time_ns
        assert (
            loaded_trace.request_send_start_perf_ns == trace.request_send_start_perf_ns
        )
        assert loaded_trace.request_headers == sample_headers

    def test_trace_type_is_frozen(self):
        """Trace type field is frozen and cannot be modified."""
        trace = create_base_trace_data()

        with pytest.raises(
            ValidationError,
            match="Field is frozen",
        ):
            trace.trace_type = "modified"


class TestAioHttpTraceData:
    """Tests for AioHttpTraceData model."""

    def test_creation_with_minimal_fields(self):
        """AioHttpTraceData can be created with default values."""
        trace = create_aiohttp_trace_data()

        assert trace.trace_type == "aiohttp"
        assert trace.reference_time_ns is not None
        assert trace.reference_perf_ns is not None

    def test_trace_type_is_aiohttp(self):
        """Trace type is automatically set to 'aiohttp'."""
        trace = create_aiohttp_trace_data()
        assert trace.trace_type == "aiohttp"

    def test_connection_pool_timing(self, aiohttp_trace_timestamps):
        """Connection pool wait timing is properly stored."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            connection_pool_wait_start_perf_ns=ts["pool_wait_start"],
            connection_pool_wait_end_perf_ns=ts["pool_wait_end"],
        )

        assert trace.connection_pool_wait_start_perf_ns == ts["pool_wait_start"]
        assert trace.connection_pool_wait_end_perf_ns == ts["pool_wait_end"]

    def test_tcp_connection_timing(self, aiohttp_trace_timestamps):
        """TCP connection timing is properly stored."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            tcp_connect_start_perf_ns=ts["tcp_start"],
            tcp_connect_end_perf_ns=ts["tcp_end"],
        )

        assert trace.tcp_connect_start_perf_ns == ts["tcp_start"]
        assert trace.tcp_connect_end_perf_ns == ts["tcp_end"]

    def test_connection_reuse_tracking(self, aiohttp_trace_timestamps):
        """Connection reuse is properly tracked."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            connection_reused_perf_ns=ts["connection_reused"],
        )

        assert trace.connection_reused_perf_ns == ts["connection_reused"]

    def test_dns_resolution_timing(self, aiohttp_trace_timestamps):
        """DNS resolution timing is properly stored."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            dns_lookup_start_perf_ns=ts["dns_start"],
            dns_lookup_end_perf_ns=ts["dns_end"],
        )

        assert trace.dns_lookup_start_perf_ns == ts["dns_start"]
        assert trace.dns_lookup_end_perf_ns == ts["dns_end"]

    def test_dns_cache_tracking(self, aiohttp_trace_timestamps):
        """DNS cache hit/miss events are properly tracked."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            dns_cache_hit_perf_ns=ts["dns_start"],
        )

        assert trace.dns_cache_hit_perf_ns == ts["dns_start"]
        assert trace.dns_cache_miss_perf_ns is None

    def test_inherits_base_trace_fields(self, base_trace_timestamps, sample_headers):
        """AioHttpTraceData inherits all BaseTraceData fields."""
        ts = base_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_headers=sample_headers,
            response_status_code=200,
        )

        assert trace.request_send_start_perf_ns == ts["request_start"]
        assert trace.request_headers == sample_headers
        assert trace.response_status_code == 200

    def test_full_request_lifecycle(self, aiohttp_trace_timestamps, sample_headers):
        """Complete request lifecycle with all timing data."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            # Connection pool
            connection_pool_wait_start_perf_ns=ts["pool_wait_start"],
            connection_pool_wait_end_perf_ns=ts["pool_wait_end"],
            # DNS
            dns_lookup_start_perf_ns=ts["dns_start"],
            dns_lookup_end_perf_ns=ts["dns_end"],
            # TCP (includes TLS for HTTPS)
            tcp_connect_start_perf_ns=ts["tcp_start"],
            tcp_connect_end_perf_ns=ts["tcp_end"],
            # Request
            request_send_start_perf_ns=ts["request_start"],
            request_headers=sample_headers,
            request_send_end_perf_ns=ts["request_end"],
            # Response
            response_status_code=200,
            response_headers=sample_headers,
            response_receive_end_perf_ns=ts["response_end"],
        )

        # Verify all phases are captured
        assert trace.connection_pool_wait_start_perf_ns is not None
        assert trace.dns_lookup_start_perf_ns is not None
        assert trace.tcp_connect_start_perf_ns is not None
        assert trace.request_send_start_perf_ns is not None
        assert trace.response_receive_end_perf_ns is not None


class TestTraceDataExport:
    """Tests for TraceDataExport model."""

    def test_base_trace_to_export_conversion(self, base_trace_timestamps):
        """BaseTraceData converts to TraceDataExport correctly."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_send_end_perf_ns=ts["request_end"],
            response_receive_end_perf_ns=ts["response_end"],
        )

        export = trace.to_export()

        assert isinstance(export, TraceDataExport)
        assert export.trace_type == "base"
        # Timestamps should be converted from perf to wall-clock
        expected_request_start = ts["reference_time"] + (
            ts["request_start"] - ts["reference_perf"]
        )
        assert export.request_send_start_ns == expected_request_start

    def test_export_excludes_reference_timestamps(self, base_trace_timestamps):
        """Export model excludes reference timestamp fields."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
        )

        export = trace.to_export()
        export_dict = export.model_dump()

        assert "reference_time_ns" not in export_dict
        assert "reference_perf_ns" not in export_dict

    def test_export_converts_chunk_tuples(self, base_trace_timestamps):
        """Export converts chunk tuples (timestamp, size) correctly."""
        ts = base_trace_timestamps
        request_chunks = [
            (ts["request_start"] + 1_000_000, 100),
            (ts["request_start"] + 2_000_000, 200),
        ]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_chunks=request_chunks,
        )

        export = trace.to_export()

        # Verify chunk tuples are converted (timestamps converted, sizes preserved)
        assert len(export.request_chunks) == 2
        expected_first_ts = ts["reference_time"] + (
            request_chunks[0][0] - ts["reference_perf"]
        )
        assert export.request_chunks[0][0] == expected_first_ts
        assert export.request_chunks[0][1] == 100  # size preserved

    def test_computed_sending_duration(self, base_trace_timestamps):
        """Sending duration is computed correctly (k6: http_req_sending)."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_send_end_perf_ns=ts["request_end"],
        )

        export = trace.to_export()

        expected_sending = ts["request_end"] - ts["request_start"]
        assert export.sending_ns == expected_sending

    def test_computed_waiting_duration(self, base_trace_timestamps):
        """Waiting duration is computed correctly (k6: http_req_waiting)."""
        ts = base_trace_timestamps
        first_response_ts = ts["response_start"]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_end_perf_ns=ts["request_end"],
            response_chunks=[(first_response_ts, 100)],
        )

        export = trace.to_export()

        expected_waiting = first_response_ts - ts["request_end"]
        assert export.waiting_ns == expected_waiting

    def test_computed_receiving_duration_multiple_chunks(self, base_trace_timestamps):
        """Receiving duration with multiple chunks (k6: http_req_receiving)."""
        ts = base_trace_timestamps
        response_chunks = [
            (ts["response_start"], 100),
            (ts["response_start"] + 10_000_000, 200),
            (ts["response_start"] + 20_000_000, 150),
        ]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            response_chunks=response_chunks,
        )

        export = trace.to_export()

        expected_receiving = response_chunks[-1][0] - response_chunks[0][0]
        assert export.receiving_ns == expected_receiving

    def test_computed_receiving_duration_single_chunk(self, base_trace_timestamps):
        """Receiving duration with single chunk returns 0."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            response_chunks=[(ts["response_start"], 100)],
        )

        export = trace.to_export()
        assert export.receiving_ns == 0

    def test_computed_total_duration(self, base_trace_timestamps):
        """Total duration is computed correctly (k6: http_req_duration)."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            response_receive_end_perf_ns=ts["response_end"],
        )

        export = trace.to_export()

        expected_total = ts["response_end"] - ts["request_start"]
        assert export.duration_ns == expected_total

    def test_computed_fields_none_when_data_missing(self):
        """Computed fields return None when required data is missing."""
        trace = create_base_trace_data()
        export = trace.to_export()

        assert export.sending_ns is None
        assert export.waiting_ns is None
        assert export.receiving_ns is None
        assert export.duration_ns is None

    def test_export_serialization_includes_computed_fields(self, base_trace_timestamps):
        """Serialization includes computed fields."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_send_end_perf_ns=ts["request_end"],
            response_chunks=[(ts["response_start"], 100)],
            response_receive_end_perf_ns=ts["response_end"],
        )

        export = trace.to_export()
        export_dict = export.model_dump()

        assert "sending_ns" in export_dict
        assert "waiting_ns" in export_dict
        assert "receiving_ns" in export_dict
        assert "duration_ns" in export_dict


class TestAioHttpTraceDataExport:
    """Tests for AioHttpTraceDataExport model."""

    def test_aiohttp_trace_to_export_conversion(self, aiohttp_trace_timestamps):
        """AioHttpTraceData converts to AioHttpTraceDataExport correctly."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            connection_pool_wait_start_perf_ns=ts["pool_wait_start"],
            dns_lookup_start_perf_ns=ts["dns_start"],
        )

        export = trace.to_export()

        assert isinstance(export, AioHttpTraceDataExport)
        assert export.trace_type == "aiohttp"

    def test_export_uses_discriminator_routing(self, aiohttp_trace_timestamps):
        """to_export() uses discriminator to select correct export class."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
        )

        export = trace.to_export()

        # Should return AioHttpTraceDataExport, not base TraceDataExport
        assert isinstance(export, AioHttpTraceDataExport)
        assert hasattr(export, "blocked_ns")
        assert hasattr(export, "dns_lookup_ns")

    def test_computed_blocked_duration(self, aiohttp_trace_timestamps):
        """Blocked duration is computed correctly (k6: http_req_blocked)."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            connection_pool_wait_start_perf_ns=ts["pool_wait_start"],
            connection_pool_wait_end_perf_ns=ts["pool_wait_end"],
        )

        export = trace.to_export()

        expected_blocked = ts["pool_wait_end"] - ts["pool_wait_start"]
        assert export.blocked_ns == expected_blocked

    def test_computed_dns_lookup_duration(self, aiohttp_trace_timestamps):
        """DNS lookup duration is computed correctly (k6: http_req_looking_up)."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            dns_lookup_start_perf_ns=ts["dns_start"],
            dns_lookup_end_perf_ns=ts["dns_end"],
        )

        export = trace.to_export()

        expected_dns = ts["dns_end"] - ts["dns_start"]
        assert export.dns_lookup_ns == expected_dns

    def test_computed_connecting_duration(self, aiohttp_trace_timestamps):
        """Connecting duration is computed correctly (k6: http_req_connecting)."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            tcp_connect_start_perf_ns=ts["tcp_start"],
            tcp_connect_end_perf_ns=ts["tcp_end"],
        )

        export = trace.to_export()

        expected_connecting = ts["tcp_end"] - ts["tcp_start"]
        assert export.connecting_ns == expected_connecting

    def test_inherits_base_computed_fields(self, aiohttp_trace_timestamps):
        """AioHttpTraceDataExport inherits base computed fields."""
        ts = aiohttp_trace_timestamps
        trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            request_send_end_perf_ns=ts["request_end"],
        )

        export = trace.to_export()

        # Should have both base and aiohttp computed fields
        assert hasattr(export, "sending_ns")
        assert hasattr(export, "blocked_ns")
        assert export.sending_ns is not None

    def test_aiohttp_computed_fields_none_when_data_missing(self):
        """AioHttp computed fields return None when data is missing."""
        trace = create_aiohttp_trace_data()
        export = trace.to_export()

        assert export.blocked_ns is None
        assert export.dns_lookup_ns is None
        assert export.connecting_ns is None


# Edge cases and error scenarios
class TestTraceDataEdgeCases:
    """Edge case tests for trace data models."""

    def test_empty_chunk_lists(self):
        """Empty chunk lists are handled correctly."""
        trace = create_base_trace_data(
            request_chunks=[],
            response_chunks=[],
        )

        assert trace.request_chunks == []
        assert trace.response_chunks == []

        export = trace.to_export()
        assert export.request_chunks == []
        assert export.response_chunks == []
        assert export.receiving_ns is None

    def test_zero_duration_measurements(self, base_trace_timestamps):
        """Zero duration measurements are handled correctly."""
        ts = base_trace_timestamps
        same_time = ts["request_start"]

        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=same_time,
            request_send_end_perf_ns=same_time,
        )

        export = trace.to_export()
        assert export.sending_ns == 0

    def test_negative_duration_from_incorrect_timestamps(self, base_trace_timestamps):
        """Negative durations from incorrect ordering are computed as-is."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_end"],  # Swapped
            request_send_end_perf_ns=ts["request_start"],
        )

        export = trace.to_export()
        # Should compute negative value (caller's responsibility to ensure correct ordering)
        assert export.sending_ns < 0

    def test_only_partial_timing_data(self, base_trace_timestamps):
        """Partial timing data results in some computed fields being None."""
        ts = base_trace_timestamps
        trace = create_base_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            request_send_start_perf_ns=ts["request_start"],
            # Missing request_send_end_perf_ns
        )

        export = trace.to_export()
        assert export.sending_ns is None  # Can't compute without end time

    def test_status_code_edge_cases(self):
        """Various HTTP status codes are stored correctly."""
        status_codes = [200, 201, 301, 404, 500, 503]

        for status in status_codes:
            trace = create_base_trace_data(response_status_code=status)
            assert trace.response_status_code == status

    @pytest.mark.parametrize(
        "trace_type,expected_export_class",  # fmt: skip
        [
            ("base", TraceDataExport),
            ("aiohttp", AioHttpTraceDataExport),
        ],
    )
    def test_discriminator_based_export_routing(
        self, trace_type, expected_export_class, base_trace_timestamps
    ):
        """Discriminator correctly routes to appropriate export class."""
        ts = base_trace_timestamps

        if trace_type == "base":
            trace = BaseTraceData(
                trace_type=trace_type,
                reference_time_ns=ts["reference_time"],
                reference_perf_ns=ts["reference_perf"],
            )
        else:
            trace = AioHttpTraceData(
                reference_time_ns=ts["reference_time"],
                reference_perf_ns=ts["reference_perf"],
            )

        export = trace.to_export()
        assert isinstance(export, expected_export_class)

    def test_connection_reuse_vs_new_connection(self, aiohttp_trace_timestamps):
        """Connection reuse and new connection are mutually exclusive scenarios."""
        ts = aiohttp_trace_timestamps

        # Reused connection scenario
        reused_trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            connection_reused_perf_ns=ts["connection_reused"],
        )
        assert reused_trace.connection_reused_perf_ns is not None
        assert reused_trace.tcp_connect_start_perf_ns is None

        # New connection scenario
        new_trace = create_aiohttp_trace_data(
            reference_time_ns=ts["reference_time"],
            reference_perf_ns=ts["reference_perf"],
            tcp_connect_start_perf_ns=ts["tcp_start"],
            tcp_connect_end_perf_ns=ts["tcp_end"],
        )
        assert new_trace.tcp_connect_start_perf_ns is not None
        assert new_trace.connection_reused_perf_ns is None
