# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for export model serialization edge cases."""

from datetime import UTC, datetime

import orjson

from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    TelemetryExportData,
    TelemetrySummary,
)


class TestGpuSummaryHostnameDefault:
    """GpuSummary.hostname must default so exclude_none round-trips survive."""

    def test_hostname_is_optional_like_siblings(self) -> None:
        hostname = GpuSummary.model_fields["hostname"]
        assert hostname.is_required() is False
        for sibling in ("namespace", "pod_name"):
            assert GpuSummary.model_fields[sibling].is_required() is False

    def test_exclude_none_round_trip_with_missing_hostname(self) -> None:
        """Reproduce #1417: hostname=None must validate after exclude_none dump.

        dcgm_collector sets hostname from an optional Prometheus label. Message
        serialization uses model_dump(exclude_none=True), so a None hostname is
        omitted on the wire. Without a default, receive-side model_validate fails
        with 'Field required' and profile finalization hangs.
        """
        gpus = {
            "gpu_0": GpuSummary(
                gpu_index=0,
                gpu_name="GPU",
                gpu_uuid="GPU-0",
                hostname=None,
                namespace=None,
                pod_name=None,
                metrics={},
            )
        }
        now = datetime.now(UTC)
        data = TelemetryExportData(
            summary=TelemetrySummary(start_time=now, end_time=now),
            endpoints={"localhost:9400": EndpointData(gpus=gpus)},
        )

        wire = data.model_dump(
            exclude_none=True,
            mode="json",
            context={"include_internal": True},
        )
        gpu0 = wire["endpoints"]["localhost:9400"]["gpus"]["gpu_0"]
        assert "hostname" not in gpu0

        restored = TelemetryExportData.model_validate(orjson.loads(orjson.dumps(wire)))
        assert restored.endpoints["localhost:9400"].gpus["gpu_0"].hostname is None
