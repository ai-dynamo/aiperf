# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NPUSMITelemetryCollector.

The collector shells out to ``npu-smi``; tests replace the module-level
``_run_npu_smi`` coroutine with a dispatcher that returns the documented
``npu-smi info -t <type>`` key/value output for each argument tuple. No Ascend
hardware (or ``npu-smi`` binary) is required.

Output fixtures follow the CANN driver formats:
    * ``npu-smi info -m``               - fixed-width NPU/chip mapping table
    * ``npu-smi info -t common -i N``   - batched power/temp/usage key/value
    * ``npu-smi info -t memory -i N``   - HBM capacity
    * ``npu-smi info -t usages|power|temp -i N`` - per-signal fallbacks
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.gpu_telemetry import npu_smi_collector as collector_module
from aiperf.gpu_telemetry.constants import (
    ASCEND_GPU_TELEMETRY_PLATFORM,
    NPU_SMI_SOURCE_IDENTIFIER,
)
from aiperf.gpu_telemetry.npu_smi_collector import (
    MIN_COLLECTION_INTERVAL,
    NpuSmiError,
    NPUSMITelemetryCollector,
)

# ---------------------------------------------------------------------------
# npu-smi output fixtures
# ---------------------------------------------------------------------------

MAPPING_OUTPUT = """\
NPU ID   Chip ID   Chip Logic ID   Chip Name
0        0         0               Ascend 910B3
0        1         -               MCU
1        0         1               Ascend 910B3
1        1         -               MCU
"""

MEMORY_OUTPUT = {
    0: """\
        NPU ID                         : 0
        Chip ID                        : 0
        HBM Capacity(MB)               : 65536
        HBM Clock Speed(MHz)           : 1600
""",
    1: """\
        NPU ID                         : 1
        Chip ID                        : 0
        HBM Capacity(MB)               : 32768
        HBM Clock Speed(MHz)           : 1600
""",
}

COMMON_OUTPUT = {
    0: """\
        NPU ID                         : 0
        Chip Count                     : 1
        Chip ID                        : 0
        Chip Type                      : Ascend
        Chip Name                      : 910B3
        Chip Health                    : OK
        Rated Power Dissipation(W)     : 300
        NPU Real-time Power(W)         : 92.9
        Temperature(C)                 : 37
        HBM Usage Rate(%)              : 50
        Aicore Usage Rate(%)           : 47
        Aicore Freq(MHZ)               : 1800
        Aicore curFreq(MHZ)            : 800
""",
    1: """\
        NPU ID                         : 1
        Chip Count                     : 1
        Chip ID                        : 0
        Chip Type                      : Ascend
        Chip Name                      : 910B3
        Chip Health                    : Alarm
        Rated Power Dissipation(W)     : 300
        NPU Real-time Power(W)         : 218.5
        Temperature(C)                 : 61
        HBM Usage Rate(%)              : 5
        Aicore Usage Rate(%)           : 0
""",
}

USAGES_OUTPUT = {
    0: """\
        NPU ID                         : 0
        Chip ID                        : 0
        HBM Capacity(MB)               : 65536
        HBM Usage Rate(%)              : 50
        Aicore Usage Rate(%)           : 47
        Aivector Usage Rate(%)         : 30
        HBM Bandwidth Usage Rate(%)    : 12
""",
    1: """\
        NPU ID                         : 1
        Chip ID                        : 0
        HBM Capacity(MB)               : 32768
        HBM Usage Rate(%)              : 5
        Aicore Usage Rate(%)           : 0
""",
}

POWER_OUTPUT = {
    0: """\
        NPU ID                         : 0
        Chip ID                        : 0
        NPU Real-time Power(W)         : 92.9
""",
    1: """\
        NPU ID                         : 1
        Chip ID                        : 0
        NPU Real-time Power(W)         : 218.5
""",
}

TEMP_OUTPUT = {
    0: """\
        NPU ID                         : 0
        Chip ID                        : 0
        NPU Temperature(C)             : 37
""",
    1: """\
        NPU ID                         : 1
        Chip ID                        : 0
        NPU Temperature(C)             : 61
""",
}


def _standard_responses(
    *, common: bool = True, npu_ids: tuple[int, ...] = (0, 1)
) -> dict[tuple[str, ...], str]:
    """Successful npu-smi responses for ``npu_ids``."""
    responses: dict[tuple[str, ...], str] = {("info", "-m"): MAPPING_OUTPUT}
    for npu_id in npu_ids:
        responses[("info", "-t", "memory", "-i", str(npu_id))] = MEMORY_OUTPUT[npu_id]
        responses[("info", "-t", "usages", "-i", str(npu_id))] = USAGES_OUTPUT[npu_id]
        responses[("info", "-t", "power", "-i", str(npu_id))] = POWER_OUTPUT[npu_id]
        responses[("info", "-t", "temp", "-i", str(npu_id))] = TEMP_OUTPUT[npu_id]
        if common:
            responses[("info", "-t", "common", "-i", str(npu_id))] = COMMON_OUTPUT[
                npu_id
            ]
    return responses


class FakeNpuSmi:
    """Dispatches ``npu-smi`` invocations to canned output or failures."""

    def __init__(self, responses: dict[tuple[str, ...], str | BaseException]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, ...]] = []

    async def __call__(self, *args: str, timeout: float) -> str:
        self.calls.append(args)
        response = self._responses.get(args)
        if response is None:
            raise NpuSmiError(f"unexpected invocation: npu-smi {' '.join(args)}")
        if isinstance(response, BaseException):
            raise response
        return response

    def args_called(self, *args: str) -> int:
        return self.calls.count(args)


@pytest.fixture
def patch_npu_smi(monkeypatch):
    """Install a fake ``_run_npu_smi`` and return a factory for collectors."""

    def _install(
        responses: dict[tuple[str, ...], str | BaseException] | None = None,
    ) -> tuple[FakeNpuSmi, type[NPUSMITelemetryCollector]]:
        fake = FakeNpuSmi(_standard_responses() if responses is None else responses)
        monkeypatch.setattr(collector_module, "_run_npu_smi", fake)
        # validate_environment() looks the binary up on PATH; make it present.
        monkeypatch.setattr(
            collector_module.shutil, "which", lambda _name: "/usr/local/bin/npu-smi"
        )
        return fake, NPUSMITelemetryCollector

    return _install


@pytest.fixture
async def initialized_collector(patch_npu_smi):
    fake, collector_cls = patch_npu_smi()
    collector = collector_cls()
    await collector.initialize()
    yield collector, fake
    await collector.stop()


# ---------------------------------------------------------------------------
# Initialization and reachability
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_default_values(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi()
        c = collector_cls()
        assert c.id == "npu_smi_collector"
        assert c.endpoint_url == NPU_SMI_SOURCE_IDENTIFIER
        assert c._record_callback is None
        assert c._error_callback is None

    def test_custom_values(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi()
        c = collector_cls(collection_interval=5.0, collector_id="custom_id")
        assert c.id == "custom_id"
        assert c.collection_interval == 5.0

    def test_collection_interval_floored_for_subprocess_cost(self, patch_npu_smi):
        # The GPU-wide default is 333 ms, which would fork one process per NPU
        # three times a second. The collector floors it at MIN_COLLECTION_INTERVAL.
        _, collector_cls = patch_npu_smi()
        c = collector_cls(collection_interval=0.333)
        assert c.collection_interval == MIN_COLLECTION_INTERVAL

    def test_validate_environment_raises_without_npu_smi(self, monkeypatch):
        monkeypatch.setattr(collector_module.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="npu-smi not found on PATH"):
            NPUSMITelemetryCollector.validate_environment()

    def test_validate_environment_passes_with_npu_smi(self, monkeypatch):
        monkeypatch.setattr(
            collector_module.shutil, "which", lambda _name: "/usr/local/bin/npu-smi"
        )
        NPUSMITelemetryCollector.validate_environment()


class TestReachability:
    @pytest.mark.asyncio
    async def test_reachable_when_npus_present(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi()
        assert await collector_cls().is_url_reachable() is True

    @pytest.mark.asyncio
    async def test_not_reachable_when_mapping_query_fails(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi(
            {("info", "-m"): NpuSmiError("driver not loaded")}
        )
        assert await collector_cls().is_url_reachable() is False

    @pytest.mark.asyncio
    async def test_not_reachable_when_no_devices(self, patch_npu_smi):
        empty_mapping = (
            "NPU ID   Chip ID   Chip Logic ID   Chip Name\n"
            "0        1         -               MCU\n"
        )
        _, collector_cls = patch_npu_smi({("info", "-m"): empty_mapping, ("info",): ""})
        assert await collector_cls().is_url_reachable() is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_enumerates_npus(self, initialized_collector):
        collector, _ = initialized_collector
        assert collector._initialized
        assert [gpu.npu_id for gpu in collector._gpus] == [0, 1]
        metadata = collector._gpus[0].metadata
        assert metadata.gpu_index == 0
        assert metadata.gpu_uuid == "NPU-0"
        assert metadata.gpu_model_name == "Ascend 910B3"
        assert metadata.device == "npu0"
        assert metadata.platform == ASCEND_GPU_TELEMETRY_PLATFORM

    @pytest.mark.asyncio
    async def test_initialize_reads_static_hbm_capacity(self, initialized_collector):
        collector, _ = initialized_collector
        assert collector._gpus[0].hbm_capacity_mb == pytest.approx(65536.0)
        assert collector._gpus[1].hbm_capacity_mb == pytest.approx(32768.0)
        # Capacity is static: read once at init, never during a scrape.
        assert collector._gpus[0].hbm_capacity_mb is not None

    @pytest.mark.asyncio
    async def test_initialize_raises_when_no_npus(self, patch_npu_smi):
        # AIPerfLifecycleMixin wraps hook failures as CancelledError, matching
        # PyNVMLTelemetryCollector and AMDSMITelemetryCollector.
        mapping = (
            "NPU ID   Chip ID   Chip Logic ID   Chip Name\n"
            "0        1         -               MCU\n"
        )
        _, collector_cls = patch_npu_smi({("info", "-m"): mapping, ("info",): ""})
        c = collector_cls()
        with pytest.raises(asyncio.CancelledError, match="No Ascend NPUs detected"):
            await c.initialize()
        assert not c._initialized
        assert c._gpus == []

    @pytest.mark.asyncio
    async def test_initialize_failure_propagates(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi(
            {("info", "-m"): NpuSmiError("dcmi init failed")}
        )
        c = collector_cls()
        with pytest.raises(
            asyncio.CancelledError, match="Failed to enumerate Ascend NPUs"
        ):
            await c.initialize()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, initialized_collector):
        collector, _ = initialized_collector
        await collector.stop()
        await collector.stop()
        assert collector._initialized is False
        assert collector._gpus == []


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_mapping_query_skips_mcu_and_dedupes(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi()
        c = collector_cls()
        devices = await c._discover_devices()
        # Chip Logic ID '-' rows are MCU companions, and a card contributes one
        # record even though `-m` lists two chips for it.
        assert devices == [(0, "Ascend 910B3"), (1, "Ascend 910B3")]

    @pytest.mark.asyncio
    async def test_falls_back_to_summary_table(self, patch_npu_smi):
        summary = (
            "+---------------------------+---------------+-----------+\n"
            "| NPU   Name                | Health        | Power(W)  |\n"
            "| 0     910B3               | OK            | 92.9      |\n"
            "| 0                         | 0000:C1:00.0  | 0         |\n"
            "| 1     910B3               | OK            | 88.1      |\n"
            "+---------------------------+---------------+-----------+\n"
        )
        unparseable_mapping = "NPU ID Chip ID Chip Logic ID Chip Name\n"
        _, collector_cls = patch_npu_smi(
            {("info", "-m"): unparseable_mapping, ("info",): summary}
        )
        c = collector_cls()
        devices = await c._discover_devices()
        assert devices == [(0, "Ascend NPU"), (1, "Ascend NPU")]

    @pytest.mark.asyncio
    async def test_missing_hbm_capacity_is_tolerated(self, patch_npu_smi):
        responses = _standard_responses(common=False)
        # Driver reports HBM occupancy but no capacity: usage rate must still be
        # emitted while byte-level usage stays absent instead of fabricated.
        responses[("info", "-t", "memory", "-i", "0")] = "NPU ID : 0\nChip ID : 0\n"
        responses[("info", "-t", "usages", "-i", "0")] = (
            "NPU ID : 0\n"
            "Chip ID : 0\n"
            "HBM Usage Rate(%) : 50\n"
            "Aicore Usage Rate(%) : 47\n"
        )
        _, collector_cls = patch_npu_smi(responses)
        c = collector_cls()
        await c.initialize()
        try:
            assert c._gpus[0].hbm_capacity_mb is None
            records = await c._collect_records()
            assert records[0].telemetry_data.ascend_memory_utilization == 50.0
            assert records[0].telemetry_data.ascend_memory_used is None
        finally:
            await c.stop()


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class TestCollection:
    @pytest.mark.asyncio
    async def test_collect_emits_one_record_per_npu(self, initialized_collector):
        collector, _ = initialized_collector
        records = await collector._collect_records()
        assert len(records) == 2
        for record in records:
            assert record.telemetry_source_url == NPU_SMI_SOURCE_IDENTIFIER
            assert record.platform == ASCEND_GPU_TELEMETRY_PLATFORM
            assert record.timestamp_ns > 0

    @pytest.mark.asyncio
    async def test_collect_maps_common_output(self, initialized_collector):
        collector, _ = initialized_collector
        records = await collector._collect_records()
        metrics = records[0].telemetry_data

        assert metrics.ascend_power == pytest.approx(92.9)
        assert metrics.ascend_temperature == pytest.approx(37.0)
        assert metrics.ascend_aicore_utilization == pytest.approx(47.0)
        assert metrics.ascend_memory_utilization == pytest.approx(50.0)
        # 65536 MiB * 50% * 1.048576e-3 MiB->GB = 34.36 GB
        assert metrics.ascend_memory_used == pytest.approx(34.36, rel=1e-3)
        assert metrics.ascend_health == 1.0

    @pytest.mark.asyncio
    async def test_health_alarm_maps_to_zero(self, initialized_collector):
        collector, _ = initialized_collector
        records = await collector._collect_records()
        assert records[1].telemetry_data.ascend_health == 0.0

    @pytest.mark.asyncio
    async def test_rated_power_is_not_used_as_live_power(self, patch_npu_smi):
        responses = _standard_responses(common=False)
        responses[("info", "-t", "power", "-i", "0")] = (
            "NPU ID : 0\nChip ID : 0\nPower Limit(W) : 300\nPower(W) : 92.9\n"
        )
        _, collector_cls = patch_npu_smi(responses)
        c = collector_cls()
        await c.initialize()
        try:
            records = await c._collect_records()
            assert records[0].telemetry_data.ascend_power == pytest.approx(92.9)
        finally:
            await c.stop()

    @pytest.mark.asyncio
    async def test_unsupported_sensors_leave_fields_unset(self, patch_npu_smi):
        # Every `-t` subcommand answers with keys only, as a driver that does not
        # implement the sensor would. No metrics -> no record, never a record of
        # fabricated zeros.
        responses: dict[tuple[str, ...], str | BaseException] = {
            ("info", "-m"): MAPPING_OUTPUT
        }
        for npu_id in (0, 1):
            for subcmd in ("memory", "usages", "power", "temp"):
                responses[("info", "-t", subcmd, "-i", str(npu_id))] = (
                    "NPU ID : 0\nChip ID : 0\n"
                )
        _, collector_cls = patch_npu_smi(responses)
        c = collector_cls()
        await c.initialize()
        try:
            assert await c._collect_records() == []
        finally:
            await c.stop()

    @pytest.mark.asyncio
    async def test_falls_back_to_layered_queries_without_common(self, patch_npu_smi):
        responses = _standard_responses(common=False)
        _, collector_cls = patch_npu_smi(responses)
        c = collector_cls()
        await c.initialize()
        try:
            assert c._common_query_supported is False
            records = await c._collect_records()
            metrics = records[0].telemetry_data
            assert metrics.ascend_power == pytest.approx(92.9)
            assert metrics.ascend_temperature == pytest.approx(37.0)
            assert metrics.ascend_aicore_utilization == pytest.approx(47.0)
            assert metrics.ascend_memory_utilization == pytest.approx(50.0)
        finally:
            await c.stop()

    @pytest.mark.asyncio
    async def test_transient_common_failure_falls_back_without_disabling(
        self, initialized_collector
    ):
        collector, fake = initialized_collector
        assert collector._common_query_supported is True
        fake._responses[("info", "-t", "common", "-i", "0")] = NpuSmiError("busy")
        records = await collector._collect_records()
        # Card 0 came from the layered fallback, card 1 from `-t common`.
        assert records[0].telemetry_data.ascend_power == pytest.approx(92.9)
        assert records[1].telemetry_data.ascend_power == pytest.approx(218.5)
        assert collector._common_query_supported is True

    @pytest.mark.asyncio
    async def test_one_failing_card_does_not_drop_the_others(
        self, initialized_collector
    ):
        collector, fake = initialized_collector
        fake._responses[("info", "-t", "common", "-i", "1")] = RuntimeError("boom")
        records = await collector._collect_records()
        assert len(records) == 1
        assert records[0].gpu_uuid == "NPU-0"

    @pytest.mark.asyncio
    async def test_records_are_delivered_via_callback(self, patch_npu_smi):
        _, collector_cls = patch_npu_smi()
        received: list[tuple[list, str]] = []

        async def record_callback(records, collector_id):
            received.append((records, collector_id))

        c = collector_cls(record_callback=record_callback)
        await c.initialize()
        try:
            await c.collect_and_process_metrics()
        finally:
            await c.stop()
        assert len(received) == 1
        assert len(received[0][0]) == 2
        assert received[0][1] == "npu_smi_collector"

    @pytest.mark.asyncio
    async def test_collection_errors_go_to_error_callback(
        self, patch_npu_smi, monkeypatch
    ):
        _, collector_cls = patch_npu_smi()
        errors: list = []

        async def error_callback(error, collector_id):
            errors.append((error, collector_id))

        async def boom_collect(_self):
            raise RuntimeError("driver crashed")

        monkeypatch.setattr(collector_cls, "_collect_records", boom_collect)

        c = collector_cls(error_callback=error_callback)
        await c.initialize()
        try:
            await c._collect_and_process_metrics()
        finally:
            await c.stop()

        assert len(errors) == 1
        assert errors[0][1] == "npu_smi_collector"
        assert "driver crashed" in errors[0][0].message


# ---------------------------------------------------------------------------
# Config registration (end-to-end wiring guard: ascend_* fields must appear in
# GPU_TELEMETRY_METRICS_CONFIG so the accumulator, console exporter, dashboard,
# and CSV exports surface them).
# ---------------------------------------------------------------------------


class TestConfigRegistration:
    def test_ascend_fields_registered_in_metrics_config(self) -> None:
        from aiperf.common.models.telemetry_models import TelemetryMetrics
        from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG

        registered = {field for _, field, _ in GPU_TELEMETRY_METRICS_CONFIG}
        ascend_fields = {
            f for f in TelemetryMetrics.model_fields if f.startswith("ascend_")
        }
        missing = ascend_fields - registered
        assert not missing, (
            f"ascend_* fields on TelemetryMetrics not registered in "
            f"GPU_TELEMETRY_METRICS_CONFIG (downstream accumulator will silently "
            f"drop them): {sorted(missing)}"
        )

    def test_ascend_power_is_not_a_cumulative_counter(self) -> None:
        from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_COUNTER_METRICS

        # npu-smi reports instantaneous power only; treating it as a counter
        # would turn every avg into a delta.
        assert "ascend_power" not in GPU_TELEMETRY_COUNTER_METRICS

    def test_ascend_metrics_flow_through_telemetry_hierarchy(self) -> None:
        from aiperf.common.exceptions import NoMetricValue
        from aiperf.common.models.telemetry_models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )
        from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG

        hierarchy = TelemetryHierarchy()
        for ts, power in ((1_000_000_000, 92.9), (2_000_000_000, 210.4)):
            hierarchy.add_record(
                TelemetryRecord(
                    gpu_index=0,
                    gpu_uuid="NPU-0",
                    gpu_model_name="Ascend 910B3",
                    timestamp_ns=ts,
                    telemetry_source_url=NPU_SMI_SOURCE_IDENTIFIER,
                    telemetry_data=TelemetryMetrics(
                        ascend_power=power,
                        ascend_temperature=37.0,
                        ascend_aicore_utilization=47.0,
                        ascend_memory_utilization=50.0,
                        ascend_memory_used=34.36,
                        ascend_health=1.0,
                    ),
                )
            )

        gpu_data = hierarchy.telemetry_source_endpoints[NPU_SMI_SOURCE_IDENTIFIER][
            "NPU-0"
        ]
        seen = set()
        for _display, field, unit_enum in GPU_TELEMETRY_METRICS_CONFIG:
            if not field.startswith("ascend_"):
                continue
            try:
                result = gpu_data.get_metric_result(
                    field,
                    f"tag_{field}",
                    f"header_{field}",
                    unit_enum.value,
                )
            except NoMetricValue:
                continue
            assert result.tag == f"tag_{field}"
            assert result.unit == unit_enum.value
            seen.add(field)

        assert seen == {
            "ascend_power",
            "ascend_temperature",
            "ascend_aicore_utilization",
            "ascend_memory_utilization",
            "ascend_memory_used",
            "ascend_health",
        }

    def test_platform_maps_expose_ascend_power(self) -> None:
        from aiperf.gpu_telemetry.accumulator import _PLATFORM_POWER_FIELDS

        assert _PLATFORM_POWER_FIELDS[ASCEND_GPU_TELEMETRY_PLATFORM] == "ascend_power"
