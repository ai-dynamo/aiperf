# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AMDSMITelemetryCollector.

Tests use a mocked amdsmi module to verify collector behavior without requiring
actual AMD ROCm GPU hardware. Empirically validated against MI300X (gfx942)
and MI355X (gfx950) — see ``_AMDGpuDeviceState`` notes for AMDSMI quirks.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aiperf.gpu_telemetry.constants import AMDSMI_SOURCE_IDENTIFIER

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_amdsmi(num_gpus: int = 2) -> MagicMock:
    """Build a mock ``amdsmi`` module that mimics the real API surface.

    Models the empirically observed quirks of AMDSMI on MI300X/MI355X:
        - ``current_socket_power`` is in W (no scaling required)
        - ``average_socket_power`` returns the literal string ``'N/A'``
        - ``mm_activity`` returns ``'N/A'`` on Instinct GPUs
        - ``EDGE`` temperature raises ``AmdSmiException``; ``JUNCTION`` works
        - ``energy_accumulator * counter_resolution`` is in µJ
    """
    m = MagicMock()
    # Default to a modern (>= 26.x) binding so temperatures are returned in °C.
    # Tests that want the legacy millidegree path override this explicitly.
    m.__version__ = "26.0.2+39589fda"
    m.AmdSmiException = type("AmdSmiException", (Exception,), {})
    m.AmdSmiLibraryException = m.AmdSmiException
    m.AmdSmiMemoryType = SimpleNamespace(VRAM=0)
    m.AmdSmiTemperatureType = SimpleNamespace(EDGE=0, JUNCTION=1, HOTSPOT=2, VRAM=3)
    m.AmdSmiTemperatureMetric = SimpleNamespace(CURRENT=0)

    handles = [object() for _ in range(num_gpus)]
    m.amdsmi_init.return_value = None
    m.amdsmi_shut_down.return_value = None
    m.amdsmi_get_processor_handles.return_value = handles

    def by_idx(values: list):
        idx_map = {h: v for h, v in zip(handles, values, strict=True)}
        return lambda h, *_: idx_map[h]

    m.amdsmi_get_gpu_device_uuid.side_effect = by_idx(
        [f"06ff74a1-0000-1000-806c-{i:012x}" for i in range(num_gpus)]
    )
    m.amdsmi_get_gpu_device_bdf.side_effect = by_idx(
        [f"0000:{i:02x}:00.0" for i in range(num_gpus)]
    )
    m.amdsmi_get_gpu_board_info.side_effect = by_idx(
        [{"product_name": "AMD Instinct MI300X OAM"} for _ in range(num_gpus)]
    )

    # Power: 287 W for GPU 0, 218 W for GPU 1 — average_socket_power is N/A.
    m.amdsmi_get_power_info.side_effect = by_idx(
        [
            {"current_socket_power": 287, "average_socket_power": "N/A"},
            {"current_socket_power": 218, "average_socket_power": "N/A"},
        ][:num_gpus]
    )

    # Energy: accumulator(ticks) * counter_resolution(15.3 µJ) = ~640 J then -> MJ
    m.amdsmi_get_energy_count.side_effect = by_idx(
        [
            {"energy_accumulator": 41_797_534_008_632, "counter_resolution": 15.3},
            {"energy_accumulator": 867_336_253_691, "counter_resolution": 15.3},
        ][:num_gpus]
    )

    # Activity: gfx 47%, umc 0% (loaded gpu); gfx 0%, umc 0% (idle gpu).
    # mm_activity intentionally 'N/A' to exercise the dropout path.
    m.amdsmi_get_gpu_activity.side_effect = by_idx(
        [
            {"gfx_activity": 47, "umc_activity": 0, "mm_activity": "N/A"},
            {"gfx_activity": 0, "umc_activity": 0, "mm_activity": "N/A"},
        ][:num_gpus]
    )

    # VRAM: 183 GB used, 0.3 GB used (in bytes).
    m.amdsmi_get_gpu_memory_usage.side_effect = by_idx(
        [183_678_435_328, 297_766_912][:num_gpus]
    )

    # Temperature: EDGE raises (unsupported on Instinct), JUNCTION returns int.
    def temp_metric(handle, kind, _metric):
        if kind == m.AmdSmiTemperatureType.EDGE:
            raise m.AmdSmiException("EDGE not supported")
        if kind == m.AmdSmiTemperatureType.JUNCTION:
            return 67 if handle == handles[0] else 41
        if kind == m.AmdSmiTemperatureType.HOTSPOT:
            return 67 if handle == handles[0] else 41
        return 49

    m.amdsmi_get_temp_metric.side_effect = temp_metric

    m.amdsmi_get_gpu_total_ecc_count.side_effect = by_idx(
        [
            {"correctable_count": 0, "uncorrectable_count": 0, "deferred_count": 0},
            {"correctable_count": 1, "uncorrectable_count": 2, "deferred_count": 0},
        ][:num_gpus]
    )

    # Throttle: GPU 0 throttling, GPU 1 not throttling.
    m.amdsmi_get_gpu_metrics_info.side_effect = by_idx(
        [
            {"throttle_status": 1, "indep_throttle_status": 0},
            {"throttle_status": 0, "indep_throttle_status": 0},
        ][:num_gpus]
    )

    return m


@pytest.fixture
def mock_amdsmi():
    return _make_mock_amdsmi(num_gpus=2)


@pytest.fixture
def patch_amdsmi(mock_amdsmi):
    from aiperf.gpu_telemetry import amdsmi_collector
    from aiperf.gpu_telemetry.amdsmi_collector import AMDSMITelemetryCollector

    with patch.object(amdsmi_collector, "amdsmi", mock_amdsmi):
        yield mock_amdsmi, AMDSMITelemetryCollector


@pytest.fixture
async def initialized_collector(patch_amdsmi):
    _, AMDSMITelemetryCollector = patch_amdsmi
    collector = AMDSMITelemetryCollector()
    await collector.initialize()
    yield collector
    await collector.stop()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_default_values(self, patch_amdsmi):
        _, AMDSMITelemetryCollector = patch_amdsmi
        c = AMDSMITelemetryCollector()
        assert c.id == "amdsmi_collector"
        assert c.endpoint_url == AMDSMI_SOURCE_IDENTIFIER
        assert c._record_callback is None
        assert c._error_callback is None

    def test_custom_values(self, patch_amdsmi):
        _, AMDSMITelemetryCollector = patch_amdsmi
        c = AMDSMITelemetryCollector(collection_interval=0.5, collector_id="custom_id")
        assert c.id == "custom_id"
        assert c.collection_interval == 0.5


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestReachability:
    @pytest.mark.asyncio
    async def test_reachable_when_gpus_present(self, patch_amdsmi):
        _, AMDSMITelemetryCollector = patch_amdsmi
        c = AMDSMITelemetryCollector()
        assert await c.is_url_reachable() is True

    @pytest.mark.asyncio
    async def test_not_reachable_when_no_gpus(self, patch_amdsmi):
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.amdsmi_get_processor_handles.return_value = []
        c = AMDSMITelemetryCollector()
        assert await c.is_url_reachable() is False

    @pytest.mark.asyncio
    async def test_not_reachable_when_init_fails(self, patch_amdsmi):
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.amdsmi_init.side_effect = mock_amdsmi.AmdSmiException("driver gone")
        c = AMDSMITelemetryCollector()
        assert await c.is_url_reachable() is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_initialize_enumerates_gpus(self, initialized_collector):
        assert initialized_collector._initialized
        assert len(initialized_collector._gpus) == 2
        assert initialized_collector._gpus[0].metadata.gpu_index == 0
        assert (
            initialized_collector._gpus[0].metadata.gpu_model_name
            == "AMD Instinct MI300X OAM"
        )
        assert initialized_collector._gpus[0].metadata.device == "amd0"
        assert initialized_collector._gpus[0].metadata.pci_bus_id == "0000:00:00.0"

    @pytest.mark.asyncio
    async def test_init_failure_propagates_via_lifecycle(self, patch_amdsmi):
        # AIPerfLifecycleMixin re-raises hook failures as CancelledError with
        # the original message preserved (matches PyNVMLTelemetryCollector).
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.amdsmi_init.side_effect = mock_amdsmi.AmdSmiException("nope")
        c = AMDSMITelemetryCollector()
        with pytest.raises(asyncio.CancelledError, match="Failed to initialize amdsmi"):
            await c.initialize()
        assert not c._initialized

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, initialized_collector, mock_amdsmi):
        await initialized_collector.stop()
        await initialized_collector.stop()  # second call is no-op
        assert mock_amdsmi.amdsmi_shut_down.call_count >= 1
        assert initialized_collector._initialized is False
        assert initialized_collector._gpus == []

    @pytest.mark.asyncio
    async def test_init_raises_when_no_gpus_enumerated(self, patch_amdsmi):
        # _initialize_amdsmi must not silently leave the collector running with
        # zero devices — that would emit no records and confuse the dashboard.
        # AIPerfLifecycleMixin wraps the RuntimeError as CancelledError.
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.amdsmi_get_processor_handles.return_value = []
        c = AMDSMITelemetryCollector()
        with pytest.raises(asyncio.CancelledError, match="No AMD GPUs detected"):
            await c.initialize()
        assert not c._initialized
        assert c._gpus == []


# ---------------------------------------------------------------------------
# Config registration (regression guard for dynamo-ops finding on PR #908:
# amd_* fields must appear in GPU_TELEMETRY_METRICS_CONFIG so the accumulator,
# console exporter, dashboard, and CSV exports can surface them end-to-end.)
# ---------------------------------------------------------------------------


class TestConfigRegistration:
    def test_amd_fields_registered_in_metrics_config(self) -> None:
        from aiperf.common.models.telemetry_models import TelemetryMetrics
        from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG

        registered = {field for _, field, _ in GPU_TELEMETRY_METRICS_CONFIG}
        amd_fields = {f for f in TelemetryMetrics.model_fields if f.startswith("amd_")}
        missing = amd_fields - registered
        assert not missing, (
            f"amd_* fields on TelemetryMetrics not registered in "
            f"GPU_TELEMETRY_METRICS_CONFIG (downstream accumulator will silently "
            f"drop them): {sorted(missing)}"
        )

    def test_cumulative_amd_fields_marked_as_counters(self) -> None:
        from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_COUNTER_METRICS

        # These two AMD signals are cumulative across the device's lifetime;
        # accumulator must compute deltas, not distribution stats.
        assert "amd_energy_consumption" in GPU_TELEMETRY_COUNTER_METRICS
        assert "amd_ecc_uncorrectable" in GPU_TELEMETRY_COUNTER_METRICS

    def test_amd_metrics_flow_through_telemetry_hierarchy(self) -> None:
        # End-to-end: feed two records with amd_* values into the same
        # hierarchy the accumulator uses, then iterate the config the same
        # way summarize() does and confirm every registered amd_* metric
        # produces a MetricResult instead of a NoMetricValue.
        from aiperf.common.exceptions import NoMetricValue
        from aiperf.common.models.telemetry_models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )
        from aiperf.gpu_telemetry.constants import (
            GPU_TELEMETRY_COUNTER_METRICS,
            GPU_TELEMETRY_METRICS_CONFIG,
        )

        hierarchy = TelemetryHierarchy()
        for ts, energy in ((1_000_000_000, 100.0), (2_000_000_000, 105.0)):
            hierarchy.add_record(
                TelemetryRecord(
                    gpu_index=0,
                    gpu_uuid="GPU-amd-test-0",
                    gpu_model_name="AMD Instinct MI300X OAM",
                    timestamp_ns=ts,
                    dcgm_url="amdsmi://localhost",
                    telemetry_data=TelemetryMetrics(
                        amd_power=287.0,
                        amd_energy_consumption=energy,
                        amd_gfx_activity=47.0,
                        amd_umc_activity=12.0,
                        amd_memory_used=183.6,
                        amd_temperature=54.0,
                        amd_ecc_uncorrectable=0.0,
                        amd_throttle_status=0.0,
                        # amd_mm_activity left None on purpose (Instinct GPUs)
                    ),
                )
            )

        gpu_data = hierarchy.dcgm_endpoints["amdsmi://localhost"]["GPU-amd-test-0"]
        seen = set()
        for _display, field, unit_enum in GPU_TELEMETRY_METRICS_CONFIG:
            if not field.startswith("amd_"):
                continue
            try:
                result = gpu_data.get_metric_result(
                    field,
                    f"tag_{field}",
                    f"header_{field}",
                    unit_enum.value,
                    is_counter=field in GPU_TELEMETRY_COUNTER_METRICS,
                )
            except NoMetricValue:
                continue  # amd_mm_activity is intentionally absent
            assert result.tag == f"tag_{field}"
            assert result.unit == unit_enum.value
            seen.add(field)

        # Every amd_* field we populated should have flowed through.
        assert seen >= {
            "amd_power",
            "amd_energy_consumption",
            "amd_gfx_activity",
            "amd_umc_activity",
            "amd_memory_used",
            "amd_temperature",
            "amd_ecc_uncorrectable",
            "amd_throttle_status",
        }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class TestCollection:
    @pytest.mark.asyncio
    async def test_collect_emits_record_per_gpu(self, initialized_collector):
        records = await initialized_collector._loop_to_thread_collect()
        assert len(records) == 2
        for r in records:
            assert r.dcgm_url == AMDSMI_SOURCE_IDENTIFIER
            assert r.timestamp_ns > 0

    @pytest.mark.asyncio
    async def test_collect_has_all_expected_fields(self, initialized_collector):
        records = await initialized_collector._loop_to_thread_collect()
        td0 = records[0].telemetry_data

        # Power: passed through unscaled (W), under amd_* namespace.
        assert td0.amd_power == 287.0

        # Energy: 41_797_534_008_632 ticks * 15.3 µJ/tick / 1e12 ≈ 639.5 MJ
        assert td0.amd_energy_consumption == pytest.approx(639.5, rel=1e-3)

        # Activity: gfx/umc emitted under amd_* names; mm_activity is N/A on
        # Instinct so amd_mm_activity stays unset (not duplicated to
        # encoder/decoder fields).
        assert td0.amd_gfx_activity == 47.0
        assert td0.amd_umc_activity == 0.0
        assert td0.amd_mm_activity is None

        # NVML-named fields must NOT be set by the AMD collector.
        assert td0.gpu_utilization is None
        assert td0.sm_utilization is None
        assert td0.mem_utilization is None
        assert td0.encoder_utilization is None
        assert td0.decoder_utilization is None
        assert td0.jpg_utilization is None

        # VRAM: 183_678_435_328 bytes -> ~183.68 GB
        assert td0.amd_memory_used == pytest.approx(183.68, rel=1e-3)

        # Temperature: EDGE failed, JUNCTION returned 67.
        assert td0.amd_temperature == 67.0

    @pytest.mark.asyncio
    async def test_collect_handles_partial_failure(
        self, initialized_collector, mock_amdsmi
    ):
        # Make temperature unsupported entirely; rest must still populate.
        mock_amdsmi.amdsmi_get_temp_metric.side_effect = mock_amdsmi.AmdSmiException(
            "no temp"
        )
        records = await initialized_collector._loop_to_thread_collect()
        td = records[0].telemetry_data
        assert td.amd_temperature is None
        assert td.amd_power == 287.0  # unaffected

    @pytest.mark.asyncio
    async def test_na_strings_become_none_not_strings(
        self, initialized_collector, mock_amdsmi
    ):
        # Force every field to return 'N/A' to confirm no string leaks into model.
        mock_amdsmi.amdsmi_get_power_info.side_effect = lambda h, *_: {
            "current_socket_power": "N/A",
            "average_socket_power": "N/A",
        }
        records = await initialized_collector._loop_to_thread_collect()
        for r in records:
            assert r.telemetry_data.amd_power is None

    @pytest.mark.asyncio
    async def test_throttle_status_is_snapshot_not_accumulation(
        self, initialized_collector
    ):
        # AMDSMI exposes throttle_status as a state (bool/bitfield), not a
        # duration counter. Surface the raw snapshot per scrape rather than
        # synthesizing a duration client-side.
        records1 = await initialized_collector._loop_to_thread_collect()
        records2 = await initialized_collector._loop_to_thread_collect()

        # GPU 0 throttling (mock returns throttle_status=1) -> 1.0 every scrape.
        assert records1[0].telemetry_data.amd_throttle_status == 1.0
        assert records2[0].telemetry_data.amd_throttle_status == 1.0
        # GPU 1 not throttling -> 0.0.
        assert records1[1].telemetry_data.amd_throttle_status == 0.0
        assert records2[1].telemetry_data.amd_throttle_status == 0.0
        # The synthesized power_violation field is no longer populated.
        assert records2[0].telemetry_data.power_violation is None

    @pytest.mark.asyncio
    async def test_temperature_normalized_when_returned_in_millidegrees(
        self, patch_amdsmi
    ):
        # Legacy AMDSMI bindings (< 26.x) return temperature in millidegrees C.
        # The version gate routes those through /1000.
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.__version__ = "25.5.0"

        def temp_mdeg(handle, kind, _metric):
            if kind == mock_amdsmi.AmdSmiTemperatureType.EDGE:
                raise mock_amdsmi.AmdSmiException("EDGE not supported")
            return 67000  # 67°C reported as millidegrees

        mock_amdsmi.amdsmi_get_temp_metric.side_effect = temp_mdeg
        c = AMDSMITelemetryCollector()
        await c.initialize()
        try:
            records = await c._loop_to_thread_collect()
        finally:
            await c.stop()
        assert records[0].telemetry_data.amd_temperature == 67.0

    @pytest.mark.asyncio
    async def test_temperature_passthrough_on_modern_binding(
        self, initialized_collector
    ):
        # Default mock binding is 26.0.2 (modern), so JUNCTION's raw int 67
        # passes through unchanged — no /1000 applied.
        records = await initialized_collector._loop_to_thread_collect()
        assert records[0].telemetry_data.amd_temperature == 67.0

    @pytest.mark.asyncio
    async def test_temperature_modern_binding_with_millideg_value_normalized(
        self, initialized_collector, mock_amdsmi
    ):
        # AMD's amdsmi 26.2.2 docs document the API as returning milliCelsius,
        # even though every 26.x binding we have empirical access to returns
        # Celsius. The version gate alone would mis-handle a system that
        # matches the docs; the >200 sanity check on top guarantees we still
        # report a sane temperature in either case. Default mock binding is
        # 26.0.2 (modern) so the gate would otherwise skip the divide.
        def temp_mdeg(handle, kind, _metric):
            if kind == mock_amdsmi.AmdSmiTemperatureType.EDGE:
                raise mock_amdsmi.AmdSmiException("EDGE not supported")
            return 67000  # 67°C reported as millidegrees

        mock_amdsmi.amdsmi_get_temp_metric.side_effect = temp_mdeg
        records = await initialized_collector._loop_to_thread_collect()
        assert records[0].telemetry_data.amd_temperature == 67.0

    @pytest.mark.asyncio
    async def test_temperature_unparsable_version_assumes_modern(self, patch_amdsmi):
        # If amdsmi.__version__ is missing or unparsable, default to "modern"
        # (no /1000) since every currently-deployed binding is >= 26.x.
        mock_amdsmi, AMDSMITelemetryCollector = patch_amdsmi
        mock_amdsmi.__version__ = "garbage-not-a-version"

        def temp_pass(handle, kind, _metric):
            if kind == mock_amdsmi.AmdSmiTemperatureType.EDGE:
                raise mock_amdsmi.AmdSmiException("EDGE not supported")
            return 54  # already-Celsius

        mock_amdsmi.amdsmi_get_temp_metric.side_effect = temp_pass
        c = AMDSMITelemetryCollector()
        await c.initialize()
        try:
            records = await c._loop_to_thread_collect()
        finally:
            await c.stop()
        assert records[0].telemetry_data.amd_temperature == 54.0

    @pytest.mark.asyncio
    async def test_energy_falls_back_to_power_field_for_rocm6(
        self, initialized_collector, mock_amdsmi
    ):
        # ROCm 6.x AMDSMI exposes the energy accumulator as `power` before
        # being renamed to `energy_accumulator` in 6.2.
        mock_amdsmi.amdsmi_get_energy_count.side_effect = lambda h, *_: {
            "power": 1_000_000_000_000,  # 1e12 ticks
            "counter_resolution": 15.3,
            "timestamp": 0,
        }
        records = await initialized_collector._loop_to_thread_collect()
        # 1e12 * 15.3 * 1e-12 = 15.3 MJ
        assert records[0].telemetry_data.amd_energy_consumption == pytest.approx(15.3)

    @pytest.mark.asyncio
    async def test_throttle_unset_when_sensors_unsupported(
        self, initialized_collector, mock_amdsmi
    ):
        # If amdsmi returns 'N/A' for both throttle signals, leave the field
        # unset rather than emitting 0.0 — we cannot distinguish "supported and
        # not throttled" from "sensor unsupported" in that case.
        mock_amdsmi.amdsmi_get_gpu_metrics_info.side_effect = lambda h, *_: {
            "throttle_status": "N/A",
            "indep_throttle_status": "N/A",
        }
        records = await initialized_collector._loop_to_thread_collect()
        for r in records:
            assert r.telemetry_data.amd_throttle_status is None

    @pytest.mark.asyncio
    async def test_throttle_handles_bool_int_and_na(
        self, initialized_collector, mock_amdsmi
    ):
        # AMDSMI returns throttle_status as bool on some platforms, int on
        # others, and 'N/A' string on unsupported sensors. All three must
        # be classified correctly by _is_throttled.
        from aiperf.gpu_telemetry.amdsmi_collector import _is_throttled

        assert _is_throttled(True) is True
        assert _is_throttled(False) is False
        assert _is_throttled(1) is True
        assert _is_throttled(0) is False
        assert _is_throttled(0x10) is True  # bitfield
        assert _is_throttled("N/A") is False
        assert _is_throttled(None) is False

    @pytest.mark.asyncio
    async def test_ecc_uncorrectable_emitted_under_amd_namespace(
        self, initialized_collector
    ):
        records = await initialized_collector._loop_to_thread_collect()
        assert records[0].telemetry_data.amd_ecc_uncorrectable == 0.0
        assert records[1].telemetry_data.amd_ecc_uncorrectable == 2.0
        # The synthesized xid_errors alias is no longer populated.
        assert records[0].telemetry_data.xid_errors is None
        assert records[1].telemetry_data.xid_errors is None


# ---------------------------------------------------------------------------
# Helper: collectors expose _collect_gpu_metrics synchronously, but most
# call sites in this file want to await a thread-friendly variant for parity
# with how the background_task invokes it.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _attach_collect_helper():
    from aiperf.gpu_telemetry.amdsmi_collector import AMDSMITelemetryCollector

    async def _loop_to_thread_collect(self):
        import asyncio

        return await asyncio.to_thread(self._collect_gpu_metrics)

    AMDSMITelemetryCollector._loop_to_thread_collect = _loop_to_thread_collect
    yield
    del AMDSMITelemetryCollector._loop_to_thread_collect
