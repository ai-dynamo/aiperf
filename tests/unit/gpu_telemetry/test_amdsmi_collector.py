# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AMDSMITelemetryCollector.

Tests use a mocked amdsmi module to verify collector behavior without requiring
actual AMD ROCm GPU hardware. Empirically validated against MI300X (gfx942)
and MI355X (gfx950) — see ``GpuDeviceState`` notes for AMDSMI quirks.
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

        # Power: passed through unscaled (W).
        assert td0.gpu_power_usage == 287.0

        # Energy: 41_797_534_008_632 ticks * 15.3 µJ/tick / 1e12 ≈ 639.5 MJ
        assert td0.energy_consumption == pytest.approx(639.5, rel=1e-3)

        # Activity: 47% gfx mirrored to both gpu_utilization and sm_utilization.
        assert td0.gpu_utilization == 47.0
        assert td0.sm_utilization == 47.0
        assert td0.mem_utilization == 0.0

        # mm_activity is N/A on Instinct -> encoder/decoder dropped.
        assert td0.encoder_utilization is None
        assert td0.decoder_utilization is None

        # VRAM: 183_678_435_328 bytes -> ~183.68 GB
        assert td0.gpu_memory_used == pytest.approx(183.68, rel=1e-3)

        # Temperature: EDGE failed, JUNCTION returned 67.
        assert td0.gpu_temperature == 67.0

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
        assert td.gpu_temperature is None
        assert td.gpu_power_usage == 287.0  # unaffected

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
            assert r.telemetry_data.gpu_power_usage is None

    @pytest.mark.asyncio
    async def test_throttle_accumulates_across_collections(self, initialized_collector):
        # GPU 0 is throttling; second collection should accumulate >0 µs.
        records1 = await initialized_collector._loop_to_thread_collect()
        records2 = await initialized_collector._loop_to_thread_collect()
        assert records1[0].telemetry_data.power_violation == 0.0
        assert records2[0].telemetry_data.power_violation > 0.0
        # GPU 1 is not throttling -> stays at 0.
        assert records2[1].telemetry_data.power_violation == 0.0

    @pytest.mark.asyncio
    async def test_xid_errors_uses_uncorrectable_count(self, initialized_collector):
        records = await initialized_collector._loop_to_thread_collect()
        assert records[0].telemetry_data.xid_errors == 0.0
        assert records[1].telemetry_data.xid_errors == 2.0


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
