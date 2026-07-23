# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fake ``amdsmi`` bindings shipped in ``tests/aiperf_mock_amdsmi``.

These load the in-repo package directly via ``sys.path`` so they run whether or
not ``aiperf-mock-amdsmi`` is pip-installed. The final test confirms the real
``AMDSMITelemetryCollector`` consumes the fake end-to-end.
"""

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch, param

_PKG_PARENT = Path(__file__).resolve().parents[3] / "tests" / "aiperf_mock_amdsmi"
_FAKE_MODULES = ("amdsmi", "amdsmi._state", "amdsmi._models")


def _purge_fake() -> None:
    for name in _FAKE_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture
def load_fake_amdsmi(monkeypatch: MonkeyPatch) -> Callable[..., types.ModuleType]:
    """Import a fresh fake ``amdsmi`` module configured via the given env vars."""

    _KNOWN_VARS = (
        "AIPERF_MOCK_AMDSMI",
        "AIPERF_MOCK_AMDSMI_NUM_GPUS",
        "AIPERF_MOCK_AMDSMI_MODEL",
        "AIPERF_MOCK_AMDSMI_GFX_ACTIVITY",
        "AIPERF_MOCK_AMDSMI_POWER_W",
        "AIPERF_MOCK_AMDSMI_TEMP_C",
        "AIPERF_MOCK_AMDSMI_VRAM_USED_FRACTION",
    )

    def _load(**env: object) -> types.ModuleType:
        for var in _KNOWN_VARS:
            monkeypatch.delenv(var, raising=False)
        env.setdefault("AIPERF_MOCK_AMDSMI", "1")
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        monkeypatch.syspath_prepend(str(_PKG_PARENT))
        _purge_fake()
        return importlib.import_module("amdsmi")

    yield _load
    _purge_fake()


class TestDormancyGate:
    def test_import_without_enable_raises_oserror(self, monkeypatch):
        monkeypatch.delenv("AIPERF_MOCK_AMDSMI", raising=False)
        monkeypatch.syspath_prepend(str(_PKG_PARENT))
        _purge_fake()
        with pytest.raises(OSError):
            importlib.import_module("amdsmi")
        _purge_fake()

    def test_import_with_enable_succeeds(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi()
        assert amdsmi.__version__.startswith("26")


class TestAMDHardwareGuard:
    def test_raises_when_kfd_present(self, load_fake_amdsmi, monkeypatch):
        amdsmi = load_fake_amdsmi()
        monkeypatch.setattr(amdsmi, "_amd_hardware_present", lambda: True)
        with pytest.raises(RuntimeError, match="real AMD GPU hardware"):
            amdsmi.amdsmi_init()

    def test_no_error_when_kfd_absent(self, load_fake_amdsmi, monkeypatch):
        amdsmi = load_fake_amdsmi()
        monkeypatch.setattr(amdsmi, "_amd_hardware_present", lambda: False)
        amdsmi.amdsmi_init()


class TestEnumeration:
    def test_num_gpus_respected(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_NUM_GPUS=4)
        amdsmi.amdsmi_init()
        assert len(amdsmi.amdsmi_get_processor_handles()) == 4

    def test_shutdown_clears_handles(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_NUM_GPUS=2)
        amdsmi.amdsmi_init()
        amdsmi.amdsmi_shut_down()
        assert amdsmi.amdsmi_get_processor_handles() == []

    def test_unique_uuid_and_bdf_per_gpu(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_NUM_GPUS=3)
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        uuids = {amdsmi.amdsmi_get_gpu_device_uuid(h) for h in handles}
        bdfs = {amdsmi.amdsmi_get_gpu_device_bdf(h) for h in handles}
        assert len(uuids) == 3
        assert len(bdfs) == 3


class TestModelSpecs:
    @pytest.mark.parametrize(
        ("model", "product_name"),
        [
            param("mi300x", "AMD Instinct MI300X OAM", id="mi300x"),
            param("mi325x", "AMD Instinct MI325X OAM", id="mi325x"),
            param("mi355x", "AMD Instinct MI355X OAM", id="mi355x"),
            param("mi250x", "AMD Instinct MI250X", id="mi250x"),
        ],
    )  # fmt: skip
    def test_product_name_per_model(
        self,
        load_fake_amdsmi: Callable[..., types.ModuleType],
        model: str,
        product_name: str,
    ) -> None:
        amdsmi = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_MODEL=model)
        amdsmi.amdsmi_init()
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        assert amdsmi.amdsmi_get_gpu_board_info(handle)["product_name"] == product_name

    def test_invalid_model_falls_back_to_default(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_MODEL="not-a-real-gpu")
        amdsmi.amdsmi_init()
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        info = amdsmi.amdsmi_get_gpu_board_info(handle)
        assert info["product_name"] == "AMD Instinct MI300X OAM"


class TestReadingQuirks:
    @pytest.fixture
    def amdsmi(
        self, load_fake_amdsmi: Callable[..., types.ModuleType]
    ) -> types.ModuleType:
        mod = load_fake_amdsmi(AIPERF_MOCK_AMDSMI_MODEL="mi300x")
        mod.amdsmi_init()
        return mod

    def test_average_socket_power_is_na(self, amdsmi):
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        power = amdsmi.amdsmi_get_power_info(handle)
        assert power["average_socket_power"] == "N/A"
        assert power["current_socket_power"] == 600.0

    def test_mm_activity_is_na(self, amdsmi):
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        activity = amdsmi.amdsmi_get_gpu_activity(handle)
        assert activity["mm_activity"] == "N/A"
        assert activity["gfx_activity"] == 85.0

    def test_edge_temp_raises_junction_returns_celsius(self, amdsmi):
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        with pytest.raises(amdsmi.AmdSmiException):
            amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
        junction = amdsmi.amdsmi_get_temp_metric(
            handle,
            amdsmi.AmdSmiTemperatureType.JUNCTION,
            amdsmi.AmdSmiTemperatureMetric.CURRENT,
        )
        assert junction == 80.0

    def test_energy_accumulator_is_monotonic(self, amdsmi):
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        first = amdsmi.amdsmi_get_energy_count(handle)["energy_accumulator"]
        second = amdsmi.amdsmi_get_energy_count(handle)["energy_accumulator"]
        assert second > first


class TestOverrides:
    def test_env_overrides_applied(self, load_fake_amdsmi):
        amdsmi = load_fake_amdsmi(
            AIPERF_MOCK_AMDSMI_MODEL="mi300x",
            AIPERF_MOCK_AMDSMI_GFX_ACTIVITY=42.0,
            AIPERF_MOCK_AMDSMI_POWER_W=333.0,
            AIPERF_MOCK_AMDSMI_TEMP_C=55.0,
            AIPERF_MOCK_AMDSMI_VRAM_USED_FRACTION=0.5,
        )
        amdsmi.amdsmi_init()
        handle = amdsmi.amdsmi_get_processor_handles()[0]
        assert amdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"] == 42.0
        assert amdsmi.amdsmi_get_power_info(handle)["current_socket_power"] == 333.0
        assert (
            amdsmi.amdsmi_get_temp_metric(
                handle,
                amdsmi.AmdSmiTemperatureType.JUNCTION,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            )
            == 55.0
        )
        used = amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
        assert used == int(192 * 1024**3 * 0.5)


class TestCollectorConsumesFake:
    @pytest.mark.asyncio
    async def test_collector_produces_amd_records(self, load_fake_amdsmi):
        from aiperf.gpu_telemetry import amdsmi_collector
        from aiperf.gpu_telemetry.amdsmi_collector import AMDSMITelemetryCollector

        fake = load_fake_amdsmi(
            AIPERF_MOCK_AMDSMI_MODEL="mi355x",
            AIPERF_MOCK_AMDSMI_NUM_GPUS=2,
        )

        records = []

        async def record_cb(recs: list, _collector_id: str) -> None:
            records.extend(recs)

        with patch.object(amdsmi_collector, "amdsmi", fake):
            collector = AMDSMITelemetryCollector(record_callback=record_cb)
            await collector.initialize()
            await collector.collect_and_process_metrics()
            await collector.stop()

        assert len(records) == 2
        record = records[0]
        assert record.gpu_model_name == "AMD Instinct MI355X OAM"
        assert record.platform == "amd"
        telemetry = record.telemetry_data
        assert telemetry.amd_power == 1100.0
        assert telemetry.amd_temperature == 85.0
        assert telemetry.amd_gfx_activity == 88.0
        assert telemetry.amd_memory_used > 0
