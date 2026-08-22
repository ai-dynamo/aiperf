# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate native GPU telemetry parity fixtures from pinned origin/main code."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).parent


def module(name: str) -> types.ModuleType:
    result = types.ModuleType(name)
    sys.modules[name] = result
    return result


class Metrics:
    def __init__(self) -> None:
        self._values: dict[str, float] = {}

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_values":
            object.__setattr__(self, name, value)
        else:
            self._values[name] = float(value)

    @property
    def model_fields_set(self) -> set[str]:
        return set(self._values)

    def model_dump(self) -> dict[str, float]:
        return self._values


@dataclass
class Metadata:
    gpu_index: int
    gpu_uuid: str
    gpu_model_name: str
    pci_bus_id: str | None = None
    device: str | None = None
    hostname: str | None = None
    platform: str = "unknown"

    def model_dump(self) -> dict[str, object]:
        return self.__dict__.copy()


class Record:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)


def install_common_stubs() -> None:
    for name in ("aiperf", "aiperf.common", "aiperf.gpu_telemetry"):
        module(name)
    environment = module("aiperf.common.environment")
    environment.Environment = types.SimpleNamespace(
        GPU=types.SimpleNamespace(COLLECTION_INTERVAL=1.0)
    )
    hooks = module("aiperf.common.hooks")
    hooks.background_task = lambda **_: lambda function: function
    hooks.on_init = lambda function: function
    hooks.on_stop = lambda function: function
    mixins = module("aiperf.common.mixins")
    mixins.AIPerfLifecycleMixin = type(
        "AIPerfLifecycleMixin", (), {"__init__": lambda self, **_: None}
    )
    models = module("aiperf.common.models")
    models.ErrorDetails = object
    models.GpuMetadata = Metadata
    models.TelemetryMetrics = Metrics
    models.TelemetryRecord = Record
    constants = module("aiperf.gpu_telemetry.constants")
    constants.NVIDIA_GPU_TELEMETRY_PLATFORM = "nvidia"
    constants.PYNVML_SOURCE_IDENTIFIER = "pynvml://localhost"
    constants.AMD_GPU_TELEMETRY_PLATFORM = "amd"
    constants.AMDSMI_SOURCE_IDENTIFIER = "amdsmi://localhost"
    protocols = module("aiperf.gpu_telemetry.protocols")
    protocols.TErrorCallback = object
    protocols.TRecordCallback = object


def load_collector(path: str, name: str, vendor: types.ModuleType) -> types.ModuleType:
    install_common_stubs()
    sys.modules[vendor.__name__] = vendor
    source = subprocess.check_output(
        ["git", "show", f"origin/main:{path}"], cwd=ROOT, text=True
    )
    result = types.ModuleType(name)
    result.__file__ = path
    sys.modules[name] = result
    exec(compile(source, path, "exec"), result.__dict__)
    return result


def record_json(record: Record) -> dict[str, object]:
    values = record.__dict__.copy()
    metrics = values.pop("telemetry_data")
    values["telemetry_data"] = metrics.model_dump()
    return values


def nvml_fixture() -> list[dict[str, object]]:
    vendor = module("pynvml")
    vendor.NVMLError = type("NVMLError", (Exception,), {})
    vendor.NVML_TEMPERATURE_GPU = 0
    vendor.NVML_PERF_POLICY_POWER = 0
    vendor.nvmlDeviceGetPowerUsage = lambda _: 250_000
    vendor.nvmlDeviceGetTotalEnergyConsumption = lambda _: 3_000_000
    vendor.nvmlDeviceGetUtilizationRates = lambda _: types.SimpleNamespace(
        gpu=80, memory=40
    )
    vendor.nvmlDeviceGetMemoryInfo = lambda _: types.SimpleNamespace(
        used=12_000_000_000
    )
    vendor.nvmlDeviceGetTemperature = lambda *_: 65
    vendor.nvmlDeviceGetDecoderUtilization = lambda _: (12, 0)
    vendor.nvmlDeviceGetEncoderUtilization = lambda _: (34, 0)
    vendor.nvmlDeviceGetJpgUtilization = lambda _: (56, 0)
    vendor.nvmlDeviceGetProcessesUtilizationInfo = lambda *_: [
        types.SimpleNamespace(smUtil=25)
    ]
    vendor.nvmlDeviceGetViolationStatus = lambda *_: types.SimpleNamespace(
        violationTime=4_000
    )
    collector = load_collector(
        "src/aiperf/gpu_telemetry/pynvml_collector.py", "fixture_nvml", vendor
    )
    instance = collector.PyNVMLTelemetryCollector()
    instance._nvml_initialized = True
    instance._gpus = [
        collector.GpuDeviceState(
            "gpu",
            Metadata(
                0, "GPU-nvml", "H100", "0000:01:00.0", "nvidia0", "localhost", "nvidia"
            ),
        )
    ]
    collector.time.time_ns = lambda: 123
    return [record_json(record) for record in instance._collect_gpu_metrics()]


def amdsmi_fixture() -> list[dict[str, object]]:
    vendor = module("amdsmi")
    vendor.AmdSmiException = type("AmdSmiException", (Exception,), {})
    vendor.__version__ = "26.2.2"
    vendor.AmdSmiMemoryType = types.SimpleNamespace(VRAM=0)
    vendor.AmdSmiTemperatureType = types.SimpleNamespace(JUNCTION=1, HOTSPOT=1)
    vendor.AmdSmiTemperatureMetric = types.SimpleNamespace(CURRENT=0)
    vendor.amdsmi_get_power_info = lambda _: {"socket_power": 300}
    vendor.amdsmi_get_energy_count = lambda _: {
        "energy_accumulator": 2_000_000,
        "counter_resolution": 2.0,
    }
    vendor.amdsmi_get_gpu_activity = lambda _: {
        "gfx_activity": 70,
        "umc_activity": 30,
        "mm_activity": "N/A",
    }
    vendor.amdsmi_get_gpu_memory_usage = lambda *_: 16_106_127_360
    vendor.amdsmi_get_temp_metric = lambda *_: 72
    vendor.amdsmi_get_gpu_total_ecc_count = lambda _: {"uncorrectable_count": 3}
    vendor.amdsmi_get_gpu_metrics_info = lambda _: {
        "throttle_status": 0,
        "indep_throttle_status": 2,
    }
    collector = load_collector(
        "src/aiperf/gpu_telemetry/amdsmi_collector.py", "fixture_amdsmi", vendor
    )
    instance = collector.AMDSMITelemetryCollector()
    instance._initialized = True
    instance._gpus = [
        collector._AMDGpuDeviceState(
            "gpu",
            Metadata(
                0, "GPU-amd", "MI300X", "0000:41:00.0", "amd0", "localhost", "amd"
            ),
        )
    ]
    collector.time.time_ns = lambda: 123
    return [record_json(record) for record in instance._collect_gpu_metrics()]


if __name__ == "__main__":
    (OUT / "nvml_origin_main.json").write_text(
        json.dumps(nvml_fixture(), indent=2, sort_keys=True) + "\n"
    )
    (OUT / "amdsmi_origin_main.json").write_text(
        json.dumps(amdsmi_fixture(), indent=2, sort_keys=True) + "\n"
    )
