# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AMDDMETelemetryCollector.

Parses a two-GPU AMD Device Metrics Exporter exposition. Every DME metric is a
Prometheus gauge upstream (ROCm/device-metrics-exporter, gpuagent_gpu_metrics.go),
so the samples here are declared the same way.
"""

import pytest

from aiperf.gpu_telemetry.amd_dme_collector import AMDDMETelemetryCollector
from aiperf.gpu_telemetry.constants import AMD_GPU_TELEMETRY_PLATFORM

LABELS_0 = (
    'gpu_id="0",card_model="Instinct MI300X",serial_number="SN-0",hostname="node-a"'
)
LABELS_1 = (
    'gpu_id="1",card_model="Instinct MI300X",serial_number="SN-1",hostname="node-a"'
)

EXPOSITION = f"""
# TYPE gpu_package_power gauge
gpu_package_power{{{LABELS_0}}} 748
gpu_package_power{{{LABELS_1}}} 751
# TYPE gpu_energy_consumed gauge
gpu_energy_consumed{{{LABELS_0}}} 1.4e17
# TYPE gpu_used_vram gauge
gpu_used_vram{{{LABELS_0}}} 183265
# TYPE gpu_junction_temperature gauge
gpu_junction_temperature{{{LABELS_0}}} 76
# TYPE gpu_clock gauge
gpu_clock{{{LABELS_0},clock_type="GPU_CLOCK_TYPE_SYSTEM",clock_index="0"}} 1533
gpu_clock{{{LABELS_0},clock_type="GPU_CLOCK_TYPE_MEMORY",clock_index="8"}} 1292
gpu_clock{{{LABELS_0},clock_type="GPU_CLOCK_TYPE_VIDEO",clock_index="2"}} 900
"""


@pytest.fixture
def collector() -> AMDDMETelemetryCollector:
    return AMDDMETelemetryCollector(dcgm_url="http://node-a:5000/metrics")


def _by_index(records) -> dict[int, object]:
    return {r.gpu_index: r for r in records}


def test_parses_one_record_per_gpu(collector) -> None:
    records = _by_index(collector._parse_metrics_to_records(EXPOSITION))

    assert set(records) == {0, 1}
    assert records[0].telemetry_data.amd_power == 748.0
    assert records[1].telemetry_data.amd_power == 751.0
    assert records[0].gpu_model_name == "Instinct MI300X"
    assert records[0].gpu_uuid == "SN-0"
    assert records[0].hostname == "node-a"


def test_records_carry_the_amd_platform(collector) -> None:
    """Without this the records default to 'unknown', which reads through to the
    console platform banner and drops the GPUs from the accumulator's per-vendor
    power and energy totals (those look the power field up by platform)."""
    records = collector._parse_metrics_to_records(EXPOSITION)

    assert records
    assert all(r.platform == AMD_GPU_TELEMETRY_PLATFORM for r in records)


def test_clocks_route_by_label_and_unmapped_clocks_are_dropped(collector) -> None:
    record = _by_index(collector._parse_metrics_to_records(EXPOSITION))[0]

    assert record.telemetry_data.amd_sm_clock == 1533.0
    assert record.telemetry_data.amd_mem_clock == 1292.0


def test_scaling_converts_exporter_units(collector) -> None:
    record = _by_index(collector._parse_metrics_to_records(EXPOSITION))[0]

    assert record.telemetry_data.amd_energy_consumption == pytest.approx(1.4e17 * 1e-12)
    assert record.telemetry_data.amd_memory_used == pytest.approx(183265 / 1024)
    assert record.telemetry_data.amd_temperature == 76.0


@pytest.mark.parametrize(
    "sample",
    [
        'gpu_package_power{gpu_id="0",card_model="MI300X",serial_number="SN-0"} NaN',
        'gpu_package_power{card_model="MI300X",serial_number="SN-0"} 700',
        'gpu_package_power{gpu_id="not-an-int",card_model="MI300X"} 700',
    ],
    ids=["nan-value", "no-gpu-id", "unparsable-gpu-id"],
)
def test_unusable_samples_are_skipped(collector, sample: str) -> None:
    assert (
        collector._parse_metrics_to_records(
            f"# TYPE gpu_package_power gauge\n{sample}\n"
        )
        == []
    )


def test_empty_payload_returns_no_records(collector) -> None:
    assert collector._parse_metrics_to_records("   \n") == []


def test_one_bad_gpu_does_not_discard_the_rest_of_the_tick(collector, caplog) -> None:
    """Record construction is per-GPU, so a single out-of-range sample costs
    that GPU only. Building every record inside one try block would let one bad
    GPU blank the whole node for that scrape."""
    import logging

    caplog.set_level(logging.WARNING)
    payload = f"""
# TYPE gpu_package_power gauge
gpu_package_power{{{LABELS_0}}} 748
gpu_package_power{{{LABELS_1}}} 751
# TYPE gpu_free_vram gauge
gpu_free_vram{{{LABELS_1}}} -1
"""

    records = _by_index(collector._parse_metrics_to_records(payload))

    assert set(records) == {0}, "GPU 0's valid record must survive GPU 1's bad sample"
    assert records[0].telemetry_data.amd_power == 748.0


def test_memory_temperature_is_unbounded_like_its_sibling(collector) -> None:
    """amd_temperature carries no lower bound, so amd_memory_temperature should
    not either. A sensor reporting below zero is a reading to pass through, not
    a reason to reject the record."""
    payload = f"""
# TYPE gpu_memory_temperature gauge
gpu_memory_temperature{{{LABELS_0}}} -1
"""

    record = _by_index(collector._parse_metrics_to_records(payload))[0]

    assert record.telemetry_data.amd_memory_temperature == -1.0
