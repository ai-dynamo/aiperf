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


def test_one_bad_gpu_does_not_discard_the_rest_of_the_tick(
    collector, monkeypatch, caplog
) -> None:
    """Record construction is per-GPU, so a record that fails validation costs
    that GPU only. Building every record inside one try block would let one bad
    GPU blank the whole node for that scrape."""
    import logging

    caplog.set_level(logging.WARNING)
    scale = collector._apply_scaling_factors

    def fail_gpu_1(metrics: dict) -> dict:
        scaled = scale(metrics)
        if scaled.get("amd_power") == 751.0:
            scaled["amd_power"] = "not a number"
        return scaled

    monkeypatch.setattr(collector, "_apply_scaling_factors", fail_gpu_1)
    payload = f"""
# TYPE gpu_package_power gauge
gpu_package_power{{{LABELS_0}}} 748
gpu_package_power{{{LABELS_1}}} 751
"""

    records = _by_index(collector._parse_metrics_to_records(payload))

    assert set(records) == {0}, "GPU 0's valid record must survive GPU 1's"
    assert records[0].telemetry_data.amd_power == 748.0
    assert "Dropping GPU 1" in caplog.text


def test_an_out_of_range_reading_does_not_cost_the_record(collector) -> None:
    """None of the AMD fields carry a range bound, so an odd reading is passed
    through rather than taking the GPU's power and activity down with it."""
    payload = f"""
# TYPE gpu_package_power gauge
gpu_package_power{{{LABELS_1}}} 751
# TYPE gpu_free_vram gauge
gpu_free_vram{{{LABELS_1}}} -1
"""

    record = _by_index(collector._parse_metrics_to_records(payload))[1]

    assert record.telemetry_data.amd_power == 751.0
    assert record.telemetry_data.amd_memory_free < 0


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


def _one_gpu(*sample_lines: str) -> str:
    """An exposition for GPU 0 built from bare sample lines, all declared gauges."""
    names = {line.split("{", 1)[0] for line in sample_lines}
    header = "".join(f"# TYPE {name} gauge\n" for name in sorted(names))
    return header + "\n".join(sample_lines) + "\n"


def test_current_dme_clock_labels_are_read(collector) -> None:
    """Current DME lower-cases clock_type and drops the GPU_CLOCK_TYPE_ prefix.

    ROCm/device-metrics-exporter normalises it with
    NormalizeStringWithoutPrefix(clock.Type, "GPU_CLOCK_TYPE_"), which also
    lower-cases. Matching only the prefixed spelling read no clocks at all.
    """
    payload = _one_gpu(
        f'gpu_clock{{{LABELS_0},clock_type="system",clock_index="0"}} 1533',
        f'gpu_clock{{{LABELS_0},clock_type="memory",clock_index="8"}} 1292',
    )
    telemetry = collector._parse_metrics_to_records(payload)[0].telemetry_data

    assert telemetry.amd_sm_clock == 1533.0
    assert telemetry.amd_mem_clock == 1292.0


@pytest.mark.parametrize("lowest_index_first", [True, False])
def test_the_first_clock_of_each_type_wins(collector, lowest_index_first: bool) -> None:
    """clock_index is a list position, not a domain: MI300X lists one system
    clock per XCD before its memory clock, so the lowest index of each type is
    taken rather than a hard-coded position, whichever order samples arrive in."""
    lowest = [
        f'gpu_clock{{{LABELS_0},clock_type="system",clock_index="0"}} 1533',
        f'gpu_clock{{{LABELS_0},clock_type="memory",clock_index="8"}} 1292',
    ]
    later = [
        f'gpu_clock{{{LABELS_0},clock_type="system",clock_index="3"}} 1400',
        f'gpu_clock{{{LABELS_0},clock_type="memory",clock_index="9"}} 900',
    ]
    lines = lowest + later if lowest_index_first else later + lowest
    telemetry = collector._parse_metrics_to_records(_one_gpu(*lines))[0].telemetry_data

    assert telemetry.amd_sm_clock == 1533.0
    assert telemetry.amd_mem_clock == 1292.0


def test_power_falls_back_to_gpu_power_usage(collector) -> None:
    """DME exports gpu_package_power only when the device reports it."""
    payload = _one_gpu(f"gpu_power_usage{{{LABELS_0}}} 430")

    assert (
        collector._parse_metrics_to_records(payload)[0].telemetry_data.amd_power
        == 430.0
    )


@pytest.mark.parametrize("package_power_first", [True, False])
def test_package_power_wins_whichever_family_arrives_first(
    collector, package_power_first: bool
) -> None:
    lines = [
        f"gpu_package_power{{{LABELS_0}}} 748",
        f"gpu_power_usage{{{LABELS_0}}} 430",
    ]
    if not package_power_first:
        lines.reverse()
    payload = "".join(
        f"# TYPE {line.split('{', 1)[0]} gauge\n{line}\n" for line in lines
    )

    assert (
        collector._parse_metrics_to_records(payload)[0].telemetry_data.amd_power
        == 748.0
    )


def test_a_gpu_with_only_unused_samples_produces_no_record(collector) -> None:
    payload = _one_gpu(
        f'gpu_clock{{{LABELS_0},clock_type="video",clock_index="2"}} 900',
        f"gpu_edge_temperature{{{LABELS_0}}} 40",
    )

    assert collector._parse_metrics_to_records(payload) == []


def test_endpoint_credentials_stay_out_of_the_record() -> None:
    """telemetry_source_url becomes a hierarchy key, a metric tag and an export
    field, so any userinfo in the exporter URL must be redacted there."""
    collector = AMDDMETelemetryCollector(
        dcgm_url="http://metrics:s3cr3t@node-a:5000/metrics"
    )
    record = collector._parse_metrics_to_records(
        _one_gpu(f"gpu_package_power{{{LABELS_0}}} 748")
    )[0]

    assert "s3cr3t" not in record.telemetry_source_url
    assert "node-a:5000" in record.telemetry_source_url
