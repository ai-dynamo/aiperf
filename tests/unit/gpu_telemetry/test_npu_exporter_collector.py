# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NpuExporterTelemetryCollector.

The payload fixtures mirror a real ``npu-exporter`` v7.3.0 scrape captured on an
Ascend 910C (A3) node: one ``npu_chip_info_*`` family per line, the label set
``container_name/id/model_name/namespace/pcie_bus_info/pod_name/vdie_id``, and a
trailing millisecond timestamp. The exporter also emits info-style families
(``npu_chip_info_name``, ``npu_chip_info_serial_number``) whose values are not
numbers; those must be skipped without failing the scrape.
"""

from __future__ import annotations

import pytest

from aiperf.common.mixins.base_metrics_collector_mixin import FetchResult
from aiperf.gpu_telemetry.constants import ASCEND_GPU_TELEMETRY_PLATFORM
from aiperf.gpu_telemetry.npu_exporter_collector import (
    NPU_EXPORTER_TO_FIELD_MAPPING,
    NpuExporterTelemetryCollector,
    _iter_samples,
    _parse_labels,
)

ENDPOINT = "http://10.233.224.133:8082/metrics"

_VDIE_0 = "E29EF66C-00603645-2E2CECF3-83D00485-144301E3"
_VDIE_1 = "E29EF664-01405745-2E2CECF3-83D00485-164301E3"


def _labels(chip_id: str, vdie_id: str | None, pci: str = "0000:9D:00.0") -> str:
    parts = [
        'container_name="vllm"',
        f'id="{chip_id}"',
        'model_name="Ascend910-Ascend-V1"',
        'namespace="qyf-pymotor-glm52"',
        f'pcie_bus_info="{pci}"',
        'pod_name="vllm-p2-7bc5c785b5-btlts"',
    ]
    if vdie_id is not None:
        parts.append(f'vdie_id="{vdie_id}"')
    return ",".join(parts)


def _sample(family: str, chip_id: str, value: float, vdie_id: str | None) -> str:
    return f"{family}{{{_labels(chip_id, vdie_id)}}} {value} 1795486097636"


def _scrape_text(
    *,
    power0: float = 92.9,
    power1: float = 218.5,
    health0: float = 1,
    health1: float = 0,
    include_vdie_id: bool = True,
) -> str:
    """Build a two-chip scrape resembling the real exporter output."""
    v0 = _VDIE_0 if include_vdie_id else None
    v1 = _VDIE_1 if include_vdie_id else None
    lines = [
        "# HELP npu_chip_info_power npu power",
        "# TYPE npu_chip_info_power gauge",
        _sample("npu_chip_info_power", "0", power0, v0),
        _sample("npu_chip_info_power", "1", power1, v1),
        _sample("npu_chip_info_temperature", "0", 37, v0),
        _sample("npu_chip_info_temperature", "1", 61, v1),
        _sample("npu_chip_info_utilization", "0", 47, v0),
        _sample("npu_chip_info_utilization", "1", 96, v1),
        _sample("npu_chip_info_hbm_utilization", "0", 50, v0),
        _sample("npu_chip_info_hbm_utilization", "1", 12, v1),
        # Exponential-form unit (MB) — converted to GB by the collector.
        _sample("npu_chip_info_hbm_used_memory", "0", 65536, v0),
        _sample("npu_chip_info_hbm_used_memory", "1", 8192, v1),
        _sample("npu_chip_info_health_status", "0", health0, v0),
        _sample("npu_chip_info_health_status", "1", health1, v1),
        # Info-style families: non-numeric values must not break the scrape.
        f'npu_chip_info_name{{{_labels("0", v0)}}} "910C" 1795486097636',
        f'npu_chip_info_serial_number{{{_labels("0", v0)}}} "SN-123" 1795486097636',
        # Unmapped numeric family: parsed but ignored.
        _sample("npu_chip_info_aicore_current_freq", "0", 1800, v0),
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def collector() -> NpuExporterTelemetryCollector:
    return NpuExporterTelemetryCollector(ENDPOINT, collection_interval=1.0)


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------


class TestParsingHelpers:
    def test_parse_labels_handles_quoted_values(self) -> None:
        labels = _parse_labels('a="1",b="x,y",c="z"')
        assert labels == {"a": "1", "b": "x,y", "c": "z"}

    def test_parse_labels_empty(self) -> None:
        assert _parse_labels(None) == {}
        assert _parse_labels("") == {}

    def test_iter_samples_skips_comments_and_non_numeric(self) -> None:
        text = "\n".join(
            [
                "# HELP something",
                "",
                'npu_chip_info_name{id="0"} "910C" 123',
                'npu_chip_info_power{id="0"} 92.9 1795486097636',
                "garbage line without value",
            ]
        )
        samples = list(_iter_samples(text))
        assert [(name, labels["id"], value) for name, labels, value in samples] == [
            ("npu_chip_info_power", "0", 92.9)
        ]

    def test_iter_samples_skips_non_finite(self) -> None:
        text = 'npu_chip_info_power{id="0"} NaN\nnpu_chip_info_power{id="1"} 12\n'
        samples = list(_iter_samples(text))
        assert [value for _, _, value in samples] == [12.0]


# ---------------------------------------------------------------------------
# Record mapping
# ---------------------------------------------------------------------------


class TestParseMetricsToRecords:
    def test_one_record_per_chip(self, collector) -> None:
        records = collector._parse_metrics_to_records(_scrape_text())
        assert len(records) == 2
        assert {r.gpu_index for r in records} == {0, 1}

    def test_fields_are_mapped_and_scaled(self, collector) -> None:
        records = collector._parse_metrics_to_records(_scrape_text())
        chip0 = next(r for r in records if r.gpu_index == 0)
        metrics = chip0.telemetry_data

        assert metrics.ascend_power == pytest.approx(92.9)
        assert metrics.ascend_temperature == pytest.approx(37.0)
        assert metrics.ascend_aicore_utilization == pytest.approx(47.0)
        assert metrics.ascend_memory_utilization == pytest.approx(50.0)
        # 65536 MB / 1024 = 64 GB
        assert metrics.ascend_memory_used == pytest.approx(64.0)
        assert metrics.ascend_health == 1.0

    def test_health_zero_maps_to_zero(self, collector) -> None:
        records = collector._parse_metrics_to_records(_scrape_text())
        chip1 = next(r for r in records if r.gpu_index == 1)
        assert chip1.telemetry_data.ascend_health == 0.0

    def test_metadata_comes_from_exporter_labels(self, collector) -> None:
        records = collector._parse_metrics_to_records(_scrape_text())
        chip0 = next(r for r in records if r.gpu_index == 0)

        assert chip0.gpu_uuid == _VDIE_0
        assert chip0.gpu_model_name == "Ascend910-Ascend-V1"
        assert chip0.pci_bus_id == "0000:9D:00.0"
        assert chip0.device == "npu0"
        assert chip0.platform == ASCEND_GPU_TELEMETRY_PLATFORM
        assert chip0.namespace == "qyf-pymotor-glm52"
        assert chip0.pod_name == "vllm-p2-7bc5c785b5-btlts"
        assert chip0.telemetry_source_url == ENDPOINT

    def test_missing_vdie_id_falls_back_to_chip_id(self, collector) -> None:
        records = collector._parse_metrics_to_records(
            _scrape_text(include_vdie_id=False)
        )
        chip0 = next(r for r in records if r.gpu_index == 0)
        assert chip0.gpu_uuid == "NPU-0"

    def test_chip_without_mapped_families_is_omitted(self, collector) -> None:
        # Only an unmapped family is reported for chip 5: no record, never a
        # record full of zeros.
        text = _sample("npu_chip_info_aicore_current_freq", "5", 800, _VDIE_0) + "\n"
        assert collector._parse_metrics_to_records(text) == []

    def test_empty_payload(self, collector) -> None:
        assert collector._parse_metrics_to_records("") == []
        assert collector._parse_metrics_to_records("\n\n") == []

    def test_all_records_share_scrape_timestamp(self, collector) -> None:
        records = collector._parse_metrics_to_records(_scrape_text())
        assert len({r.timestamp_ns for r in records}) == 1
        assert records[0].timestamp_ns > 0


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------


class TestCollection:
    @pytest.mark.asyncio
    async def test_records_are_delivered_via_callback(self) -> None:
        received: list[tuple[list, str]] = []

        async def record_callback(records, collector_id):
            received.append((records, collector_id))

        collector = NpuExporterTelemetryCollector(
            ENDPOINT, record_callback=record_callback, collector_id="npu_exporter_x"
        )
        collector._fetch_metrics_text = _async_return(
            FetchResult(text=_scrape_text(), trace_timing=None)
        )

        await collector._collect_and_process_metrics()

        assert len(received) == 1
        assert len(received[0][0]) == 2
        assert received[0][1] == "npu_exporter_x"

    @pytest.mark.asyncio
    async def test_duplicate_scrape_is_skipped(self) -> None:
        received: list[list] = []

        async def record_callback(records, _collector_id):
            received.append(records)

        collector = NpuExporterTelemetryCollector(
            ENDPOINT, record_callback=record_callback
        )
        collector._fetch_metrics_text = _async_return(
            FetchResult(text=_scrape_text(), trace_timing=None, is_duplicate=True)
        )

        await collector._collect_and_process_metrics()

        assert received == []

    def test_validate_environment_is_a_noop(self) -> None:
        # A URL-based collector must be selectable on a host with no Ascend tools.
        NpuExporterTelemetryCollector.validate_environment()


def _async_return(value):
    """Build an async function returning ``value`` (fetch stub)."""

    async def _inner(*_args, **_kwargs):
        return value

    return _inner


# ---------------------------------------------------------------------------
# Mapping contract
# ---------------------------------------------------------------------------


class TestMappingContract:
    def test_mapping_targets_registered_ascend_fields(self) -> None:
        from aiperf.common.models.telemetry_models import TelemetryMetrics

        for family, field_name in NPU_EXPORTER_TO_FIELD_MAPPING.items():
            assert family.startswith("npu_chip_info_"), family
            assert field_name in TelemetryMetrics.model_fields, field_name
            assert field_name.startswith("ascend_"), field_name

    def test_parity_with_npu_smi_field_set(self) -> None:
        # Both Ascend sources must feed the same six fields so downstream
        # consumers cannot tell them apart.
        from aiperf.gpu_telemetry.constants import (
            ASCEND_AICORE_UTILIZATION_FIELD,
            ASCEND_HEALTH_FIELD,
            ASCEND_MEMORY_USED_FIELD,
            ASCEND_MEMORY_UTILIZATION_FIELD,
            ASCEND_POWER_FIELD,
            ASCEND_TEMPERATURE_FIELD,
        )

        assert set(NPU_EXPORTER_TO_FIELD_MAPPING.values()) == {
            ASCEND_POWER_FIELD,
            ASCEND_TEMPERATURE_FIELD,
            ASCEND_AICORE_UTILIZATION_FIELD,
            ASCEND_MEMORY_UTILIZATION_FIELD,
            ASCEND_MEMORY_USED_FIELD,
            ASCEND_HEALTH_FIELD,
        }


class TestPluginAndCliWiring:
    def test_plugin_is_registered_and_remote(self) -> None:
        from aiperf.plugin import plugins
        from aiperf.plugin.enums import GPUTelemetryCollectorType

        assert GPUTelemetryCollectorType.NPU_EXPORTER == "npu_exporter"
        meta = plugins.get_gpu_telemetry_collector_metadata(
            GPUTelemetryCollectorType.NPU_EXPORTER
        )
        assert meta.is_local is False
        assert (
            plugins.get_class("gpu_telemetry_collector", "npu_exporter").__name__
            == "NpuExporterTelemetryCollector"
        )

    def test_cli_keyword_selects_collector_and_keeps_urls(self) -> None:
        from aiperf.common.enums import GPUTelemetryMode
        from aiperf.config.flags import _converter_telemetry as conv
        from aiperf.plugin.enums import GPUTelemetryCollectorType

        collector_type, mode, urls, metrics_file = conv._classify_gpu_telemetry_items(
            ["npu_exporter", "10.233.224.133:8082/metrics"],
            all_keywords=conv._collector_keywords(),
            collector_type=GPUTelemetryCollectorType.DCGM,
            mode=GPUTelemetryMode.SUMMARY,
        )

        assert collector_type == GPUTelemetryCollectorType.NPU_EXPORTER
        assert urls == ["http://10.233.224.133:8082/metrics"]
        assert metrics_file is None
        assert mode == GPUTelemetryMode.SUMMARY

    def test_conflicting_collector_keywords_are_rejected(self) -> None:
        from aiperf.common.enums import GPUTelemetryMode
        from aiperf.config.flags import _converter_telemetry as conv
        from aiperf.plugin.enums import GPUTelemetryCollectorType

        with pytest.raises(ValueError, match="Conflicting GPU telemetry collectors"):
            conv._classify_gpu_telemetry_items(
                ["npu_exporter", "dcgm", "http://x:8082/metrics"],
                all_keywords=conv._collector_keywords(),
                collector_type=GPUTelemetryCollectorType.DCGM,
                mode=GPUTelemetryMode.SUMMARY,
            )

    def test_config_requires_url_for_npu_exporter(self) -> None:
        from aiperf.config.gpu_telemetry import GpuTelemetryConfig

        with pytest.raises(ValueError, match="requires at least one metrics URL"):
            GpuTelemetryConfig(collector="npu_exporter")

        cfg = GpuTelemetryConfig(
            collector="npu_exporter", urls=["http://10.233.224.133:8082/metrics"]
        )
        assert cfg.collector == "npu_exporter"
        assert cfg.urls == ["http://10.233.224.133:8082/metrics"]
