# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AggregateSweepCsvExporter and AggregateSweepJsonExporter.

The constructor surface matches sibling aggregate exporters:
``(config)`` only, with sweep sections nested in
``result.metadata`` (sweep_parameters, num_combinations,
best_configurations, pareto_optimal) and ``result.metrics``
(per_combination_metrics). Wrapper fields (aggregation_type,
num_runs, num_successful_runs, failed_runs) live on the AggregateResult
itself. Output JSON/CSV stays byte-compatible with PR #699.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest

from aiperf.exporters.aggregate import (
    AggregateExporterConfig,
    AggregateSweepCsvExporter,
    AggregateSweepJsonExporter,
)
from aiperf.orchestrator.aggregation.base import AggregateResult


def _build_result(
    sweep_dict: dict[str, Any],
    *,
    num_runs: int = 2,
    num_successful_runs: int = 2,
    failed_runs: list[dict] | None = None,
) -> AggregateResult:
    """Build an AggregateResult from a sweep_dict shape (SweepAnalyzer.compute output).

    Mirrors origin/main's SweepConfidenceStrategy.aggregate: lifts
    best_configurations + pareto_optimal into metadata alongside the rest
    of the sweep metadata, and stuffs per_combination_metrics into the
    metrics field.
    """
    metadata = dict(sweep_dict.get("metadata", {}))
    metadata["best_configurations"] = sweep_dict.get("best_configurations", {})
    metadata["pareto_optimal"] = sweep_dict.get("pareto_optimal", [])
    return AggregateResult(
        aggregation_type="sweep",
        num_runs=num_runs,
        num_successful_runs=num_successful_runs,
        failed_runs=failed_runs or [],
        metadata=metadata,
        metrics=sweep_dict.get("per_combination_metrics", []),
    )


def _exporter_config(
    tmp_path,  # noqa: ANN001
    sweep_dict: dict[str, Any],
    *,
    num_runs: int = 2,
    num_successful_runs: int = 2,
    failed_runs: list[dict] | None = None,
    output_subdir: str | None = None,
) -> AggregateExporterConfig:
    """Build an AggregateExporterConfig with sweep sections wrapped on the result."""
    output_dir = tmp_path / output_subdir if output_subdir else tmp_path
    result = _build_result(
        sweep_dict,
        num_runs=num_runs,
        num_successful_runs=num_successful_runs,
        failed_runs=failed_runs,
    )
    return AggregateExporterConfig(result=result, output_dir=output_dir)


def _two_combo_sweep_dict() -> dict:
    """A sweep dict shaped like the SweepAnalyzer.compute output.

    Two combos at concurrency=10 and concurrency=20, both with throughput
    and latency stats; the throughput-best is concurrency=20, the
    latency-best is concurrency=10.
    """
    return {
        "metadata": {
            "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}],
            "num_combinations": 2,
        },
        "per_combination_metrics": [
            {
                "parameters": {"concurrency": 10},
                "metrics": {
                    "request_throughput_avg": {
                        "mean": 100.0,
                        "std": 1.0,
                        "min": 99.0,
                        "max": 101.0,
                        "cv": 0.01,
                        "unit": "requests/sec",
                    },
                    "time_to_first_token_p99": {
                        "mean": 50.0,
                        "std": 0.5,
                        "min": 49.5,
                        "max": 50.5,
                        "cv": 0.01,
                        "unit": "ms",
                    },
                },
            },
            {
                "parameters": {"concurrency": 20},
                "metrics": {
                    "request_throughput_avg": {
                        "mean": 180.0,
                        "std": 2.0,
                        "min": 178.0,
                        "max": 182.0,
                        "cv": 0.011,
                        "unit": "requests/sec",
                    },
                    "time_to_first_token_p99": {
                        "mean": 80.0,
                        "std": 1.0,
                        "min": 79.0,
                        "max": 81.0,
                        "cv": 0.0125,
                        "unit": "ms",
                    },
                },
            },
        ],
        "best_configurations": {
            "best_throughput": {
                "parameters": {"concurrency": 20},
                "metric": 180.0,
                "unit": "requests/sec",
            },
            "best_latency_p99": {
                "parameters": {"concurrency": 10},
                "metric": 50.0,
                "unit": "ms",
            },
        },
        "pareto_optimal": [{"concurrency": 10}, {"concurrency": 20}],
    }


@pytest.fixture
def three_combo_sweep_dict() -> dict:
    """Three-combination sweep dict (concurrency=10/20/30) matching main's fixture."""
    return {
        "metadata": {
            "sweep_parameters": [{"name": "concurrency", "values": [10, 20, 30]}],
            "num_combinations": 3,
        },
        "per_combination_metrics": [
            {
                "parameters": {"concurrency": 10},
                "metrics": {
                    "request_throughput_avg": {
                        "mean": 100.5,
                        "std": 5.2,
                        "min": 95.0,
                        "max": 108.0,
                        "cv": 0.052,
                        "unit": "requests/sec",
                    },
                    "time_to_first_token_p99": {
                        "mean": 120.5,
                        "std": 8.1,
                        "min": 110.0,
                        "max": 130.0,
                        "cv": 0.067,
                        "unit": "ms",
                    },
                },
            },
            {
                "parameters": {"concurrency": 20},
                "metrics": {
                    "request_throughput_avg": {
                        "mean": 180.2,
                        "std": 9.5,
                        "min": 170.0,
                        "max": 195.0,
                        "cv": 0.053,
                        "unit": "requests/sec",
                    },
                    "time_to_first_token_p99": {
                        "mean": 135.8,
                        "std": 10.2,
                        "min": 125.0,
                        "max": 150.0,
                        "cv": 0.075,
                        "unit": "ms",
                    },
                },
            },
            {
                "parameters": {"concurrency": 30},
                "metrics": {
                    "request_throughput_avg": {
                        "mean": 250.7,
                        "std": 12.3,
                        "min": 235.0,
                        "max": 270.0,
                        "cv": 0.049,
                        "unit": "requests/sec",
                    },
                    "time_to_first_token_p99": {
                        "mean": 155.3,
                        "std": 15.5,
                        "min": 140.0,
                        "max": 175.0,
                        "cv": 0.100,
                        "unit": "ms",
                    },
                },
            },
        ],
        "best_configurations": {
            "best_throughput": {
                "parameters": {"concurrency": 30},
                "metric": 250.7,
                "unit": "requests/sec",
            },
            "best_latency_p99": {
                "parameters": {"concurrency": 10},
                "metric": 120.5,
                "unit": "ms",
            },
        },
        "pareto_optimal": [{"concurrency": 10}, {"concurrency": 30}],
    }


class TestAggregateSweepCsvExporter:
    def test_generate_content_has_header_and_rows(self, tmp_path):
        sweep_dict = _two_combo_sweep_dict()
        cfg = _exporter_config(tmp_path, sweep_dict)
        exporter = AggregateSweepCsvExporter(cfg)

        content = exporter._generate_content()

        # Section 1: per-combination header includes the param column and
        # the per-metric mean/std/min/max/cv quintet.
        rows = list(csv.reader(io.StringIO(content)))
        header = rows[0]
        assert "concurrency" in header
        assert "request_throughput_avg_mean" in header
        assert "request_throughput_avg_cv" in header
        assert "time_to_first_token_p99_mean" in header

        # Two data rows (concurrency=10 and =20)
        assert rows[1][0] == "10"
        assert rows[2][0] == "20"

        # Best Configurations section appears
        assert any(r and r[0] == "Best Configurations" for r in rows)
        # Pareto Optimal section appears
        assert any(r and r[0] == "Pareto Optimal Points" for r in rows)
        # Metadata section appears
        assert any(r and r[0] == "Metadata" for r in rows)

    @pytest.mark.asyncio
    async def test_export_writes_file(self, tmp_path):
        sweep_dict = _two_combo_sweep_dict()
        cfg = _exporter_config(tmp_path, sweep_dict)
        exporter = AggregateSweepCsvExporter(cfg)

        path = await exporter.export()

        assert path.exists()
        assert path.name == "profile_export_aiperf_sweep.csv"
        text = path.read_text()
        assert "request_throughput_avg_mean" in text
        assert "Pareto Optimal Points" in text

    @pytest.mark.asyncio
    async def test_export_csv_creates_file_file_created(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Test that CSV export creates the expected file."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()

        assert csv_path.exists()
        assert csv_path.name == "profile_export_aiperf_sweep.csv"
        assert csv_path.parent == tmp_path / "sweep_aggregate"

    @pytest.mark.asyncio
    async def test_export_csv_format_contains_sections(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Test that CSV export has correct format and sections."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        assert "concurrency" in csv_content
        assert "Best Configurations" in csv_content
        assert "Pareto Optimal Points" in csv_content
        assert "Metadata" in csv_content

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        assert header[0] == "concurrency"
        assert "request_throughput_avg_mean" in header
        assert "time_to_first_token_p99_mean" in header

        assert rows[1][0] == "10"
        assert rows[2][0] == "20"
        assert rows[3][0] == "30"

    @pytest.mark.asyncio
    async def test_export_csv_per_combination_metrics_table_correct_values(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Per-combination metrics table is correctly formatted."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        assert header[0] == "concurrency"

        row_10 = rows[1]
        assert row_10[0] == "10"

        throughput_mean_idx = header.index("request_throughput_avg_mean")
        assert float(row_10[throughput_mean_idx]) == 100.5

    @pytest.mark.asyncio
    async def test_export_csv_best_configurations_section_present(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Best configurations section is present and correct."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        assert "Best Configurations" in csv_content
        assert "Best Throughput" in csv_content
        assert "Best Latency P99" in csv_content

    @pytest.mark.asyncio
    async def test_export_csv_pareto_optimal_section_present(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Pareto optimal section is present."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        assert "Pareto Optimal Points" in csv_content

    @pytest.mark.asyncio
    async def test_export_csv_metadata_section_present(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Metadata section includes aggregation type, sweep parameters, combo count,
        and profile-run counts (PR #699 schema)."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        assert "Metadata" in csv_content
        assert "Aggregation Type" in csv_content
        assert "Sweep Parameters" in csv_content
        assert "Number of Combinations" in csv_content
        assert "Number of Profile Runs" in csv_content
        assert "Number of Successful Runs" in csv_content

    @pytest.mark.asyncio
    async def test_export_csv_metadata_section_carries_run_counts(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Metadata section's run-count rows reflect the AggregateResult wrapper."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=12,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))

        meta_field_to_value = {
            row[0]: row[1] for row in rows if len(row) == 2 and row[0] not in ("Field",)
        }
        assert meta_field_to_value["Aggregation Type"] == "sweep"
        assert meta_field_to_value["Number of Profile Runs"] == "15"
        assert meta_field_to_value["Number of Successful Runs"] == "12"

    @pytest.mark.asyncio
    async def test_export_csv_number_formatting_two_decimal_places(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Numbers in the per-combination table are formatted to two decimals."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        throughput_mean_idx = header.index("request_throughput_avg_mean")
        value = rows[1][throughput_mean_idx]
        assert value == "100.50"

    @pytest.mark.asyncio
    async def test_export_csv_empty_pareto_optimal_reports_none(self, tmp_path):
        """CSV export when no Pareto optimal points exist renders 'None'."""
        sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10]}],
                "num_combinations": 1,
            },
            "per_combination_metrics": [
                {
                    "parameters": {"concurrency": 10},
                    "metrics": {
                        "request_throughput_avg": {
                            "mean": 100.0,
                            "std": 5.0,
                            "min": 95.0,
                            "max": 105.0,
                            "cv": 0.05,
                        },
                    },
                },
            ],
            "best_configurations": {},
            "pareto_optimal": [],
        }
        cfg = _exporter_config(
            tmp_path,
            sweep_dict,
            num_runs=5,
            num_successful_runs=5,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepCsvExporter(cfg)

        csv_path = await exporter.export()
        csv_content = csv_path.read_text()

        assert "Pareto Optimal Points" in csv_content
        assert "None" in csv_content


class TestAggregateSweepJsonExporter:
    @pytest.mark.asyncio
    async def test_export_writes_expected_schema(self, tmp_path):
        sweep_dict = _two_combo_sweep_dict()
        cfg = _exporter_config(tmp_path, sweep_dict)
        exporter = AggregateSweepJsonExporter(cfg)

        path = await exporter.export()

        assert path.exists()
        assert path.name == "profile_export_aiperf_sweep.json"

        data = json.loads(path.read_text())
        # Wrapper fields (PR #699 schema) + sweep sections.
        assert set(data.keys()) >= {
            "aggregation_type",
            "num_profile_runs",
            "num_successful_runs",
            "failed_runs",
            "metadata",
            "per_combination_metrics",
            "best_configurations",
            "pareto_optimal",
        }
        assert data["aggregation_type"] == "sweep"
        assert data["num_profile_runs"] == 2
        assert data["num_successful_runs"] == 2
        assert data["failed_runs"] == []
        assert data["metadata"]["num_combinations"] == 2
        # best_configurations / pareto_optimal must NOT double-emit under metadata.
        assert "best_configurations" not in data["metadata"]
        assert "pareto_optimal" not in data["metadata"]
        assert len(data["per_combination_metrics"]) == 2
        assert data["best_configurations"]["best_throughput"]["parameters"] == {
            "concurrency": 20
        }
        assert data["pareto_optimal"] == [
            {"concurrency": 10},
            {"concurrency": 20},
        ]

    def test_generate_content_serializable_with_numpy_scalars(self, tmp_path):
        """OPT_SERIALIZE_NUMPY must unwrap numpy floats from the sweep dict."""
        import numpy as np

        sweep_dict = _two_combo_sweep_dict()
        # Inject a numpy scalar where SweepAnalyzer would produce one.
        sweep_dict["per_combination_metrics"][0]["metrics"]["request_throughput_avg"][
            "mean"
        ] = np.float64(100.0)
        cfg = _exporter_config(tmp_path, sweep_dict)
        exporter = AggregateSweepJsonExporter(cfg)

        content = exporter._generate_content()
        data = json.loads(content)
        assert (
            data["per_combination_metrics"][0]["metrics"]["request_throughput_avg"][
                "mean"
            ]
            == 100.0
        )

    @pytest.mark.asyncio
    async def test_export_json_creates_file_file_created(
        self, tmp_path, three_combo_sweep_dict
    ):
        """JSON export creates the expected file at the expected path."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepJsonExporter(cfg)

        json_path = await exporter.export()

        assert json_path.exists()
        assert json_path.name == "profile_export_aiperf_sweep.json"
        assert json_path.parent == tmp_path / "sweep_aggregate"

    @pytest.mark.asyncio
    async def test_export_json_schema_compliant_valid_data(
        self, tmp_path, three_combo_sweep_dict
    ):
        """JSON export conforms to the full PR #699 schema."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepJsonExporter(cfg)

        json_path = await exporter.export()
        with open(json_path) as f:
            data = json.load(f)

        # Wrapper fields
        assert data["aggregation_type"] == "sweep"
        assert data["num_profile_runs"] == 15
        assert data["num_successful_runs"] == 15

        # Metadata
        metadata = data["metadata"]
        assert "sweep_parameters" in metadata
        assert metadata["sweep_parameters"][0]["name"] == "concurrency"
        assert metadata["sweep_parameters"][0]["values"] == [10, 20, 30]
        assert metadata["num_combinations"] == 3

        # Per-combination metrics
        per_combination = data["per_combination_metrics"]
        assert len(per_combination) == 3
        combo_0 = per_combination[0]
        assert combo_0["parameters"]["concurrency"] == 10
        metrics_10 = combo_0["metrics"]
        assert metrics_10["request_throughput_avg"]["mean"] == 100.5
        assert metrics_10["request_throughput_avg"]["std"] == 5.2
        assert metrics_10["request_throughput_avg"]["unit"] == "requests/sec"

        # Best configurations
        best = data["best_configurations"]
        assert best["best_throughput"]["parameters"] == {"concurrency": 30}
        assert best["best_throughput"]["metric"] == 250.7
        assert best["best_latency_p99"]["parameters"] == {"concurrency": 10}

        # Pareto optimal
        pareto = data["pareto_optimal"]
        assert len(pareto) == 2
        assert {"concurrency": 10} in pareto
        assert {"concurrency": 30} in pareto

    @pytest.mark.asyncio
    async def test_export_json_with_failed_runs_includes_failed_runs(self, tmp_path):
        """JSON export surfaces failed_runs and the right success counts."""
        sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10, 20]}],
                "num_combinations": 2,
            },
            "per_combination_metrics": [
                {
                    "parameters": {"concurrency": 10},
                    "metrics": {
                        "request_throughput_avg": {
                            "mean": 100.0,
                            "std": 5.0,
                            "min": 95.0,
                            "max": 105.0,
                            "cv": 0.05,
                        },
                    },
                },
            ],
            "best_configurations": {},
            "pareto_optimal": [],
        }
        failed = [
            {"label": "concurrency=20#0", "error": "Connection timeout"},
            {"label": "concurrency=20#1", "error": "Connection timeout"},
        ]
        cfg = _exporter_config(
            tmp_path,
            sweep_dict,
            num_runs=10,
            num_successful_runs=5,
            failed_runs=failed,
            output_subdir="sweep_aggregate",
        )
        exporter = AggregateSweepJsonExporter(cfg)

        json_path = await exporter.export()
        with open(json_path) as f:
            data = json.load(f)

        assert data["num_profile_runs"] == 10
        assert data["num_successful_runs"] == 5
        assert data["failed_runs"] == failed

    @pytest.mark.asyncio
    async def test_export_json_creates_directory_directory_created(
        self, tmp_path, three_combo_sweep_dict
    ):
        """Export creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "sweep_aggregate"
        assert not output_dir.exists()

        result = _build_result(
            three_combo_sweep_dict, num_runs=15, num_successful_runs=15
        )
        cfg = AggregateExporterConfig(result=result, output_dir=output_dir)
        exporter = AggregateSweepJsonExporter(cfg)

        json_path = await exporter.export()

        assert output_dir.exists()
        assert json_path.exists()


class TestSweepExportersIntegration:
    """Integration tests covering both JSON and CSV sweep exporters."""

    @pytest.mark.asyncio
    async def test_exporters_consistency_json_csv_matching_values(
        self, tmp_path, three_combo_sweep_dict
    ):
        """JSON and CSV exporters produce consistent data for the same input."""
        cfg = _exporter_config(
            tmp_path,
            three_combo_sweep_dict,
            num_runs=15,
            num_successful_runs=15,
            output_subdir="sweep_aggregate",
        )
        json_exporter = AggregateSweepJsonExporter(cfg)
        csv_exporter = AggregateSweepCsvExporter(cfg)

        json_path = await json_exporter.export()
        csv_path = await csv_exporter.export()

        with open(json_path) as f:
            json_data = json.load(f)
        csv_content = csv_path.read_text()

        assert str(json_data["metadata"]["num_combinations"]) in csv_content
        assert "concurrency" in csv_content

        for combo_params in json_data["pareto_optimal"]:
            assert str(combo_params["concurrency"]) in csv_content

    @pytest.mark.asyncio
    async def test_exporters_handle_minimal_data_no_exceptions(self, tmp_path):
        """Exporters handle minimal sweep data without raising."""
        minimal_sweep_dict = {
            "metadata": {
                "sweep_parameters": [{"name": "concurrency", "values": [10]}],
                "num_combinations": 1,
            },
            "per_combination_metrics": [],
            "best_configurations": {},
            "pareto_optimal": [],
        }

        cfg = _exporter_config(
            tmp_path,
            minimal_sweep_dict,
            num_runs=1,
            num_successful_runs=1,
            output_subdir="sweep_aggregate",
        )
        json_exporter = AggregateSweepJsonExporter(cfg)
        csv_exporter = AggregateSweepCsvExporter(cfg)

        json_path = await json_exporter.export()
        csv_path = await csv_exporter.export()

        assert json_path.exists()
        assert csv_path.exists()
