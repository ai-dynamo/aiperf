# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AggregateSweepCsvExporter and AggregateSweepJsonExporter."""

from __future__ import annotations

import csv
import io
import json

import pytest

from aiperf.exporters.aggregate import (
    AggregateExporterConfig,
    AggregateSweepCsvExporter,
    AggregateSweepJsonExporter,
)
from aiperf.orchestrator.aggregation.base import AggregateResult


def _exporter_config(tmp_path) -> AggregateExporterConfig:  # noqa: ANN001
    """Build an AggregateExporterConfig with a placeholder AggregateResult.

    Sweep exporters carry the sweep dict separately from the placeholder
    AggregateResult; the latter exists only to satisfy the base class
    constructor surface.
    """
    placeholder = AggregateResult(
        aggregation_type="sweep", num_runs=2, num_successful_runs=2
    )
    return AggregateExporterConfig(result=placeholder, output_dir=tmp_path)


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


class TestAggregateSweepCsvExporter:
    def test_generate_content_has_header_and_rows(self, tmp_path):
        sweep_dict = _two_combo_sweep_dict()
        cfg = _exporter_config(tmp_path)
        exporter = AggregateSweepCsvExporter(cfg, sweep_dict)

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
        cfg = _exporter_config(tmp_path)
        exporter = AggregateSweepCsvExporter(cfg, sweep_dict)

        path = await exporter.export()

        assert path.exists()
        assert path.name == "profile_export_aiperf_sweep.csv"
        text = path.read_text()
        assert "request_throughput_avg_mean" in text
        assert "Pareto Optimal Points" in text


class TestAggregateSweepJsonExporter:
    @pytest.mark.asyncio
    async def test_export_writes_expected_schema(self, tmp_path):
        sweep_dict = _two_combo_sweep_dict()
        cfg = _exporter_config(tmp_path)
        exporter = AggregateSweepJsonExporter(cfg, sweep_dict)

        path = await exporter.export()

        assert path.exists()
        assert path.name == "profile_export_aiperf_sweep.json"

        data = json.loads(path.read_text())
        assert set(data.keys()) >= {
            "metadata",
            "per_combination_metrics",
            "best_configurations",
            "pareto_optimal",
        }
        assert data["metadata"]["num_combinations"] == 2
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
        cfg = _exporter_config(tmp_path)
        exporter = AggregateSweepJsonExporter(cfg, sweep_dict)

        content = exporter._generate_content()
        data = json.loads(content)
        assert (
            data["per_combination_metrics"][0]["metrics"]["request_throughput_avg"][
                "mean"
            ]
            == 100.0
        )
