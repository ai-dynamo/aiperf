# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for DetailedAggregation strategy."""

import numpy as np
import orjson
import pytest

from aiperf.orchestrator.aggregation.detailed import DetailedAggregation
from aiperf.orchestrator.models import RunResult


class TestDetailedAggregation:
    """Tests for DetailedAggregation."""

    def test_get_aggregation_type(self) -> None:
        agg = DetailedAggregation()
        assert agg.get_aggregation_type() == "detailed"

    def test_aggregate_three_runs_combined_percentiles(
        self, make_results_with_jsonl
    ) -> None:
        """3 runs with known values -- verify combined percentiles match numpy.percentile."""
        run1 = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        run2 = np.array([105.0, 115.0, 125.0, 135.0, 145.0])
        run3 = np.array([102.0, 112.0, 122.0, 132.0, 142.0])
        results = make_results_with_jsonl([run1, run2, run3])

        agg = DetailedAggregation()
        result = agg.aggregate(results)

        combined = np.concatenate([run1, run2, run3])
        metric = result.metrics["time_to_first_token"]
        assert metric["combined"]["count"] == len(combined)
        assert metric["combined"]["mean"] == pytest.approx(float(np.mean(combined)))
        assert metric["combined"]["std"] == pytest.approx(
            float(np.std(combined, ddof=1))
        )
        assert metric["combined"]["p50"] == pytest.approx(
            float(np.percentile(combined, 50))
        )
        assert metric["combined"]["p90"] == pytest.approx(
            float(np.percentile(combined, 90))
        )
        assert metric["combined"]["p95"] == pytest.approx(
            float(np.percentile(combined, 95))
        )
        assert metric["combined"]["p99"] == pytest.approx(
            float(np.percentile(combined, 99))
        )

    def test_aggregate_count_equals_sum_of_per_run_counts(
        self, make_results_with_jsonl
    ) -> None:
        run1 = np.array([1.0, 2.0, 3.0])
        run2 = np.array([4.0, 5.0, 6.0, 7.0])
        run3 = np.array([8.0, 9.0])
        results = make_results_with_jsonl([run1, run2, run3])

        result = DetailedAggregation().aggregate(results)
        metric = result.metrics["time_to_first_token"]

        per_run_total = sum(entry["count"] for entry in metric["per_run"])
        assert metric["combined"]["count"] == per_run_total
        assert metric["combined"]["count"] == 9

    def test_per_run_breakdown_labels_and_means(self, make_results_with_jsonl) -> None:
        run1 = np.array([10.0, 20.0, 30.0])
        run2 = np.array([40.0, 50.0, 60.0])
        results = make_results_with_jsonl([run1, run2])

        result = DetailedAggregation().aggregate(results)
        per_run = result.metrics["time_to_first_token"]["per_run"]

        assert len(per_run) == 2
        assert per_run[0]["label"] == "run_0001"
        assert per_run[0]["mean"] == pytest.approx(20.0)
        assert per_run[0]["count"] == 3
        assert per_run[1]["label"] == "run_0002"
        assert per_run[1]["mean"] == pytest.approx(50.0)
        assert per_run[1]["count"] == 3

    def test_mixed_success_failure_only_successful_runs(
        self, make_results_with_jsonl
    ) -> None:
        run1 = np.array([100.0, 200.0])
        results = make_results_with_jsonl([run1])
        results.append(RunResult(label="run_0002", success=False, error="timeout"))

        result = DetailedAggregation().aggregate(results)

        assert result.num_runs == 2
        assert result.num_successful_runs == 1
        assert len(result.failed_runs) == 1
        assert result.failed_runs[0]["label"] == "run_0002"
        metric = result.metrics["time_to_first_token"]
        assert len(metric["per_run"]) == 1
        assert metric["combined"]["count"] == 2

    def test_missing_jsonl_skipped(self, tmp_path) -> None:
        """Successful run with no JSONL file is skipped gracefully."""
        result_no_jsonl = RunResult(
            label="run_0001",
            success=True,
            artifacts_path=tmp_path / "empty_run",
        )
        (tmp_path / "empty_run").mkdir()

        result = DetailedAggregation().aggregate([result_no_jsonl])

        assert result.aggregation_type == "detailed"
        assert result.num_successful_runs == 1
        assert result.metrics == {}

    def test_empty_jsonl_skipped(self, tmp_path) -> None:
        """Successful run with empty JSONL file is skipped gracefully."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_bytes(b"")

        result_empty = RunResult(
            label="run_0001",
            success=True,
            artifacts_path=run_dir,
        )

        result = DetailedAggregation().aggregate([result_empty])

        assert result.aggregation_type == "detailed"
        assert result.metrics == {}

    def test_single_run_valid_output(self, make_results_with_jsonl) -> None:
        values = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
        results = make_results_with_jsonl([values])

        result = DetailedAggregation().aggregate(results)

        metric = result.metrics["time_to_first_token"]
        assert metric["combined"]["count"] == 5
        assert metric["combined"]["mean"] == pytest.approx(float(np.mean(values)))
        assert metric["combined"]["p50"] == pytest.approx(
            float(np.percentile(values, 50))
        )
        assert len(metric["per_run"]) == 1
        # Single run: std with ddof=1
        assert metric["combined"]["std"] == pytest.approx(float(np.std(values, ddof=1)))

    def test_aggregation_type_in_result(self, make_results_with_jsonl) -> None:
        results = make_results_with_jsonl([np.array([1.0, 2.0])])
        result = DetailedAggregation().aggregate(results)
        assert result.aggregation_type == "detailed"

    def test_empty_results_list(self) -> None:
        result = DetailedAggregation().aggregate([])
        assert result.aggregation_type == "detailed"
        assert result.num_runs == 0
        assert result.num_successful_runs == 0
        assert result.metrics == {}

    def test_all_failed_runs(self) -> None:
        results = [
            RunResult(label="run_0001", success=False, error="crash"),
            RunResult(label="run_0002", success=False, error="timeout"),
        ]
        result = DetailedAggregation().aggregate(results)

        assert result.num_runs == 2
        assert result.num_successful_runs == 0
        assert result.metrics == {}
        assert len(result.failed_runs) == 2

    def test_no_artifacts_path_skipped(self) -> None:
        """Successful run with artifacts_path=None is skipped."""
        results = [RunResult(label="run_0001", success=True, artifacts_path=None)]
        result = DetailedAggregation().aggregate(results)
        assert result.metrics == {}

    def test_warmup_records_excluded(self, tmp_path) -> None:
        """Records with benchmark_phase='warmup' are not included."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        with open(jsonl_path, "wb") as f:
            for phase, val in [
                ("warmup", 999.0),
                ("profiling", 42.0),
                ("profiling", 43.0),
            ]:
                record = {
                    "metadata": {
                        "benchmark_phase": phase,
                        "session_num": 0,
                        "request_start_ns": 0,
                        "request_end_ns": 1,
                        "worker_id": "w0",
                        "record_processor_id": "rp0",
                    },
                    "metrics": {"time_to_first_token": {"value": val, "unit": "ms"}},
                    "error": None,
                }
                f.write(orjson.dumps(record))
                f.write(b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)

        metric = result.metrics["time_to_first_token"]
        assert metric["combined"]["count"] == 2
        assert metric["combined"]["mean"] == pytest.approx(42.5)

    def test_error_records_excluded(self, tmp_path) -> None:
        """Records with non-null error are excluded."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        with open(jsonl_path, "wb") as f:
            for error, val in [(None, 10.0), ("request failed", 999.0), (None, 20.0)]:
                record = {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "session_num": 0,
                        "request_start_ns": 0,
                        "request_end_ns": 1,
                        "worker_id": "w0",
                        "record_processor_id": "rp0",
                    },
                    "metrics": {"time_to_first_token": {"value": val, "unit": "ms"}},
                    "error": error,
                }
                f.write(orjson.dumps(record))
                f.write(b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)

        metric = result.metrics["time_to_first_token"]
        assert metric["combined"]["count"] == 2
        assert metric["combined"]["mean"] == pytest.approx(15.0)

    def test_malformed_jsonl_lines_skipped(self, tmp_path) -> None:
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        good_record = orjson.dumps(
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "session_num": 0,
                    "request_start_ns": 0,
                    "request_end_ns": 1,
                    "worker_id": "w0",
                    "record_processor_id": "rp0",
                },
                "metrics": {"time_to_first_token": {"value": 50.0, "unit": "ms"}},
                "error": None,
            }
        )
        with open(jsonl_path, "wb") as f:
            f.write(good_record + b"\n")
            f.write(b"not valid json\n")
            f.write(good_record + b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)

        assert result.metrics["time_to_first_token"]["combined"]["count"] == 2

    @pytest.mark.parametrize(
        "metric_name",
        [
            "time_to_first_token",
            "request_latency",
            "embedding_latency",
            "audio_duration",
        ],
    )
    def test_varied_metric_names_same_schema(
        self, make_results_with_jsonl, metric_name
    ) -> None:
        """Chat, embeddings, audio metrics all produce the same output schema."""
        values = np.array([10.0, 20.0, 30.0])
        results = make_results_with_jsonl([values, values], metric=metric_name)

        result = DetailedAggregation().aggregate(results)

        metric = result.metrics[metric_name]
        assert "combined" in metric
        assert "per_run" in metric
        for key in ("mean", "std", "p50", "p90", "p95", "p99", "count"):
            assert key in metric["combined"], f"Missing key: {key}"
        assert len(metric["per_run"]) == 2
        for entry in metric["per_run"]:
            assert "label" in entry
            assert "mean" in entry
            assert "count" in entry

    def test_multiple_metrics_in_same_jsonl(self, tmp_path) -> None:
        """JSONL records with multiple metrics produce separate entries in output."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        with open(jsonl_path, "wb") as f:
            for i in range(5):
                record = {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "session_num": i,
                        "request_start_ns": 0,
                        "request_end_ns": 1,
                        "worker_id": "w0",
                        "record_processor_id": "rp0",
                    },
                    "metrics": {
                        "time_to_first_token": {"value": 100.0 + i, "unit": "ms"},
                        "request_latency": {"value": 500.0 + i, "unit": "ms"},
                    },
                    "error": None,
                }
                f.write(orjson.dumps(record))
                f.write(b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)

        assert "time_to_first_token" in result.metrics
        assert "request_latency" in result.metrics
        assert result.metrics["time_to_first_token"]["combined"]["count"] == 5
        assert result.metrics["request_latency"]["combined"]["count"] == 5

    def test_metadata_contains_run_labels(self, make_results_with_jsonl) -> None:
        results = make_results_with_jsonl([np.array([1.0]), np.array([2.0])])
        result = DetailedAggregation().aggregate(results)
        assert result.metadata["run_labels"] == ["run_0001", "run_0002"]

    def test_blank_lines_in_jsonl_skipped(self, tmp_path) -> None:
        """Blank lines in JSONL are skipped without error."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        good_record = orjson.dumps(
            {
                "metadata": {"benchmark_phase": "profiling"},
                "metrics": {"time_to_first_token": {"value": 42.0, "unit": "ms"}},
                "error": None,
            }
        )
        with open(jsonl_path, "wb") as f:
            f.write(good_record + b"\n")
            f.write(b"\n")
            f.write(b"   \n")
            f.write(good_record + b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)
        assert result.metrics["time_to_first_token"]["combined"]["count"] == 2

    def test_non_dict_metric_entry_skipped(self, tmp_path) -> None:
        """Metric entries that are not dicts (e.g. a raw number) are skipped."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        record = {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {
                "bad_metric": "not_a_dict",
                "also_bad": 123,
                "good_metric": {"value": 10.0, "unit": "ms"},
            },
            "error": None,
        }
        with open(jsonl_path, "wb") as f:
            f.write(orjson.dumps(record) + b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)
        assert "bad_metric" not in result.metrics
        assert "also_bad" not in result.metrics
        assert result.metrics["good_metric"]["combined"]["count"] == 1

    def test_io_error_returns_empty(self, tmp_path) -> None:
        """OSError during JSONL read returns empty metrics for that run."""
        from unittest.mock import patch

        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"
        jsonl_path.write_bytes(b"data\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]

        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = DetailedAggregation().aggregate(results)

        assert result.metrics == {}

    def test_metric_value_none_skipped(self, tmp_path) -> None:
        """Metric entry with value=None is skipped."""
        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        jsonl_path = run_dir / "profile_export.jsonl"

        record = {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {
                "ttft": {"value": None, "unit": "ms"},
                "latency": {"value": 5.0, "unit": "ms"},
            },
            "error": None,
        }
        with open(jsonl_path, "wb") as f:
            f.write(orjson.dumps(record) + b"\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        result = DetailedAggregation().aggregate(results)
        assert "ttft" not in result.metrics
        assert result.metrics["latency"]["combined"]["count"] == 1

    def test_empty_values_after_concatenation_skipped(self, tmp_path) -> None:
        """Metric with empty arrays in per_run_data is skipped."""
        from unittest.mock import patch

        run_dir = tmp_path / "run_0001"
        run_dir.mkdir()
        (run_dir / "profile_export.jsonl").write_bytes(b"{}\n")

        results = [RunResult(label="run_0001", success=True, artifacts_path=run_dir)]
        agg = DetailedAggregation()

        with patch(
            "aiperf.orchestrator.aggregation.detailed.load_all_metrics",
            return_value={"phantom_metric": []},
        ):
            result = agg.aggregate(results)

        assert "phantom_metric" not in result.metrics
