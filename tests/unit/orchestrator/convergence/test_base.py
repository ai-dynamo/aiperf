# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ConvergenceCriterion base class and _load_request_metrics.

Feature: adaptive-sweep-and-detailed-aggregation
Property 1: JSONL loader extracts exactly the profiling-phase metric values
"""

from pathlib import Path

import orjson
import pytest

from aiperf.orchestrator.convergence.base import ConvergenceCriterion
from aiperf.orchestrator.models import RunResult


class _StubCriterion(ConvergenceCriterion):
    """Concrete stub so we can test the base class utility method."""

    def is_converged(self, results: list[RunResult]) -> bool:
        return False


def _write_jsonl(path: Path, records: list[bytes | dict]) -> None:
    with open(path, "wb") as f:
        for rec in records:
            if isinstance(rec, dict):
                f.write(orjson.dumps(rec))
            else:
                f.write(rec)
            f.write(b"\n")


def _profiling_record(metric: str, value: float) -> dict:
    return {
        "metadata": {
            "benchmark_phase": "profiling",
            "session_num": 0,
            "request_start_ns": 0,
            "request_end_ns": 1,
            "worker_id": "w0",
            "record_processor_id": "rp0",
        },
        "metrics": {metric: {"value": value, "unit": "ms"}},
        "error": None,
    }


def _warmup_record(metric: str, value: float) -> dict:
    rec = _profiling_record(metric, value)
    rec["metadata"]["benchmark_phase"] = "warmup"
    return rec


class TestLoadRequestMetrics:
    """Tests for _load_request_metrics."""

    def test_valid_jsonl_returns_correct_values(self, tmp_path: Path) -> None:
        records = [
            _profiling_record("time_to_first_token", v) for v in [10.0, 20.0, 30.0]
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0, 20.0, 30.0]

    def test_mixed_warmup_and_profiling_filters_correctly(self, tmp_path: Path) -> None:
        records = [
            _warmup_record("time_to_first_token", 999.0),
            _profiling_record("time_to_first_token", 10.0),
            _warmup_record("time_to_first_token", 888.0),
            _profiling_record("time_to_first_token", 20.0),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0, 20.0]

    def test_malformed_json_line_skipped(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "profile_export.jsonl"
        with open(jsonl_path, "wb") as f:
            f.write(orjson.dumps(_profiling_record("ttft", 10.0)))
            f.write(b"\n")
            f.write(b"NOT VALID JSON\n")
            f.write(orjson.dumps(_profiling_record("ttft", 20.0)))
            f.write(b"\n")

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "ttft")
        assert result == [10.0, 20.0]

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "profile_export.jsonl").write_bytes(b"")

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == []

    def test_error_record_skipped(self, tmp_path: Path) -> None:
        error_record = _profiling_record("time_to_first_token", 50.0)
        error_record["error"] = {"message": "timeout", "code": 500}

        records = [
            _profiling_record("time_to_first_token", 10.0),
            error_record,
            _profiling_record("time_to_first_token", 30.0),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0, 30.0]

    def test_record_missing_target_metric_skipped(self, tmp_path: Path) -> None:
        records = [
            _profiling_record("time_to_first_token", 10.0),
            _profiling_record("request_latency", 500.0),
            _profiling_record("time_to_first_token", 20.0),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0, 20.0]

    def test_record_with_null_value_skipped(self, tmp_path: Path) -> None:
        null_val_record = _profiling_record("time_to_first_token", 0.0)
        null_val_record["metrics"]["time_to_first_token"]["value"] = None

        records = [
            _profiling_record("time_to_first_token", 10.0),
            null_val_record,
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0]

    def test_record_missing_metrics_dict_skipped(self, tmp_path: Path) -> None:
        no_metrics = {
            "metadata": {
                "benchmark_phase": "profiling",
                "session_num": 0,
                "request_start_ns": 0,
                "request_end_ns": 1,
                "worker_id": "w0",
                "record_processor_id": "rp0",
            },
            "error": None,
        }
        records = [
            _profiling_record("time_to_first_token", 10.0),
            no_metrics,
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0]

    def test_record_missing_metadata_skipped(self, tmp_path: Path) -> None:
        no_metadata = {
            "metrics": {"time_to_first_token": {"value": 99.0, "unit": "ms"}},
            "error": None,
        }
        records = [
            _profiling_record("time_to_first_token", 10.0),
            no_metadata,
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "time_to_first_token")
        assert result == [10.0]

    def test_blank_lines_in_jsonl_ignored(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "profile_export.jsonl"
        with open(jsonl_path, "wb") as f:
            f.write(orjson.dumps(_profiling_record("ttft", 10.0)))
            f.write(b"\n\n\n")
            f.write(orjson.dumps(_profiling_record("ttft", 20.0)))
            f.write(b"\n")

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "ttft")
        assert result == [10.0, 20.0]

    def test_io_error_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_jsonl(
            tmp_path / "profile_export.jsonl", [_profiling_record("ttft", 10.0)]
        )

        original_open = open

        def _broken_open(path, *args, **kwargs):
            if "profile_export.jsonl" in str(path):
                raise OSError("disk failure")
            return original_open(path, *args, **kwargs)

        import builtins

        monkeypatch.setattr(builtins, "open", _broken_open)

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "ttft")
        assert result == []

    def test_integer_value_converted_to_float(self, tmp_path: Path) -> None:
        record = _profiling_record("output_token_count", 0)
        record["metrics"]["output_token_count"]["value"] = 42

        _write_jsonl(tmp_path / "profile_export.jsonl", [record])

        criterion = _StubCriterion()
        result = criterion._load_request_metrics(tmp_path, "output_token_count")
        assert result == [42.0]
        assert isinstance(result[0], float)
