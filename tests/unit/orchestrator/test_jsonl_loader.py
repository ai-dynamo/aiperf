# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.orchestrator.jsonl_loader``.

Covers the three public functions across:
- happy paths (single metric, all metrics, multi-line),
- filter behaviors (skip non-profiling phase, skip error rows, skip malformed
  JSON, skip non-dict records, skip non-numeric values),
- I/O edge cases (missing file → empty/no-yield, unreadable file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.orchestrator.jsonl_loader import (
    DEFAULT_JSONL_FILENAME,
    iter_profiling_records,
    load_all_metrics,
    load_single_metric,
)


def _record(
    *,
    phase: str = "profiling",
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    md = {"benchmark_phase": phase, **(metadata_extra or {})}
    rec: dict[str, Any] = {"metadata": md}
    if metrics is not None:
        rec["metrics"] = metrics
    if error is not None:
        rec["error"] = error
    return rec


def _write_jsonl(path: Path, records: list[Any]) -> None:
    """Records may be dicts (serialized) or raw bytes (passed through)."""
    with open(path, "wb") as f:
        for r in records:
            if isinstance(r, (bytes, bytearray)):
                f.write(r)
            else:
                f.write(orjson.dumps(r))
            f.write(b"\n")


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def jsonl_path(artifacts_dir: Path) -> Path:
    return artifacts_dir / DEFAULT_JSONL_FILENAME


class TestIterProfilingRecords:
    def test_yields_profiling_records(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": {"value": 100.0}}),
                _record(metrics={"ttft": {"value": 200.0}}),
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert len(result) == 2
        assert result[0] == {"ttft": {"value": 100.0}}
        assert result[1] == {"ttft": {"value": 200.0}}

    def test_skips_non_profiling_phase(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(phase="warmup", metrics={"ttft": {"value": 1.0}}),
                _record(phase="profiling", metrics={"ttft": {"value": 100.0}}),
                _record(phase="cooldown", metrics={"ttft": {"value": 999.0}}),
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_skips_error_records(self, artifacts_dir: Path, jsonl_path: Path) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(error="upstream timeout", metrics={"ttft": {"value": 1.0}}),
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_skips_malformed_lines(self, artifacts_dir: Path, jsonl_path: Path) -> None:
        _write_jsonl(
            jsonl_path,
            [
                b"{not valid json",
                _record(metrics={"ttft": {"value": 100.0}}),
                b"also bad",
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_skips_blank_lines(self, artifacts_dir: Path, jsonl_path: Path) -> None:
        with open(jsonl_path, "wb") as f:
            f.write(b"\n\n")
            f.write(orjson.dumps(_record(metrics={"ttft": {"value": 100.0}})))
            f.write(b"\n\n   \n")
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_skips_non_dict_records(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [[1, 2, 3], "string-record", 42, _record(metrics={"ttft": {"value": 1.0}})],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 1.0}}]

    def test_skips_records_with_non_dict_metadata(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                {"metadata": "not_a_dict", "metrics": {"ttft": {"value": 1.0}}},
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_skips_records_with_non_dict_metrics(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                {"metadata": {"benchmark_phase": "profiling"}, "metrics": "not_a_dict"},
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        result = list(iter_profiling_records(artifacts_dir))
        assert result == [{"ttft": {"value": 100.0}}]

    def test_missing_file_yields_nothing(self, artifacts_dir: Path) -> None:
        result = list(iter_profiling_records(artifacts_dir))
        assert result == []

    def test_custom_filename(self, artifacts_dir: Path) -> None:
        path = artifacts_dir / "custom.jsonl"
        _write_jsonl(path, [_record(metrics={"ttft": {"value": 42.0}})])
        result = list(
            iter_profiling_records(artifacts_dir, jsonl_filename="custom.jsonl")
        )
        assert result == [{"ttft": {"value": 42.0}}]


class TestLoadSingleMetric:
    def test_extracts_named_metric(self, artifacts_dir: Path, jsonl_path: Path) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(
                    metrics={"ttft": {"value": 100.0}, "throughput": {"value": 50.0}}
                ),
                _record(metrics={"ttft": {"value": 200.0}}),
            ],
        )
        assert load_single_metric(artifacts_dir, "ttft") == [100.0, 200.0]

    def test_returns_empty_when_metric_absent(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(jsonl_path, [_record(metrics={"throughput": {"value": 50.0}})])
        assert load_single_metric(artifacts_dir, "ttft") == []

    def test_returns_empty_when_file_missing(self, artifacts_dir: Path) -> None:
        assert load_single_metric(artifacts_dir, "ttft") == []

    def test_skips_non_dict_metric_entry(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": "not_a_dict"}),
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        assert load_single_metric(artifacts_dir, "ttft") == [100.0]

    def test_skips_entry_with_no_value(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": {}}),  # no "value" key
                _record(metrics={"ttft": {"value": None}}),  # explicit None
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        assert load_single_metric(artifacts_dir, "ttft") == [100.0]

    def test_skips_non_numeric_values(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": {"value": "not-a-number"}}),
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        assert load_single_metric(artifacts_dir, "ttft") == [100.0]

    def test_coerces_numeric_strings_via_float(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": {"value": "150.5"}}),  # float-coercible string
            ],
        )
        assert load_single_metric(artifacts_dir, "ttft") == [150.5]


class TestLoadAllMetrics:
    def test_collects_every_metric(self, artifacts_dir: Path, jsonl_path: Path) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(
                    metrics={"ttft": {"value": 100.0}, "throughput": {"value": 50.0}}
                ),
                _record(
                    metrics={"ttft": {"value": 200.0}, "throughput": {"value": 60.0}}
                ),
            ],
        )
        result = load_all_metrics(artifacts_dir)
        assert result == {
            "ttft": [100.0, 200.0],
            "throughput": [50.0, 60.0],
        }

    def test_handles_metrics_only_in_some_records(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": {"value": 100.0}}),
                _record(metrics={"throughput": {"value": 50.0}}),
                _record(
                    metrics={"ttft": {"value": 200.0}, "throughput": {"value": 60.0}}
                ),
            ],
        )
        result = load_all_metrics(artifacts_dir)
        assert result == {
            "ttft": [100.0, 200.0],
            "throughput": [50.0, 60.0],
        }

    def test_empty_when_file_missing(self, artifacts_dir: Path) -> None:
        assert load_all_metrics(artifacts_dir) == {}

    def test_empty_when_no_profiling_records(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [_record(phase="warmup", metrics={"ttft": {"value": 100.0}})],
        )
        assert load_all_metrics(artifacts_dir) == {}

    def test_skips_non_numeric_per_metric(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(
                    metrics={"ttft": {"value": "bad"}, "throughput": {"value": 50.0}}
                ),
                _record(metrics={"ttft": {"value": 100.0}}),
            ],
        )
        assert load_all_metrics(artifacts_dir) == {
            "ttft": [100.0],
            "throughput": [50.0],
        }

    def test_skips_non_dict_metric_entries(
        self, artifacts_dir: Path, jsonl_path: Path
    ) -> None:
        _write_jsonl(
            jsonl_path,
            [
                _record(metrics={"ttft": "not_a_dict", "throughput": {"value": 50.0}}),
            ],
        )
        assert load_all_metrics(artifacts_dir) == {"throughput": [50.0]}
