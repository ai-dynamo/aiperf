# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPEED-Bench matrix report module."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.analysis.speed_bench_report import (
    PROFILE_JSON,
    SERVER_METRICS_JSON,
    SpeedBenchReportError,
    _get_metric_stat,
    build_report,
    detect_columns,
    extract_accept_length,
    extract_accept_rate,
    extract_category,
    extract_model,
    extract_throughput,
    find_run_dirs,
    generate_report,
    load_profile,
    load_server_metrics,
    print_table,
    write_csv,
)


def _server_metric(name: str, stats: dict) -> dict:
    """Construct a ``{metrics: {name: {series: [{stats}]}}}`` wrapper."""
    return {"metrics": {name: {"series": [{"stats": stats}]}}}


def _profile(dataset: str | None = None, model: str | None = "test-model") -> dict:
    """Construct a minimal profile export with the fields the report reads."""
    endpoint: dict = {}
    if model is not None:
        endpoint["model_names"] = [model]
    input_block: dict = {}
    if dataset is not None:
        input_block["public_dataset"] = dataset
    return {"input_config": {"endpoint": endpoint, "input": input_block}}


def _write_run_dir(
    tmp_path: Path,
    name: str,
    profile: dict | None,
    server_metrics: dict | None = None,
) -> Path:
    """Materialize a fake run directory on disk and return its path."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    if profile is not None:
        (run_dir / PROFILE_JSON).write_bytes(orjson.dumps(profile))
    if server_metrics is not None:
        (run_dir / SERVER_METRICS_JSON).write_bytes(orjson.dumps(server_metrics))
    return run_dir


class TestExtractCategory:
    def test_extract_category_valid_prefix_returns_suffix(self):
        assert extract_category(_profile(dataset="speed_bench_coding")) == "coding"

    def test_extract_category_missing_input_returns_none(self):
        assert extract_category({"input_config": {"endpoint": {}}}) is None

    def test_extract_category_non_speed_bench_dataset_returns_none(self):
        assert extract_category(_profile(dataset="sharegpt")) is None

    def test_extract_category_non_string_dataset_returns_none(self):
        profile = {"input_config": {"input": {"public_dataset": 42}}}
        assert extract_category(profile) is None

    def test_extract_category_missing_input_config_returns_none(self):
        assert extract_category({}) is None


class TestExtractModel:
    def test_extract_model_returns_first_name(self):
        assert extract_model(_profile(model="llama-3.1")) == "llama-3.1"

    def test_extract_model_empty_names_falls_back_to_unknown(self):
        profile = {"input_config": {"endpoint": {"model_names": []}}}
        assert extract_model(profile) == "unknown"

    def test_extract_model_missing_endpoint_falls_back_to_unknown(self):
        assert extract_model({"input_config": {}}) == "unknown"

    def test_extract_model_missing_input_config_falls_back_to_unknown(self):
        assert extract_model({}) == "unknown"


class TestExtractAcceptLength:
    def test_extract_accept_length_sglang_gauge_takes_priority(self):
        metrics = _server_metric("sglang:spec_accept_length", {"avg": 2.5})
        assert extract_accept_length(metrics) == 2.5

    def test_extract_accept_length_vllm_counters_compute_ratio_plus_one(self):
        # acceptance_length = (accepted / drafts) + 1 = (300/200) + 1 = 2.5
        metrics = {
            "metrics": {
                "vllm:spec_decode_num_accepted_tokens": {
                    "series": [{"stats": {"total": 300.0}}]
                },
                "vllm:spec_decode_num_drafts": {
                    "series": [{"stats": {"total": 200.0}}]
                },
            }
        }
        assert extract_accept_length(metrics) == 2.5

    def test_extract_accept_length_vllm_zero_drafts_falls_through(self):
        metrics = {
            "metrics": {
                "vllm:spec_decode_num_accepted_tokens": {
                    "series": [{"stats": {"total": 100.0}}]
                },
                "vllm:spec_decode_num_drafts": {"series": [{"stats": {"total": 0.0}}]},
            }
        }
        assert extract_accept_length(metrics) is None

    def test_extract_accept_length_fuzzy_fallback_matches_spec_metric(self):
        # Fuzzy fallback requires all three of "spec", "accept", "length".
        metrics = _server_metric("custom_engine_spec_accept_length", {"avg": 1.8})
        assert extract_accept_length(metrics) == 1.8

    def test_extract_accept_length_fuzzy_fallback_ignores_non_spec_metric(self):
        # Metrics that happen to mention accept+length but aren't speculative-
        # decoding metrics must not match.
        metrics = _server_metric("request_acceptance_total_length", {"avg": 99.0})
        assert extract_accept_length(metrics) is None

    def test_extract_accept_length_no_matching_metric_returns_none(self):
        assert extract_accept_length({"metrics": {}}) is None

    def test_extract_accept_length_prefers_sglang_over_vllm(self):
        metrics = {
            "metrics": {
                "sglang:spec_accept_length": {"series": [{"stats": {"avg": 3.0}}]},
                "vllm:spec_decode_num_accepted_tokens": {
                    "series": [{"stats": {"total": 100.0}}]
                },
                "vllm:spec_decode_num_drafts": {
                    "series": [{"stats": {"total": 100.0}}]
                },
            }
        }
        assert extract_accept_length(metrics) == 3.0


class TestExtractAcceptRate:
    def test_extract_accept_rate_sglang_gauge_takes_priority(self):
        metrics = _server_metric("sglang:spec_accept_rate", {"avg": 0.75})
        assert extract_accept_rate(metrics) == 0.75

    def test_extract_accept_rate_vllm_counters_compute_ratio(self):
        # rate = accepted / draft_tokens = 150/200 = 0.75
        metrics = {
            "metrics": {
                "vllm:spec_decode_num_accepted_tokens": {
                    "series": [{"stats": {"total": 150.0}}]
                },
                "vllm:spec_decode_num_draft_tokens": {
                    "series": [{"stats": {"total": 200.0}}]
                },
            }
        }
        assert extract_accept_rate(metrics) == 0.75

    def test_extract_accept_rate_zero_draft_tokens_returns_none(self):
        metrics = {
            "metrics": {
                "vllm:spec_decode_num_accepted_tokens": {
                    "series": [{"stats": {"total": 100.0}}]
                },
                "vllm:spec_decode_num_draft_tokens": {
                    "series": [{"stats": {"total": 0.0}}]
                },
            }
        }
        assert extract_accept_rate(metrics) is None

    def test_extract_accept_rate_no_metrics_returns_none(self):
        assert extract_accept_rate({"metrics": {}}) is None


class TestExtractThroughput:
    def test_extract_throughput_reads_output_token_throughput_avg(self):
        profile = {"output_token_throughput": {"avg": 123.4}}
        assert extract_throughput(profile) == 123.4

    def test_extract_throughput_missing_key_returns_none(self):
        assert extract_throughput({}) is None

    def test_extract_throughput_avg_none_returns_none(self):
        assert extract_throughput({"output_token_throughput": {"avg": None}}) is None


class TestDetectColumns:
    def test_detect_columns_all_qualitative_returns_ordered_subset(self):
        results = {
            "model-a": {"coding": 1.0, "math": 2.0, "writing": 3.0},
        }
        # Ordering must follow QUALITATIVE_CATEGORIES, not insertion order.
        assert detect_columns(results) == ["coding", "math", "writing"]

    def test_detect_columns_all_throughput_tiers_returns_tier_order(self):
        results = {
            "model-a": {"high_entropy": 1.0, "low_entropy": 2.0, "mixed": 3.0},
        }
        assert detect_columns(results) == ["low_entropy", "mixed", "high_entropy"]

    def test_detect_columns_mixed_categories_returns_sorted(self):
        # Mixing qualitative with throughput tier forces alphabetical fallback.
        results = {"model-a": {"coding": 1.0, "mixed": 2.0}}
        assert detect_columns(results) == ["coding", "mixed"]

    def test_detect_columns_empty_results_returns_empty_list(self):
        assert detect_columns({}) == []


class TestFindRunDirs:
    def test_find_run_dirs_parent_dir_discovers_children(self, tmp_path: Path):
        _write_run_dir(tmp_path, "run_coding", _profile(dataset="speed_bench_coding"))
        _write_run_dir(tmp_path, "run_math", _profile(dataset="speed_bench_math"))
        (tmp_path / "not_a_run").mkdir()  # directory without profile export

        discovered = find_run_dirs([tmp_path])

        assert sorted(d.name for d in discovered) == ["run_coding", "run_math"]

    def test_find_run_dirs_direct_run_dir_included(self, tmp_path: Path):
        run = _write_run_dir(
            tmp_path, "run_coding", _profile(dataset="speed_bench_coding")
        )
        assert find_run_dirs([run]) == [run]

    def test_find_run_dirs_non_directory_path_is_skipped(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        assert find_run_dirs([missing]) == []


class TestLoadJson:
    def test_load_profile_reads_valid_json(self, tmp_path: Path):
        run = _write_run_dir(tmp_path, "run", _profile(dataset="speed_bench_coding"))
        loaded = load_profile(run)
        assert loaded is not None
        assert loaded["input_config"]["input"]["public_dataset"] == "speed_bench_coding"

    def test_load_profile_missing_file_returns_none(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert load_profile(empty) is None

    def test_load_profile_malformed_json_returns_none(self, tmp_path: Path):
        run = tmp_path / "bad"
        run.mkdir()
        (run / PROFILE_JSON).write_text("not json {")
        assert load_profile(run) is None

    def test_load_server_metrics_reads_valid_json(self, tmp_path: Path):
        run = _write_run_dir(
            tmp_path,
            "run",
            _profile(dataset="speed_bench_coding"),
            server_metrics={"metrics": {}},
        )
        assert load_server_metrics(run) == {"metrics": {}}

    def test_load_server_metrics_missing_file_returns_none(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert load_server_metrics(empty) is None

    def test_load_server_metrics_malformed_json_returns_none(self, tmp_path: Path):
        run = tmp_path / "bad"
        run.mkdir()
        (run / SERVER_METRICS_JSON).write_text("not json {")
        assert load_server_metrics(run) is None


class TestGetMetricStat:
    def test_get_metric_stat_empty_series_returns_none(self):
        # Metric present, but its series list is empty — should short-circuit.
        metrics = {"some_metric": {"series": []}}
        assert _get_metric_stat(metrics, "some_metric", "total") is None

    def test_get_metric_stat_missing_metric_returns_none(self):
        assert _get_metric_stat({}, "missing", "total") is None


class TestBuildReport:
    def test_build_report_accept_length_per_model_and_category(self, tmp_path: Path):
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_length", {"avg": 2.1}),
        )
        _write_run_dir(
            tmp_path,
            "run_math",
            _profile(dataset="speed_bench_math", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_length", {"avg": 3.3}),
        )

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="accept_length")

        assert report == {"m1": {"coding": 2.1, "math": 3.3}}

    def test_build_report_throughput_reads_profile_not_server_metrics(
        self, tmp_path: Path
    ):
        profile = _profile(dataset="speed_bench_coding", model="m1")
        profile["output_token_throughput"] = {"avg": 512.0}
        _write_run_dir(tmp_path, "run_coding", profile)  # no server metrics file

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="throughput")

        assert report == {"m1": {"coding": 512.0}}

    def test_build_report_missing_server_metrics_records_none(self, tmp_path: Path):
        # accept_length requires server metrics; absence should surface as None rather than crash.
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
        )

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="accept_length")

        assert report == {"m1": {"coding": None}}

    def test_build_report_skips_runs_with_non_speed_bench_dataset(self, tmp_path: Path):
        _write_run_dir(tmp_path, "run_other", _profile(dataset="sharegpt", model="m1"))

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="accept_length")

        # sharegpt lacks the speed_bench_ prefix, so the run is skipped entirely.
        assert report == {}

    def test_build_report_empty_server_metrics_dict_still_dispatches_extractor(
        self, tmp_path: Path
    ):
        # A valid-but-empty server_metrics_export.json must not be misreported
        # as "missing"; the extractor should run and return None on no matches.
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics={},
        )
        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="accept_length")

        assert report == {"m1": {"coding": None}}

    def test_build_report_unknown_metric_type_records_none(self, tmp_path: Path):
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_length", {"avg": 2.0}),
        )

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="nonsense")  # type: ignore[arg-type]

        assert report == {"m1": {"coding": None}}

    def test_build_report_accept_rate_dispatches_to_rate_extractor(
        self, tmp_path: Path
    ):
        # Drives the accept_rate branch that existing tests skipped.
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_rate", {"avg": 0.8}),
        )

        run_dirs = find_run_dirs([tmp_path])
        report = build_report(run_dirs, metric_type="accept_rate")

        assert report == {"m1": {"coding": 0.8}}

    def test_build_report_skips_run_dir_without_profile_file(self, tmp_path: Path):
        # build_report can be called with run dirs lacking profile_export_aiperf.json
        # (e.g. caller passes dirs directly rather than through find_run_dirs).
        empty = tmp_path / "no_profile"
        empty.mkdir()
        assert build_report([empty], metric_type="accept_length") == {}


class TestPrintTable:
    def test_print_table_rich_branch_renders_model_and_overall(
        self, capsys: pytest.CaptureFixture[str]
    ):
        results = {"model-a": {"coding": 2.0, "math": 4.0}}
        print_table(results, columns=["coding", "math"], metric_type="accept_length")
        out = capsys.readouterr().out
        assert "model-a" in out
        assert "Acceptance Length" in out  # title_map entry rendered

    def test_print_table_rich_branch_handles_missing_values(
        self, capsys: pytest.CaptureFixture[str]
    ):
        # Missing cells render as "-" and are excluded from the Overall mean.
        results = {"model-a": {"coding": 2.0}}
        print_table(results, columns=["coding", "math"], metric_type="throughput")
        out = capsys.readouterr().out
        assert "Throughput" in out
        assert "-" in out  # missing math cell

    def test_print_table_fallback_when_rich_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        # Force `from rich.console import Console` to raise ImportError so we
        # exercise the plain-text branch (lines 334-351).
        import sys

        monkeypatch.setitem(sys.modules, "rich", None)
        monkeypatch.setitem(sys.modules, "rich.console", None)
        monkeypatch.setitem(sys.modules, "rich.table", None)

        results = {"model-a": {"coding": 2.0, "math": 4.0}}
        print_table(results, columns=["coding", "math"], metric_type="accept_length")
        out = capsys.readouterr().out
        # Plain-text output: header row + separator row + data row
        assert "Model" in out
        assert "coding" in out
        assert "model-a" in out
        assert "2.00" in out
        assert "4.00" in out
        # Plain fallback uses right-justified columns separated by two spaces
        assert "---" in out  # separator row dashes


class TestWriteCsv:
    def test_write_csv_includes_header_and_overall_mean(self, tmp_path: Path):
        results = {"model-a": {"coding": 2.0, "math": 4.0}}
        output = tmp_path / "out.csv"

        write_csv(results, columns=["coding", "math"], output=output)

        lines = output.read_text().splitlines()
        assert lines[0] == "Model,coding,math,Overall"
        assert lines[1] == "model-a,2.00,4.00,3.00"

    def test_write_csv_missing_values_leave_blanks_and_exclude_from_mean(
        self, tmp_path: Path
    ):
        results = {"model-a": {"coding": 2.0}}  # math absent
        output = tmp_path / "out.csv"

        write_csv(results, columns=["coding", "math"], output=output)

        lines = output.read_text().splitlines()
        # Missing cell is blank; overall averages only the present values.
        assert lines[1] == "model-a,2.00,,2.00"


class TestGenerateReport:
    def test_generate_report_no_run_dirs_raises(self, tmp_path: Path):
        with pytest.raises(SpeedBenchReportError, match="no aiperf run directories"):
            generate_report([tmp_path / "does_not_exist"])

    def test_generate_report_no_extractable_results_raises(self, tmp_path: Path):
        # Profile present but dataset isn't a speed_bench_* entry -> zero results.
        _write_run_dir(tmp_path, "run_other", _profile(dataset="sharegpt"))
        with pytest.raises(SpeedBenchReportError, match="no SPEED-Bench results"):
            generate_report([tmp_path])

    def test_generate_report_writes_csv_end_to_end(self, tmp_path: Path):
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_length", {"avg": 2.5}),
        )
        output = tmp_path / "report.csv"

        generate_report([tmp_path], output=output, output_format="csv")

        assert output.exists()
        assert "m1,2.50" in output.read_text()

    def test_generate_report_table_format_prints_without_writing_csv(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        # output_format="table" should invoke print_table and skip write_csv.
        _write_run_dir(
            tmp_path,
            "run_coding",
            _profile(dataset="speed_bench_coding", model="m1"),
            server_metrics=_server_metric("sglang:spec_accept_length", {"avg": 2.5}),
        )
        output = tmp_path / "report.csv"

        generate_report([tmp_path], output=output, output_format="table")

        assert not output.exists()
        out = capsys.readouterr().out
        assert "m1" in out
        assert "2.50" in out
