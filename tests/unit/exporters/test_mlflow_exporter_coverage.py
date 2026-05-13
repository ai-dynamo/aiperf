# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for MLflowDataExporter error paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.config import OutputConfig, ServiceConfig
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.mlflow_data_exporter import MLflowDataExporter
from aiperf.exporters.mlflow_metadata import normalize_mlflow_uri


def _make_config(
    tmp_path: Path,
    *,
    tracking_uri: str = "http://localhost:5000",
    benchmark_id: str = "test-bench-123",
) -> ExporterConfig:
    user_config = MagicMock()
    user_config.mlflow_enabled = True
    user_config.mlflow_tracking_uri = tracking_uri
    user_config.mlflow_experiment = "test-exp"
    user_config.mlflow_run_name = None
    user_config.mlflow_parent_run_id = None
    user_config.mlflow_tags_dict = {}
    user_config.mlflow_resolved_artifact_globs = ["*.json", "*.csv"]
    user_config.output = MagicMock(spec=OutputConfig)
    user_config.output.artifact_directory = tmp_path
    user_config.benchmark_id = benchmark_id
    user_config.endpoint.type = "chat"
    user_config.endpoint.model_names = ["mock-model"]
    user_config.endpoint.urls = ["http://localhost:8000"]
    user_config.loadgen.concurrency = 4
    user_config.loadgen.request_rate = None
    user_config.loadgen.request_count = 32
    user_config.loadgen.benchmark_duration = None
    user_config.timing_mode = "fixed_schedule"
    user_config.cli_command = "aiperf profile ..."

    results = MagicMock(spec=ProfileResults)
    results.records = []
    results.completed = 32
    results.total_expected = 32
    results.was_cancelled = False

    return ExporterConfig(
        user_config=user_config,
        results=results,
        service_config=ServiceConfig(),
        telemetry_results=None,
    )


class TestLoadExistingMetadata:
    """Cover _load_existing_metadata edge cases."""

    def test_no_metadata_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._load_existing_metadata()
        assert result == {}

    def test_malformed_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        metadata_file = tmp_path / "mlflow_export.json"
        metadata_file.write_bytes(b"not valid json{{{")
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._load_existing_metadata()
        assert result == {}

    def test_non_dict_payload(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        metadata_file = tmp_path / "mlflow_export.json"
        metadata_file.write_bytes(orjson.dumps(["a", "list"]))
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._load_existing_metadata()
        assert result == {}

    def test_valid_metadata(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        metadata_file = tmp_path / "mlflow_export.json"
        metadata_file.write_bytes(
            orjson.dumps({"tracking_uri": "http://x", "run_id": "abc"})
        )
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._load_existing_metadata()
        assert result == {"tracking_uri": "http://x", "run_id": "abc"}


class TestResolveLiveStreamingRunId:
    """Cover _resolve_live_streaming_run_id branches."""

    def test_not_live_streaming(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id({"live_streaming": False})
        assert result is None

    def test_missing_run_id(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {"live_streaming": True, "tracking_uri": "http://localhost:5000"}
        )
        assert result is None

    def test_tracking_uri_mismatch(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {
                "live_streaming": True,
                "run_id": "abc",
                "tracking_uri": "http://DIFFERENT:5000",
                "benchmark_id": "test-bench-123",
            }
        )
        assert result is None

    def test_benchmark_id_mismatch(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {
                "live_streaming": True,
                "run_id": "abc",
                "tracking_uri": "http://localhost:5000",
                "benchmark_id": "DIFFERENT-ID",
            }
        )
        assert result is None

    def test_successful_reuse(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {
                "live_streaming": True,
                "run_id": "abc123",
                "tracking_uri": "http://localhost:5000",
                "benchmark_id": "test-bench-123",
            }
        )
        assert result == "abc123"

    def test_successful_reuse_with_redacted_on_disk_credential_in_memory(
        self, tmp_path: Path
    ) -> None:
        """Regression: on-disk URI is redacted, in-memory URI carries credentials.

        mlflow_export.json is uploaded as a run artifact, so _write_export_metadata
        redacts userinfo from the tracking URI before persisting. The reuse check
        must redact the in-memory URI too, otherwise same-backend reuse breaks
        for credentialed backends like postgresql://user:secret@db/mlflow.
        """
        config = _make_config(
            tmp_path, tracking_uri="postgresql://user:secret@db:5432/mlflow"
        )
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {
                "live_streaming": True,
                "run_id": "abc123",
                "tracking_uri": "postgresql://<redacted>@db:5432/mlflow",
                "benchmark_id": "test-bench-123",
            }
        )
        assert result == "abc123"

    def test_tracking_uri_mismatch_still_rejected_after_redaction(
        self, tmp_path: Path
    ) -> None:
        """Redaction must not collapse different hosts into a match."""
        config = _make_config(
            tmp_path, tracking_uri="postgresql://user:secret@db1:5432/mlflow"
        )
        exporter = MLflowDataExporter(exporter_config=config)
        result = exporter._resolve_live_streaming_run_id(
            {
                "live_streaming": True,
                "run_id": "abc123",
                "tracking_uri": "postgresql://<redacted>@db2:5432/mlflow",
                "benchmark_id": "test-bench-123",
            }
        )
        assert result is None


class TestNormalizeUri:
    """Regression: normalize_mlflow_uri must not collapse case-distinct paths."""

    def test_different_path_case_compare_unequal(self) -> None:
        """On case-sensitive filesystems (Linux), /tmp/MLRuns != /tmp/mlruns."""
        assert normalize_mlflow_uri("file:///tmp/MLRuns") != normalize_mlflow_uri(
            "file:///tmp/mlruns"
        )

    def test_scheme_and_host_are_case_insensitive(self) -> None:
        assert normalize_mlflow_uri(
            "HTTP://Host.Com:5000/path"
        ) == normalize_mlflow_uri("http://host.com:5000/path")

    @pytest.mark.parametrize(
        "upper,lower",
        [
            param("FILE:///tmp/mlruns", "file:///tmp/mlruns", id="file-scheme"),
            param(
                "SQLITE:///tmp/mlflow.db",
                "sqlite:///tmp/mlflow.db",
                id="sqlite-scheme",
            ),
        ],
    )  # fmt: skip
    def test_scheme_case_insensitive_when_netloc_empty(self, upper: str, lower: str):
        """Regression: scheme must still be lowercased for URIs with empty
        netloc (file:///, sqlite:///). RFC 3986 §3.1 says scheme is case-
        insensitive; the early-return guard previously skipped lowercasing.
        """
        assert normalize_mlflow_uri(upper) == normalize_mlflow_uri(lower)

    def test_trailing_slash_stripped(self) -> None:
        assert normalize_mlflow_uri("http://host:5000/path/") == normalize_mlflow_uri(
            "http://host:5000/path"
        )

    def test_query_case_preserved(self) -> None:
        assert normalize_mlflow_uri(
            "http://host:5000/?MixedCase=Value"
        ) != normalize_mlflow_uri("http://host:5000/?mixedcase=value")

    @pytest.mark.parametrize("uri", [None, "", "   "])
    def test_empty_inputs(self, uri: str | None) -> None:
        assert normalize_mlflow_uri(uri) == ""


class TestDisabledExporter:
    """Cover DataExporterDisabled paths."""

    def test_disabled_when_mlflow_not_enabled(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.user_config.mlflow_enabled = False
        with pytest.raises(DataExporterDisabled):
            MLflowDataExporter(exporter_config=config)

    def test_disabled_when_no_results(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.results = None
        with pytest.raises(DataExporterDisabled):
            MLflowDataExporter(exporter_config=config)
