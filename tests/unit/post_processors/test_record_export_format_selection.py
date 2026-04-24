# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for record export format selection feature."""

from pathlib import Path

import pytest

from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.config import AIPerfConfig
from aiperf.plugin.enums import EndpointType
from aiperf.post_processors.record_export_csv_processor import (
    RecordExportCSVProcessor,
)
from aiperf.post_processors.record_export_results_processor import (
    RecordExportResultsProcessor,
)
from tests.unit.post_processors.conftest import _make_run

_MINIMAL_DATASETS = {
    "default": {
        "type": "synthetic",
        "entries": 100,
        "prompts": {"isl": 128, "osl": 64},
    }
}
_MINIMAL_PHASES = {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}}


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _make_config(tmp_artifact_dir: Path, **artifacts_overrides) -> AIPerfConfig:
    """Helper to create an AIPerfConfig with specified artifact settings."""
    artifacts = {
        "dir": str(tmp_artifact_dir),
        **artifacts_overrides,
    }
    return AIPerfConfig(
        models=["test-model"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": EndpointType.CHAT,
        },
        datasets=_MINIMAL_DATASETS,
        phases=_MINIMAL_PHASES,
        artifacts=artifacts,
    )


class TestRecordExportFormatConfig:
    """Test record export format configuration via artifacts.records."""

    def test_default_records_is_disabled(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir)
        assert config.artifacts.records is False

    def test_single_format_csv(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["csv"])
        assert isinstance(config.artifacts.records, list)
        assert "csv" in config.artifacts.records
        assert "jsonl" not in config.artifacts.records

    def test_single_format_jsonl(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["jsonl"])
        assert isinstance(config.artifacts.records, list)
        assert "jsonl" in config.artifacts.records

    def test_multiple_formats(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["jsonl", "csv"])
        assert isinstance(config.artifacts.records, list)
        assert "jsonl" in config.artifacts.records
        assert "csv" in config.artifacts.records

    def test_empty_list_raises_error(self, tmp_artifact_dir: Path):
        with pytest.raises(Exception, match="cannot be empty"):
            _make_config(tmp_artifact_dir, records=[])


class TestCSVProcessorFormatGating:
    """Test CSV processor respects format selection."""

    def test_csv_enabled_when_format_selected(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["csv"])
        processor = RecordExportCSVProcessor(
            service_id="test",
            run=_make_run(config),
        )
        assert processor is not None

    def test_csv_disabled_when_format_not_selected(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["jsonl"])
        with pytest.raises(PostProcessorDisabled, match="csv"):
            RecordExportCSVProcessor(
                service_id="test",
                run=_make_run(config),
            )

    def test_csv_disabled_when_records_false(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=False)
        with pytest.raises(PostProcessorDisabled):
            RecordExportCSVProcessor(
                service_id="test",
                run=_make_run(config),
            )


class TestJSONLProcessorFormatGating:
    """Test JSONL processor respects format selection."""

    def test_jsonl_enabled_when_format_selected(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["jsonl"])
        processor = RecordExportResultsProcessor(
            service_id="test",
            run=_make_run(config),
        )
        assert processor is not None

    def test_jsonl_disabled_when_format_not_selected(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["csv"])
        with pytest.raises(PostProcessorDisabled, match="jsonl"):
            RecordExportResultsProcessor(
                service_id="test",
                run=_make_run(config),
            )

    def test_jsonl_disabled_when_records_false(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=False)
        with pytest.raises(PostProcessorDisabled):
            RecordExportResultsProcessor(
                service_id="test",
                run=_make_run(config),
            )


class TestBothProcessorsFormatGating:
    """Test both processors with all formats selected."""

    def test_both_enabled_when_all_formats_selected(self, tmp_artifact_dir: Path):
        config = _make_config(tmp_artifact_dir, records=["jsonl", "csv"])
        csv_proc = RecordExportCSVProcessor(
            service_id="test-csv",
            run=_make_run(config),
        )
        jsonl_proc = RecordExportResultsProcessor(
            service_id="test-jsonl",
            run=_make_run(config),
        )
        assert csv_proc is not None
        assert jsonl_proc is not None
