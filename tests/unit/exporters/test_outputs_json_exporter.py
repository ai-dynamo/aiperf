# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.outputs_json_exporter import OutputsJsonExporter


def _make_fragment(
    session_num: int,
    turn_index: int = 0,
    conversation_id: str = "conv-1",
    x_request_id: str = "req-1",
    response_text: str | None = "Hello, world!",
    request_start_ns: int = 1000000000,
    request_end_ns: int = 2000000000,
) -> dict:
    """Build an output fragment dict suitable for JSONL serialization."""
    return {
        "session_num": session_num,
        "turn_index": turn_index,
        "conversation_id": conversation_id,
        "x_request_id": x_request_id,
        "response_text": response_text,
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
    }


def _make_profile_record(
    session_num: int,
    turn_index: int = 0,
    benchmark_phase: str = "profiling",
    output_token_count: int = 42,
    request_latency: float = 1000.0,
) -> dict:
    """Build a MetricRecordInfo dict suitable for profile_export.jsonl."""
    return {
        "metadata": {
            "session_num": session_num,
            "x_request_id": "req-1",
            "x_correlation_id": None,
            "conversation_id": "conv-1",
            "turn_index": turn_index,
            "credit_issued_ns": None,
            "request_start_ns": 1000000000,
            "request_ack_ns": None,
            "request_end_ns": 2000000000,
            "worker_id": "worker-1",
            "record_processor_id": "proc-1",
            "benchmark_phase": benchmark_phase,
            "was_cancelled": False,
        },
        "metrics": {
            "output_token_count": {"value": output_token_count, "unit": "tokens"},
            "request_latency": {"value": request_latency, "unit": "ms"},
        },
        "trace_data": None,
        "error": None,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")


def _make_exporter(tmp_path: Path) -> OutputsJsonExporter:
    """Create an OutputsJsonExporter with mocked config pointing to tmp_path."""
    config = MagicMock()
    config.user_config.output.export_outputs_json = True
    config.user_config.output.outputs_json_file = tmp_path / "outputs.json"
    config.user_config.output.profile_export_jsonl_file = (
        tmp_path / "profile_export.jsonl"
    )
    config.user_config.output.artifact_directory = tmp_path
    return OutputsJsonExporter(config)


class TestOutputsJsonExporter:
    def test_disabled_when_flag_not_set(self, tmp_path: Path) -> None:
        """Exporter raises DataExporterDisabled when export_outputs_json is False."""
        config = MagicMock()
        config.user_config.output.export_outputs_json = False
        with pytest.raises(DataExporterDisabled):
            OutputsJsonExporter(config)

    @pytest.mark.asyncio
    async def test_export_no_fragments_skips(self, tmp_path: Path) -> None:
        """When no fragment files exist, export completes without error and no outputs.json is produced."""
        exporter = _make_exporter(tmp_path)
        await exporter.export()

        outputs_file = tmp_path / "outputs.json"
        assert not outputs_file.exists()

    @pytest.mark.asyncio
    async def test_export_merges_fragments_with_metrics(self, tmp_path: Path) -> None:
        """Fragments are merged with metrics from profile_export.jsonl."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [
            _make_fragment(session_num=1, response_text="Hello"),
            _make_fragment(session_num=2, response_text="World"),
        ]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        profile_records = [
            _make_profile_record(
                session_num=1, output_token_count=10, request_latency=500.0
            ),
            _make_profile_record(
                session_num=2, output_token_count=20, request_latency=800.0
            ),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", profile_records)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        outputs_file = tmp_path / "outputs.json"
        assert outputs_file.exists()

        data = orjson.loads(outputs_file.read_bytes())
        assert data["schema_version"] == "1.0"
        assert len(data["data"]) == 2

        entry1 = data["data"][0]
        assert entry1["session_num"] == 1
        assert entry1["response_text"] == "Hello"
        assert entry1["metrics"]["output_token_count"] == 10
        assert entry1["metrics"]["request_latency"] == 500.0

        entry2 = data["data"][1]
        assert entry2["session_num"] == 2
        assert entry2["response_text"] == "World"
        assert entry2["metrics"]["output_token_count"] == 20

    @pytest.mark.asyncio
    async def test_export_sorts_by_session_num(self, tmp_path: Path) -> None:
        """Records in outputs.json are sorted by session_num ascending."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [
            _make_fragment(session_num=5),
            _make_fragment(session_num=2),
            _make_fragment(session_num=9),
            _make_fragment(session_num=1),
        ]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        session_nums = [r["session_num"] for r in data["data"]]
        assert session_nums == [1, 2, 5, 9]

    @pytest.mark.asyncio
    async def test_export_handles_missing_profile_jsonl(self, tmp_path: Path) -> None:
        """When profile_export.jsonl is missing, metrics are empty but export succeeds."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [_make_fragment(session_num=1, response_text="test")]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 1
        assert data["data"][0]["metrics"] == {}
        assert data["data"][0]["response_text"] == "test"

    @pytest.mark.asyncio
    async def test_export_filters_warmup_from_metrics(self, tmp_path: Path) -> None:
        """Warmup records in profile_export.jsonl are not included in the metrics map."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [_make_fragment(session_num=1)]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        profile_records = [
            _make_profile_record(
                session_num=1, benchmark_phase="warmup", output_token_count=99
            ),
            _make_profile_record(
                session_num=1, benchmark_phase="profiling", output_token_count=42
            ),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", profile_records)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert data["data"][0]["metrics"]["output_token_count"] == 42

    @pytest.mark.asyncio
    async def test_export_cleans_up_fragments(self, tmp_path: Path) -> None:
        """Fragment files and directory are removed after export."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [_make_fragment(session_num=1)]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        assert not (fragments_dir / "output_fragments_proc1.jsonl").exists()
        assert not fragments_dir.exists()

    @pytest.mark.asyncio
    async def test_export_aggregates_multiple_fragment_files(
        self, tmp_path: Path
    ) -> None:
        """Multiple fragment files from different processors are aggregated."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [_make_fragment(session_num=1, response_text="from proc1")],
        )
        _write_jsonl(
            fragments_dir / "output_fragments_proc2.jsonl",
            [_make_fragment(session_num=2, response_text="from proc2")],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 2
        texts = {r["session_num"]: r["response_text"] for r in data["data"]}
        assert texts[1] == "from proc1"
        assert texts[2] == "from proc2"
