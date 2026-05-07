# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.exporters.outputs_json_exporter import OutputsJsonExporter


def _make_record(
    session_num: int,
    benchmark_phase: str = "profiling",
    x_request_id: str = "req-1",
    conversation_id: str = "conv-1",
    turn_index: int = 0,
    request_start_ns: int = 1000000000,
    request_end_ns: int = 2000000000,
    output_token_count: int = 42,
    request_latency: float = 1000.0,
    error: dict | None = None,
) -> dict:
    """Build a MetricRecordInfo dict suitable for JSONL serialization."""
    return {
        "metadata": {
            "session_num": session_num,
            "x_request_id": x_request_id,
            "x_correlation_id": None,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "credit_issued_ns": None,
            "request_start_ns": request_start_ns,
            "request_ack_ns": None,
            "request_end_ns": request_end_ns,
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
        "error": error,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL to the given path."""
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")


def _make_exporter(tmp_path: Path) -> OutputsJsonExporter:
    """Create an OutputsJsonExporter with mocked config pointing to tmp_path."""
    config = MagicMock()
    config.user_config.output.outputs_json_file = tmp_path / "outputs.json"
    config.user_config.output.profile_export_jsonl_file = (
        tmp_path / "profile_export.jsonl"
    )
    return OutputsJsonExporter(config)


class TestOutputsJsonExporter:
    @pytest.mark.asyncio
    async def test_export_produces_valid_outputs_json(self, tmp_path: Path) -> None:
        """Warmup records are filtered; only profiling records appear in outputs.json."""
        records = [
            _make_record(session_num=0, benchmark_phase="warmup"),
            _make_record(session_num=1, benchmark_phase="profiling"),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        outputs_file = tmp_path / "outputs.json"
        assert outputs_file.exists()

        data = orjson.loads(outputs_file.read_bytes())
        assert data["schema_version"] == "1.0"
        assert len(data["data"]) == 1
        assert data["data"][0]["session_num"] == 1

    @pytest.mark.asyncio
    async def test_export_missing_jsonl_skips_gracefully(self, tmp_path: Path) -> None:
        """When the JSONL file does not exist, export completes without error and no outputs.json is produced."""
        exporter = _make_exporter(tmp_path)
        await exporter.export()

        outputs_file = tmp_path / "outputs.json"
        assert not outputs_file.exists()

    @pytest.mark.asyncio
    async def test_export_sorts_by_session_num(self, tmp_path: Path) -> None:
        """Records in outputs.json are sorted by session_num ascending regardless of input order."""
        records = [
            _make_record(session_num=5),
            _make_record(session_num=2),
            _make_record(session_num=9),
            _make_record(session_num=1),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        session_nums = [r["session_num"] for r in data["data"]]
        assert session_nums == [1, 2, 5, 9]

    @pytest.mark.asyncio
    async def test_export_extracts_correct_fields(self, tmp_path: Path) -> None:
        """Verify all expected fields are correctly extracted from a MetricRecordInfo record."""
        records = [
            _make_record(
                session_num=3,
                x_request_id="req-abc",
                conversation_id="conv-xyz",
                turn_index=2,
                request_start_ns=5000000000,
                request_end_ns=6000000000,
                output_token_count=100,
                request_latency=500.0,
                error={"error_type": "timeout", "message": "request timed out"},
            ),
        ]
        _write_jsonl(tmp_path / "profile_export.jsonl", records)

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 1

        entry = data["data"][0]
        assert entry["session_num"] == 3
        assert entry["conversation_id"] == "conv-xyz"
        assert entry["turn_index"] == 2
        assert entry["x_request_id"] == "req-abc"
        assert entry["request_start_ns"] == 5000000000
        assert entry["request_end_ns"] == 6000000000
        assert entry["metrics"]["output_token_count"] == 100
        assert entry["metrics"]["request_latency"] == 500.0
        assert entry["error"] is not None
        assert entry["error"]["error_type"] == "timeout"
        assert entry["error"]["message"] == "request timed out"
