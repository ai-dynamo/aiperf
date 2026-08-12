# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.outputs_json_exporter import OutputsJsonExporter


def _make_fragment(
    session_num: int,
    turn_index: int = 0,
    conversation_id: str = "conv-1",
    x_request_id: str = "req-1",
    response_text: str | None = "Hello, world!",
    request_start_ns: int = 1000000000,
    request_end_ns: int = 2000000000,
    metrics: dict | None = None,
) -> dict:
    """Build an output fragment dict suitable for JSONL serialization.

    Fragments now carry their own metrics (captured in display units by the
    record processor); the exporter no longer joins against profile_export.jsonl.
    """
    return {
        "session_num": session_num,
        "turn_index": turn_index,
        "conversation_id": conversation_id,
        "x_request_id": x_request_id,
        "response_text": response_text,
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
        "metrics": metrics if metrics is not None else {},
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
    config.cfg.artifacts.export_outputs_json = True
    config.cfg.artifacts.outputs_json_file = tmp_path / "outputs.json"
    config.cfg.artifacts.artifact_directory = tmp_path
    return OutputsJsonExporter(config)


class TestOutputsJsonExporter:
    def test_disabled_when_flag_not_set(self, tmp_path: Path) -> None:
        """Exporter raises DataExporterDisabled when export_outputs_json is False."""
        config = MagicMock()
        config.cfg.artifacts.export_outputs_json = False
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
    async def test_export_skips_unparseable_fragment_line(self, tmp_path: Path) -> None:
        """A corrupt line (e.g. a partial write from a crashed processor) is skipped
        rather than aborting the whole export."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        frag_file = fragments_dir / "output_fragments_proc1.jsonl"
        good = orjson.dumps(_make_fragment(session_num=1, response_text="good"))
        frag_file.write_bytes(good + b"\n" + b"THIS IS NOT JSON {{{\n")

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 1
        assert data["data"][0]["response_text"] == "good"

    @pytest.mark.asyncio
    async def test_export_skips_non_object_fragment_line(self, tmp_path: Path) -> None:
        """A syntactically-valid but non-object line (e.g. a bare literal) is skipped
        rather than crashing the export when treated as a fragment dict."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        frag_file = fragments_dir / "output_fragments_proc1.jsonl"
        good = orjson.dumps(_make_fragment(session_num=1, response_text="good"))
        frag_file.write_bytes(good + b"\n" + b"123\n" + b'"stray"\n' + b"[]\n")

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 1
        assert data["data"][0]["response_text"] == "good"

    @pytest.mark.asyncio
    async def test_export_emits_fragment_metrics(self, tmp_path: Path) -> None:
        """Metrics carried on each fragment are emitted directly into outputs.json."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [
            _make_fragment(
                session_num=1,
                response_text="Hello",
                metrics={"output_token_count": 10, "request_latency": 500.0},
            ),
            _make_fragment(
                session_num=2,
                response_text="World",
                metrics={"output_token_count": 20, "request_latency": 800.0},
            ),
        ]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

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
    async def test_export_metrics_without_records_jsonl(self, tmp_path: Path) -> None:
        """Regression (F2): metrics come from the fragment, so outputs.json is fully
        populated even when no profile_export.jsonl exists (e.g. --export-level summary
        or a YAML records: false)."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragments = [
            _make_fragment(
                session_num=1,
                response_text="test",
                metrics={"request_latency": 123.0, "output_token_count": 7},
            )
        ]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", fragments)

        # Deliberately no profile_export.jsonl written.
        assert not (tmp_path / "profile_export.jsonl").exists()

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert len(data["data"]) == 1
        assert data["data"][0]["metrics"] == {
            "request_latency": 123.0,
            "output_token_count": 7,
        }
        assert data["data"][0]["response_text"] == "test"

    @pytest.mark.asyncio
    async def test_export_empty_metrics_when_fragment_has_none(
        self, tmp_path: Path
    ) -> None:
        """A fragment with no captured metrics yields an empty metrics object."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [_make_fragment(session_num=1, response_text="test", metrics={})],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert data["data"][0]["metrics"] == {}

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
