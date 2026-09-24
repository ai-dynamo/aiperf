# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest
from aiofiles.threadpool.binary import AsyncBufferedIOBase

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters import outputs_json_exporter
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
    benchmark_phase: str = "profiling",
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
        "benchmark_phase": benchmark_phase,
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
    async def test_export_empty_fragments_dir_skips(self, tmp_path: Path) -> None:
        (tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER).mkdir()
        await _make_exporter(tmp_path).export()
        assert not (tmp_path / "outputs.json").exists()
        assert not list(tmp_path.glob(".outputs-json-*"))

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
        assert data["schema_version"] == "1.1"
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


class TestOutputsJsonWarmupPartition:
    """Warmup responses are exported, but kept out of `data`."""

    @pytest.mark.asyncio
    async def test_warmup_fragments_go_to_warmup_array(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [
                _make_fragment(session_num=1, response_text="profiled"),
                _make_fragment(
                    session_num=0,
                    response_text="warmed up",
                    benchmark_phase="warmup",
                ),
            ],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert [r["response_text"] for r in data["data"]] == ["profiled"]
        assert [r["response_text"] for r in data["warmup"]] == ["warmed up"]

    @pytest.mark.asyncio
    async def test_entries_carry_benchmark_phase(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [
                _make_fragment(session_num=1),
                _make_fragment(session_num=0, benchmark_phase="warmup"),
            ],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert data["data"][0]["benchmark_phase"] == "profiling"
        assert data["warmup"][0]["benchmark_phase"] == "warmup"

    @pytest.mark.asyncio
    async def test_warmup_array_present_and_empty_when_no_warmup(
        self, tmp_path: Path
    ) -> None:
        """`warmup` is always emitted so consumers can index it unconditionally."""
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [_make_fragment(session_num=1)],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert data["warmup"] == []
        assert len(data["data"]) == 1

    @pytest.mark.asyncio
    async def test_fragment_without_phase_treated_as_profiling(
        self, tmp_path: Path
    ) -> None:
        """Only a known-warmup phase may leave `data`.

        Pins the direction of the partition branch: inverting it to test for
        PROFILING would silently move records with a missing or newly-added
        phase into `warmup`, dropping them from consumers' denominators.
        """
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        fragment = _make_fragment(session_num=1, response_text="legacy")
        del fragment["benchmark_phase"]
        _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", [fragment])

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert [r["response_text"] for r in data["data"]] == ["legacy"]
        assert data["warmup"] == []

    @pytest.mark.asyncio
    async def test_warmup_array_sorted_independently(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir(parents=True)

        _write_jsonl(
            fragments_dir / "output_fragments_proc1.jsonl",
            [
                _make_fragment(session_num=9, benchmark_phase="warmup"),
                _make_fragment(session_num=2, benchmark_phase="warmup"),
                _make_fragment(session_num=5),
                _make_fragment(session_num=1),
            ],
        )

        exporter = _make_exporter(tmp_path)
        await exporter.export()

        data = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert [r["session_num"] for r in data["data"]] == [1, 5]
        assert [r["session_num"] for r in data["warmup"]] == [2, 9]


class TestBoundedOutputsExport:
    @pytest.mark.asyncio
    async def test_many_runs_preserve_order_and_partition(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        rows = []
        for index in range(31):
            row = _make_fragment(
                session_num=index % 4,
                turn_index=0 if index % 3 else -1,
                response_text=f"original-{index}",
                benchmark_phase="warmup" if index % 5 == 0 else "profiling",
            )
            if index == 7:
                del row["benchmark_phase"]
            if index == 8:
                row["benchmark_phase"] = "future"
            rows.append(row)
        _write_jsonl(fragments_dir / "output_fragments_a.jsonl", rows[:15])
        _write_jsonl(fragments_dir / "output_fragments_b.jsonl", rows[15:])
        traversal = [
            orjson.loads(line)
            for entry in os.scandir(fragments_dir)
            if entry.name.startswith("output_fragments_")
            and entry.name.endswith(".jsonl")
            for line in Path(entry.path).read_bytes().splitlines()
        ]
        exporter = _make_exporter(tmp_path)
        exporter.CHUNK_RECORDS = 2
        exporter.CHUNK_BYTES = 100
        exporter.MERGE_FAN_IN = 3
        await exporter.export()

        document = orjson.loads((tmp_path / "outputs.json").read_bytes())
        assert document["schema_version"] == "1.1"
        assert sorted(
            row["response_text"] for row in document["data"] + document["warmup"]
        ) == sorted(row["response_text"] for row in rows)
        for name in ("data", "warmup"):
            expected = [
                row["response_text"]
                for row in sorted(
                    (
                        r
                        for r in traversal
                        if (r.get("benchmark_phase") == "warmup") == (name == "warmup")
                    ),
                    key=lambda r: (r["session_num"], r.get("turn_index") or 0),
                )
            ]
            assert [row["response_text"] for row in document[name]] == expected
        assert document["data"][0]["benchmark_phase"] in {"profiling", "future", None}
        assert not list(tmp_path.glob(".outputs-json-*"))

    @pytest.mark.asyncio
    async def test_empty_file_exports_empty_arrays(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        fragments_dir.mkdir()
        (fragments_dir / "output_fragments_empty.jsonl").touch()
        await _make_exporter(tmp_path).export()
        assert orjson.loads((tmp_path / "outputs.json").read_bytes()) == {
            "schema_version": "1.1",
            "data": [],
            "warmup": [],
        }
        assert not (fragments_dir / "output_fragments_empty.jsonl").exists()

    @pytest.mark.asyncio
    async def test_fragment_scan_consumes_directory_in_bounded_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        for index in range(130):
            _write_jsonl(
                fragments_dir / f"output_fragments_{index:03}.jsonl",
                [_make_fragment(session_num=index)],
            )
        exporter = _make_exporter(tmp_path)
        actual_scandir = os.scandir
        scanned = 0
        scanned_at_first_read: list[int] = []

        class TrackedEntries:
            def __init__(self, entries: Iterator[os.DirEntry[str]]) -> None:
                self._entries = entries

            def __iter__(self) -> "TrackedEntries":
                return self

            def __next__(self) -> os.DirEntry[str]:
                nonlocal scanned
                scanned += 1
                return next(self._entries)

            def close(self) -> None:
                self._entries.close()

        def tracked_scandir(path: Path) -> Iterator[os.DirEntry[str]]:
            entries = actual_scandir(path)
            if path == fragments_dir:
                return TrackedEntries(entries)
            return entries

        original_lines = exporter._lines

        async def tracked_lines(path: Path) -> AsyncIterator[bytes]:
            if not scanned_at_first_read:
                scanned_at_first_read.append(scanned)
            async for line in original_lines(path):
                yield line

        monkeypatch.setattr(os, "scandir", tracked_scandir)
        monkeypatch.setattr(exporter, "_lines", tracked_lines)
        await exporter.export()
        assert scanned_at_first_read == [exporter.FRAGMENT_SCAN_BATCH]
        assert scanned >= 130
        assert (
            len(orjson.loads((tmp_path / "outputs.json").read_bytes())["data"]) == 130
        )

    @pytest.mark.asyncio
    async def test_cleanup_preserves_fragment_created_after_ingestion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        consumed = fragments_dir / "output_fragments_old.jsonl"
        late = fragments_dir / "output_fragments_late.jsonl"
        _write_jsonl(consumed, [_make_fragment(session_num=1, response_text="old")])
        exporter = _make_exporter(tmp_path)
        original_write = exporter._write_document

        async def write_then_add(staged: Path, run: Path | None) -> None:
            await original_write(staged, run)
            _write_jsonl(late, [_make_fragment(session_num=2, response_text="late")])

        monkeypatch.setattr(exporter, "_write_document", write_then_add)
        await exporter.export()
        assert not consumed.exists()
        assert late.exists()
        assert [
            row["response_text"]
            for row in orjson.loads((tmp_path / "outputs.json").read_bytes())["data"]
        ] == ["old"]

    @pytest.mark.asyncio
    async def test_large_unicode_record(self, tmp_path: Path) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        row = _make_fragment(
            session_num=1,
            response_text="雪🎵" * 100_000,
            metrics={"request_latency": None, "nested": [None, 3]},
        )
        _write_jsonl(fragments_dir / "output_fragments_a.jsonl", [row])
        exporter = _make_exporter(tmp_path)
        exporter.CHUNK_BYTES = 100
        await exporter.export()
        actual = orjson.loads((tmp_path / "outputs.json").read_bytes())["data"][0]
        assert actual["response_text"] == row["response_text"]
        assert actual["metrics"] == {"request_latency": None, "nested": [None, 3]}

    def test_entry_scrubs_non_finite_metrics(self) -> None:
        entry = OutputsJsonExporter._entry(
            {
                "session_num": 1,
                "metrics": {"request_latency": float("nan"), "nested": [float("inf")]},
            }
        )
        assert entry["metrics"] == {"request_latency": None, "nested": [None]}

    @pytest.mark.asyncio
    async def test_replace_error_retains_sources_and_old_final(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        source = fragments_dir / "output_fragments_a.jsonl"
        _write_jsonl(source, [_make_fragment(session_num=1)])
        final = tmp_path / "outputs.json"
        final.write_bytes(b"old final")

        def fail_replace(_source: Path, _target: Path) -> None:
            raise OSError("replace denied")

        monkeypatch.setattr(outputs_json_exporter.os, "replace", fail_replace)
        with pytest.raises(OSError, match="replace denied"):
            await _make_exporter(tmp_path).export()
        assert source.exists()
        assert final.read_bytes() == b"old final"
        assert not list(tmp_path.glob(".outputs-json-*"))

    @pytest.mark.asyncio
    async def test_merge_error_retains_sources_and_old_final(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        source = fragments_dir / "output_fragments_a.jsonl"
        _write_jsonl(source, [_make_fragment(session_num=i) for i in range(5)])
        final = tmp_path / "outputs.json"
        final.write_bytes(b"old final")
        exporter = _make_exporter(tmp_path)
        exporter.CHUNK_RECORDS = 1
        original = exporter._merge_group

        async def fail_after_merge(sources: list[Path], target: Path) -> None:
            await original(sources, target)
            raise OSError("merge failed")

        monkeypatch.setattr(exporter, "_merge_group", fail_after_merge)
        with pytest.raises(OSError, match="merge failed"):
            await exporter.export()
        assert source.exists()
        assert final.read_bytes() == b"old final"
        assert not list(tmp_path.glob(".outputs-json-*"))

    @pytest.mark.asyncio
    async def test_write_error_closes_file_before_scratch_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        source = fragments_dir / "output_fragments_a.jsonl"
        _write_jsonl(source, [_make_fragment(session_num=1)])
        final = tmp_path / "outputs.json"
        final.write_bytes(b"old final")
        opened: list[AsyncBufferedIOBase] = []
        original_write = AsyncBufferedIOBase.write

        async def fail_during_write(handle: AsyncBufferedIOBase, data: bytes) -> int:
            opened.append(handle)
            await original_write(handle, data[:7])
            raise OSError("write failed")

        monkeypatch.setattr(AsyncBufferedIOBase, "write", fail_during_write)
        with pytest.raises(OSError, match="write failed"):
            await _make_exporter(tmp_path).export()
        assert opened and opened[0].closed
        assert source.exists()
        assert final.read_bytes() == b"old final"
        assert not list(tmp_path.glob(".outputs-json-*"))

    @pytest.mark.asyncio
    async def test_cancelled_export_retains_sources_and_old_final(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
        source = fragments_dir / "output_fragments_a.jsonl"
        _write_jsonl(source, [_make_fragment(session_num=1)])
        final = tmp_path / "outputs.json"
        final.write_bytes(b"old final")
        entered = threading.Event()
        proceed = threading.Event()

        async def delayed_write(handle: AsyncBufferedIOBase, data: bytes) -> int:
            def write_and_wait() -> int:
                written = handle._file.write(data)
                entered.set()
                assert proceed.wait(timeout=2)
                return written

            return await asyncio.to_thread(write_and_wait)

        monkeypatch.setattr(AsyncBufferedIOBase, "write", delayed_write)
        exporter = _make_exporter(tmp_path)
        exporter.CHUNK_RECORDS = 1
        task = asyncio.create_task(exporter.export())
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert source.exists()
        assert final.read_bytes() == b"old final"
        assert not list(tmp_path.glob(".outputs-json-*"))
