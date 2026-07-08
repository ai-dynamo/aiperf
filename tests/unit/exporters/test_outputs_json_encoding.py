# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OutputsJsonExporter fragment reads must be UTF-8 regardless of the locale.

``OutputsJsonExporter._read_fragments`` / ``_build_metrics_map`` read
user-generated response text (which may contain any Unicode) from JSONL files.
A bare ``open`` decodes with the process locale, so under ``LANG=C`` /
Windows cp1252 a non-ASCII response body crashes the export with
``UnicodeDecodeError``. ``ascii_default_locale`` reproduces that decode without
mutating the real locale; the fix passes ``encoding="utf-8"`` explicitly.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.config.artifacts import OutputDefaults
from aiperf.exporters.outputs_json_exporter import OutputsJsonExporter

_NON_ASCII_TEXT = "café — 你好 🚀"


@pytest.fixture()
def ascii_default_locale(monkeypatch):
    """Force text ``open`` with no explicit encoding to use ascii (LANG=C)."""
    real_open = builtins.open

    def fake_open(file, mode="r", *args, encoding=None, **kwargs):
        if encoding is None and "b" not in mode:
            encoding = "ascii"
        return real_open(file, mode, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record) + b"\n")


def _make_exporter(tmp_path: Path) -> OutputsJsonExporter:
    config = MagicMock()
    config.cfg.artifacts.export_outputs_json = True
    config.cfg.artifacts.outputs_json_file = tmp_path / "outputs.json"
    config.cfg.artifacts.profile_export_jsonl_file = tmp_path / "profile_export.jsonl"
    config.cfg.artifacts.artifact_directory = tmp_path
    return OutputsJsonExporter(config)


@pytest.mark.asyncio
async def test_export_reads_non_ascii_fragments_under_c_locale(
    tmp_path, ascii_default_locale
):
    """Fragments containing non-ASCII response text must export under LANG=C."""
    fragments_dir = tmp_path / OutputDefaults.OUTPUT_FRAGMENTS_FOLDER
    fragment = {
        "session_num": 1,
        "turn_index": 0,
        "conversation_id": "conv-1",
        "x_request_id": "req-1",
        "response_text": _NON_ASCII_TEXT,
        "request_start_ns": 1_000_000_000,
        "request_end_ns": 2_000_000_000,
    }
    _write_jsonl(fragments_dir / "output_fragments_proc1.jsonl", [fragment])

    exporter = _make_exporter(tmp_path)
    await exporter.export()

    data = orjson.loads((tmp_path / "outputs.json").read_bytes())
    assert len(data["data"]) == 1
    assert data["data"][0]["response_text"] == _NON_ASCII_TEXT
