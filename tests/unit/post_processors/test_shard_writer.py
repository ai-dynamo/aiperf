# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared per-processor shard machinery (ShardWriterMixin / ShardAggregatorMixin)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.artifacts import OutputDefaults
from aiperf.post_processors.shard_writer import ShardAggregatorMixin, ShardWriterMixin


@pytest.mark.parametrize(
    "service_id,expected",
    [
        param("processor/1", "processor_1", id="slash"),
        param("proc:2", "proc_2", id="colon"),
        param("a b c", "a_b_c", id="spaces"),
        param(None, "processor", id="none-defaults"),
        param("", "processor", id="empty-defaults"),
    ],
)  # fmt: skip
def test_sanitize_shard_id(service_id: str | None, expected: str) -> None:
    assert ShardWriterMixin.sanitize_shard_id(service_id) == expected


def test_shard_output_file_builds_path_and_dir(tmp_path: Path) -> None:
    mixin = ShardWriterMixin()
    out = mixin.shard_output_file(
        tmp_path,
        OutputDefaults.RECORDS_SHARDS_FOLDER,
        prefix="records",
        ext="jsonl",
        service_id="rp/3",
    )
    assert out == tmp_path / "records_shards" / "records_rp_3.jsonl"
    # Directory is created eagerly; the file itself is not.
    assert out.parent.is_dir()
    assert not out.exists()


class _Aggregator(ShardAggregatorMixin):
    """Minimal concrete aggregator exposing the protected concat helper."""


@pytest.mark.asyncio
async def test_concat_shards_merges_all_and_removes_dir(tmp_path: Path) -> None:
    shard_dir = tmp_path / "records_shards"
    shard_dir.mkdir()
    # Three per-processor shards with differing line counts.
    (shard_dir / "records_0.jsonl").write_bytes(b'{"i": 0}\n{"i": 1}\n')
    (shard_dir / "records_1.jsonl").write_bytes(b'{"i": 2}\n')
    (shard_dir / "records_2.jsonl").write_bytes(b'{"i": 3}\n{"i": 4}\n{"i": 5}\n')
    out = tmp_path / "profile_export.jsonl"

    count = await _Aggregator()._concat_shards(shard_dir, "records_*.jsonl", out)

    assert count == 6
    lines = out.read_text().splitlines()
    assert len(lines) == 6
    # Sorted-order concatenation preserves shard 0's records first.
    assert lines[0] == '{"i": 0}'
    assert lines[-1] == '{"i": 5}'
    # Shards consumed and the (now empty) dir removed.
    assert not shard_dir.exists()


@pytest.mark.asyncio
async def test_concat_shards_no_shards_is_noop(tmp_path: Path) -> None:
    shard_dir = tmp_path / "records_shards"
    shard_dir.mkdir()
    out = tmp_path / "profile_export.jsonl"

    count = await _Aggregator()._concat_shards(shard_dir, "records_*.jsonl", out)

    assert count == 0
    assert not out.exists()


@pytest.mark.asyncio
async def test_concat_shards_csv_header_deduped(tmp_path: Path) -> None:
    shard_dir = tmp_path / "records_shards"
    shard_dir.mkdir()
    (shard_dir / "records_0.csv").write_bytes(b"a,b\n1,2\n3,4\n")
    (shard_dir / "records_1.csv").write_bytes(b"a,b\n5,6\n")
    out = tmp_path / "profile_export_records.csv"

    count = await _Aggregator()._concat_shards(
        shard_dir, "records_*.csv", out, header_from_first=True
    )

    # Header counted once (not as a data row), and the second shard's header dropped.
    assert count == 3
    assert out.read_text().splitlines() == ["a,b", "1,2", "3,4", "5,6"]
