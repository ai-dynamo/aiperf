# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The artifact bundle must stream, not materialize.

It used to build the whole zip with BytesIO.getvalue() before yielding a
single byte, fully decompressing every .zst member into memory on the way and
capping nothing: one GET of a run with multi-GB raw records allocated roughly
twice that in the operator pod and OOMKilled it.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import zstandard
from aiperf.operator.routers.results_files_io import (
    CHUNK_SIZE,
    _stream_artifact_bundle,
    _stream_job_bundle,
)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    (tmp_path / "summary.json").write_bytes(b'{"x":1}')
    (tmp_path / "big.txt").write_bytes(b"y" * (CHUNK_SIZE * 5))
    (tmp_path / "records.jsonl.zst").write_bytes(
        zstandard.ZstdCompressor().compress(b'{"line":1}\n' * 5000)
    )
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "ckpt.parquet").write_bytes(b"parquet-bytes")
    return tmp_path


@pytest.mark.asyncio
async def test_bundle_is_emitted_in_bounded_chunks(artifact_dir: Path) -> None:
    chunks = [c async for c in _stream_artifact_bundle(artifact_dir)]
    assert len(chunks) > 1, "a multi-chunk payload arrived as one buffer"
    assert all(0 < len(c) <= CHUNK_SIZE for c in chunks)


@pytest.mark.asyncio
async def test_bundle_round_trips(artifact_dir: Path) -> None:
    blob = b"".join([c async for c in _stream_artifact_bundle(artifact_dir)])
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert zf.testzip() is None
    assert sorted(zf.namelist()) == ["big.txt", "records.jsonl", "summary.json"]
    assert zf.read("summary.json") == b'{"x":1}'
    assert len(zf.read("big.txt")) == CHUNK_SIZE * 5
    # .zst members are decompressed into the archive under their bare name.
    assert zf.read("records.jsonl") == b'{"line":1}\n' * 5000


@pytest.mark.asyncio
async def test_job_bundle_includes_checkpoints(artifact_dir: Path) -> None:
    blob = b"".join([c async for c in _stream_job_bundle(artifact_dir)])
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert "checkpoints/ckpt.parquet" in zf.namelist()
    assert zf.read("checkpoints/ckpt.parquet") == b"parquet-bytes"


@pytest.mark.asyncio
async def test_empty_dir_yields_a_valid_empty_zip(tmp_path: Path) -> None:
    blob = b"".join([c async for c in _stream_artifact_bundle(tmp_path)])
    assert zipfile.ZipFile(io.BytesIO(blob)).namelist() == []
