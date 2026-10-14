# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
from contextlib import ExitStack
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiperf.common.record_files import iter_records
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.hf import HFSourceResolver
from tests.unit.dataset.remote.test_hf_resolver import SHA, FakeRepo

ROWS = [
    {"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0, 1]},
    {"timestamp": 5, "input_length": 12, "output_length": 3, "hash_ids": [0, 2]},
]


def parquet_bytes(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


class ParquetRepo(FakeRepo):
    """FakeRepo plus a record of snapshot allow_patterns."""

    def __init__(self, tmp_path: Path, files: dict[str, bytes]):
        super().__init__(tmp_path, files)
        self.allow_patterns: list[list[str]] = []

    def snapshot(self, repo_id, *, repo_type, revision, allow_patterns):
        self.allow_patterns.append(list(allow_patterns))
        return super().snapshot(
            repo_id,
            repo_type=repo_type,
            revision=revision,
            allow_patterns=allow_patterns,
        )


async def _resolve(repo: ParquetRepo, ref: str):
    with ExitStack() as stack:
        for p in repo.patches():
            stack.enter_context(p)
        resolver = HFSourceResolver()
        probe = await resolver.probe(RemoteRef(ref=ref))
        return probe, await resolver.materialize(probe)


async def _probe_error(repo: ParquetRepo, ref: str) -> RemoteSourceError:
    with pytest.raises(RemoteSourceError) as exc_info:
        await _resolve(repo, ref)
    return exc_info.value


@pytest.mark.asyncio
async def test_probe_push_to_hub_config_dir_selects_its_shards(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "README.md": b"# card",
        "mooncake/train-00001-of-00002.parquet": parquet_bytes(ROWS[1:]),
        "mooncake/train-00000-of-00002.parquet": parquet_bytes(ROWS[:1]),
    })  # fmt: skip
    probe, source = await _resolve(repo, "hf://datasets/org/repo/mooncake")
    shards = [
        "mooncake/train-00000-of-00002.parquet",
        "mooncake/train-00001-of-00002.parquet",
    ]
    assert probe.manifest is None and probe.data_is_dir
    assert probe.data_files == shards
    assert probe.data_relpath == "mooncake"
    assert probe.pinned_ref == f"hf://datasets/org/repo@{SHA}/mooncake"
    assert repo.allow_patterns == [shards]
    assert [f.path for f in source.files] == shards
    assert list(iter_records(source.local_path)) == ROWS


@pytest.mark.asyncio
async def test_probe_several_splits_selects_train(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "data/train-00000-of-00001.parquet": parquet_bytes(ROWS),
        "data/test-00000-of-00001.parquet": parquet_bytes(ROWS[:1]),
    })  # fmt: skip
    probe, source = await _resolve(repo, "hf://org/repo/data")
    assert probe.data_files == ["data/train-00000-of-00001.parquet"]
    assert list(iter_records(source.local_path)) == ROWS


@pytest.mark.asyncio
async def test_probe_no_train_split_lists_split_refs(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "data/test-00000-of-00001.parquet": parquet_bytes(ROWS),
        "data/validation-00000-of-00001.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    err = await _probe_error(repo, "hf://org/repo/data")
    assert err.kind == RemoteErrorKind.MANIFEST_INVALID
    assert "test, validation" in str(err)
    assert "hf://datasets/org/repo/data/test-00000-of-00001.parquet" in str(err)


@pytest.mark.asyncio
async def test_probe_repo_root_with_several_configs_lists_config_refs(
    tmp_path: Path,
) -> None:
    repo = ParquetRepo(tmp_path, {
        "conversation/train-00000-of-00001.parquet": parquet_bytes(ROWS),
        "mooncake/train-00000-of-00001.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    err = await _probe_error(repo, "hf://org/repo")
    assert err.kind == RemoteErrorKind.MANIFEST_INVALID
    assert "hf://datasets/org/repo/conversation" in str(err)
    assert "hf://datasets/org/repo/mooncake" in str(err)


@pytest.mark.asyncio
async def test_probe_convert_parquet_layout_descends_to_train(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "default/train/0000.parquet": parquet_bytes(ROWS[:1]),
        "default/train/0001.parquet": parquet_bytes(ROWS[1:]),
        "default/test/0000.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    probe, source = await _resolve(repo, "hf://org/repo")
    assert probe.data_files == [
        "default/train/0000.parquet",
        "default/train/0001.parquet",
    ]
    assert probe.pinned_ref == f"hf://datasets/org/repo@{SHA}/default/train"
    assert list(iter_records(source.local_path)) == ROWS


@pytest.mark.asyncio
async def test_probe_root_manifest_wins_over_parquet_layout(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "aiperf.json": orjson.dumps({"aiperf_manifest_version": 1, "format": "mooncake_trace", "path": "x/t.parquet"}),
        "x/t.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    probe, _ = await _resolve(repo, "hf://org/repo")
    assert probe.manifest is not None and probe.data_files is None
    assert probe.data_relpath == "x/t.parquet"


@pytest.mark.asyncio
async def test_probe_non_parquet_repo_without_manifest_keeps_manifest_error(
    tmp_path: Path,
) -> None:
    repo = ParquetRepo(tmp_path, {"data/t.jsonl": b"{}\n"})
    err = await _probe_error(repo, "hf://org/repo")
    assert err.kind == RemoteErrorKind.MANIFEST_INVALID
    assert "no aiperf.json describing it" in str(err)


@pytest.mark.asyncio
async def test_probe_missing_folder_keeps_not_found(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {"data/t.jsonl": b"{}\n"})
    err = await _probe_error(repo, "hf://org/repo/nope")
    assert err.kind == RemoteErrorKind.NOT_FOUND


def _folder_manifest(path: str) -> bytes:
    return orjson.dumps(
        {"aiperf_manifest_version": 1, "format": "mooncake_trace", "path": path}
    )


@pytest.mark.asyncio
async def test_probe_manifest_parquet_folder_lists_train_shards(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "aiperf.json": _folder_manifest("data"),
        "data/train-00000-of-00002.parquet": parquet_bytes(ROWS[:1]),
        "data/train-00001-of-00002.parquet": parquet_bytes(ROWS[1:]),
        "data/test-00000-of-00001.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    probe, source = await _resolve(repo, "hf://org/repo")
    shards = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    assert probe.manifest is not None and probe.data_is_dir
    assert probe.data_files == shards
    assert probe.pinned_ref == f"hf://datasets/org/repo@{SHA}"
    assert repo.allow_patterns == [shards]
    assert list(iter_records(source.local_path)) == ROWS


@pytest.mark.asyncio
async def test_probe_manifest_jsonl_folder_has_no_shards(tmp_path: Path) -> None:
    repo = ParquetRepo(tmp_path, {
        "aiperf.json": _folder_manifest("data"),
        "data/a.jsonl": b"{}\n",
    })  # fmt: skip
    probe, _ = await _resolve(repo, "hf://org/repo")
    assert probe.data_is_dir and probe.data_files is None
    assert repo.allow_patterns == [["data/**"]]


@pytest.mark.asyncio
async def test_probe_manifest_parquet_folder_without_train_fails(
    tmp_path: Path,
) -> None:
    repo = ParquetRepo(tmp_path, {
        "aiperf.json": _folder_manifest("data"),
        "data/test-00000-of-00001.parquet": parquet_bytes(ROWS),
        "data/validation-00000-of-00001.parquet": parquet_bytes(ROWS),
    })  # fmt: skip
    err = await _probe_error(repo, "hf://org/repo")
    assert err.kind == RemoteErrorKind.MANIFEST_INVALID
    assert "test, validation" in str(err)
