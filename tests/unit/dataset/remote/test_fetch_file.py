# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fetch_file and the first-shard timestamp sniff."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.hf import HFSourceResolver
from aiperf.dataset.remote.http import HTTPSourceResolver
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.dataset.remote.sniff import remote_first_record_has_timestamp

SHA = "a" * 40
SHARDS = ["d/train-00000-of-00002.parquet", "d/train-00001-of-00002.parquet"]


def _hf_probe() -> ProbeResult:
    ref = RemoteRef(ref="hf://datasets/org/repo/d").with_version(SHA)
    return ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath="d", data_is_dir=True,
        filename="d", manifest=None, manifest_sha256=None, needs_auth=False,
        data_files=list(SHARDS),
    )  # fmt: skip


def _shard(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train-00000-of-00002.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _resolver(path: Path) -> MagicMock:
    resolver = MagicMock()
    resolver.fetch_file = AsyncMock(return_value=path)
    return resolver


@pytest.mark.asyncio
async def test_hf_fetch_file_downloads_one_file_at_pinned_commit(
    tmp_path: Path,
) -> None:
    local = tmp_path / "train-00000-of-00002.parquet"
    local.write_bytes(b"PAR1")
    probe = _hf_probe()
    with patch("huggingface_hub.hf_hub_download", return_value=str(local)) as download:
        path = await HFSourceResolver().fetch_file(probe, SHARDS[0])
    assert path == local
    download.assert_called_once_with(
        repo_id="org/repo", filename=SHARDS[0], repo_type="dataset", revision=SHA
    )


@pytest.mark.asyncio
async def test_http_fetch_file_is_not_supported() -> None:
    ref = RemoteRef(ref="https://h.example/t.jsonl").with_version("etag:1")
    probe = ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath=None, data_is_dir=False,
        filename="t.jsonl", manifest=None, manifest_sha256=None, needs_auth=False,
    )  # fmt: skip
    with pytest.raises(NotImplementedError, match="single file"):
        await HTTPSourceResolver().fetch_file(probe, "t.jsonl")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,expected",
    [
        param([{"timestamp": 0, "input_length": 4, "output_length": 2}], True, id="timestamped"),
        param([{"input_length": 4, "output_length": 2}], False, id="untimed"),
    ],
)  # fmt: skip
async def test_remote_sniff_parquet_folder_reads_only_first_shard(
    tmp_path: Path, rows: list[dict], expected: bool
) -> None:
    resolver = _resolver(_shard(tmp_path, rows))
    probe = _hf_probe()
    with (
        patch("aiperf.dataset.remote.service.get_resolver", return_value=resolver),
        patch(
            "aiperf.dataset.remote.service.materialize_probe", AsyncMock()
        ) as materialize,
    ):
        assert await remote_first_record_has_timestamp(probe) is expected
    resolver.fetch_file.assert_awaited_once_with(probe, SHARDS[0])
    materialize.assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_sniff_parquet_folder_offline_reads_cached_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Environment.DATASET, "REMOTE_OFFLINE", True)
    folder = tmp_path / "d"
    _shard(folder, [{"timestamp": 0, "input_length": 4, "output_length": 2}])
    probe = _hf_probe()
    source = ResolvedSource(
        probe=probe, local_path=folder,
        files=[RemoteFile(path=SHARDS[0], sha256="f" * 64, size_bytes=1)],
    )  # fmt: skip
    resolver = _resolver(folder / "unused.parquet")
    with (
        patch("aiperf.dataset.remote.service.get_resolver", return_value=resolver),
        patch(
            "aiperf.dataset.remote.service.materialize_probe",
            AsyncMock(return_value=source),
        ),
    ):
        assert await remote_first_record_has_timestamp(probe) is True
    resolver.fetch_file.assert_not_awaited()
