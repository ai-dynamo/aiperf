# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest import param

from aiperf.common.record_files import PYARROW_MISSING
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.manifest import AIPerfManifest, parse_manifest
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource
from aiperf.dataset.remote.preflight import (
    _check_first_record_format,
    _require_parquet_support,
)
from aiperf.dataset.remote.sniff import remote_first_record_has_timestamp
from aiperf.plugin.enums import CustomDatasetType

ROWS = [{"timestamp": 0, "input_length": 10, "output_length": 2, "hash_ids": [0]}]
SINGLE_TURN_ROWS = [{"text": "hello"}]
REF = RemoteRef(ref="hf://datasets/org/repo/d").with_version("a" * 40)


def _shards(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(rows), directory / "train-00000-of-00001.parquet"
    )
    return directory


def _probe(
    *,
    filename: str = "d",
    data_is_dir: bool = True,
    data_files: list[str] | None = None,
    manifest: AIPerfManifest | None = None,
) -> ProbeResult:
    return ProbeResult(
        ref=REF, pinned_ref=REF.display, data_relpath="d", data_is_dir=data_is_dir,
        filename=filename, manifest=manifest, manifest_sha256=None, needs_auth=False,
        data_files=data_files,
    )  # fmt: skip


def _source(local: Path, probe: ProbeResult) -> ResolvedSource:
    return ResolvedSource(
        probe=probe, local_path=local,
        files=[RemoteFile(path="d/train-00000-of-00001.parquet", sha256="f" * 64, size_bytes=1)],
    )  # fmt: skip


@pytest.mark.asyncio
async def test_remote_sniff_parquet_layout_reads_download(tmp_path: Path) -> None:
    probe = _probe(data_files=["d/train-00000-of-00001.parquet"])
    source = _source(_shards(tmp_path / "d", ROWS), probe)
    with patch(
        "aiperf.dataset.remote.service.materialize_probe",
        AsyncMock(return_value=source),
    ) as materialize:
        assert await remote_first_record_has_timestamp(probe)
    materialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_remote_sniff_manifest_parquet_folder_reads_download(
    tmp_path: Path,
) -> None:
    manifest = parse_manifest(
        b'{"aiperf_manifest_version": 1, "format": "mooncake_trace", "path": "d"}',
        ref_display=REF.display, kind="root",
    )  # fmt: skip
    probe = _probe(data_files=["d/train-00000-of-00001.parquet"], manifest=manifest)
    source = _source(_shards(tmp_path / "d", ROWS), probe)
    with patch(
        "aiperf.dataset.remote.service.materialize_probe",
        AsyncMock(return_value=source),
    ):
        assert await remote_first_record_has_timestamp(probe)


@pytest.mark.asyncio
async def test_remote_sniff_non_parquet_folder_stays_false_without_download() -> None:
    with patch(
        "aiperf.dataset.remote.service.materialize_probe", AsyncMock()
    ) as materialize:
        assert not await remote_first_record_has_timestamp(_probe())
    materialize.assert_not_awaited()


def test_check_first_record_format_parquet_wrong_shape_raises(tmp_path: Path) -> None:
    probe = _probe(data_files=["d/train-00000-of-00001.parquet"])
    source = _source(_shards(tmp_path / "d", SINGLE_TURN_ROWS), probe)
    with pytest.raises(RemoteSourceError) as exc_info:
        _check_first_record_format(CustomDatasetType.MOONCAKE_TRACE, source)
    assert exc_info.value.kind == RemoteErrorKind.INCOMPATIBLE


def test_check_first_record_format_parquet_right_shape_passes(tmp_path: Path) -> None:
    probe = _probe(data_files=["d/train-00000-of-00001.parquet"])
    source = _source(_shards(tmp_path / "d", ROWS), probe)
    _check_first_record_format(CustomDatasetType.MOONCAKE_TRACE, source)


@pytest.mark.parametrize(
    "probe,fails",
    [
        param(_probe(filename="t.parquet", data_is_dir=False), True, id="parquet-file"),
        param(_probe(data_files=["d/train-00000-of-00001.parquet"]), True, id="hf-layout"),
        param(_probe(filename="t.jsonl", data_is_dir=False), False, id="jsonl"),
    ],
)  # fmt: skip
def test_require_parquet_support_without_pyarrow(
    probe: ProbeResult, fails: bool
) -> None:
    with patch("aiperf.dataset.remote.preflight.parquet_available", return_value=False):
        if fails:
            with pytest.raises(RemoteSourceError) as exc_info:
                _require_parquet_support(probe)
            assert exc_info.value.kind == RemoteErrorKind.MISSING_DEPENDENCY
            assert PYARROW_MISSING in str(exc_info.value)
        else:
            _require_parquet_support(probe)
