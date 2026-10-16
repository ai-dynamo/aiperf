# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3 fetch_file and materialize's reuse of the fetched shard."""

import contextlib
import errno
import hashlib
import io
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("botocore")

from botocore.response import StreamingBody  # noqa: E402
from filelock import Timeout  # noqa: E402

from aiperf.config.dataset.remote_ref import RemoteRef  # noqa: E402
from aiperf.dataset.remote import s3, service  # noqa: E402
from aiperf.dataset.remote.cache import entry_dir  # noqa: E402
from aiperf.dataset.remote.errors import (  # noqa: E402
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
)
from aiperf.dataset.remote.models import ProbeResult  # noqa: E402
from aiperf.dataset.remote.s3 import (  # noqa: E402
    S3SourceResolver,
    _listing_version,
    _Object,
)
from aiperf.dataset.remote.sniff import (  # noqa: E402
    remote_first_record_has_timestamp,
)

SHARDS = {
    "set/traces/train-00000-of-00002.parquet": b"PAR1-first",
    "set/traces/train-00001-of-00002.parquet": b"PAR1-second",
}
ETAGS = {key: f'"e{i}"' for i, key in enumerate(SHARDS)}
FIRST = next(iter(SHARDS))


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(data), len(data))


def _client(
    head_etags: dict[str, str] = ETAGS, bodies: dict[str, bytes] = SHARDS
) -> MagicMock:
    client = MagicMock()
    client.head_object.side_effect = lambda **kw: {"ETag": head_etags[kw["Key"]]}
    client.get_object.side_effect = lambda **kw: {"Body": _body(bodies[kw["Key"]])}
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": key, "ETag": ETAGS[key]} for key in SHARDS]}
    ]
    return client


def _probe() -> ProbeResult:
    objects = [_Object(key, ETAGS[key], None) for key in SHARDS]
    ref = RemoteRef(ref="s3://bucket/set/").with_version(
        _listing_version("m" * 64, objects)
    )
    return ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath="set/traces", data_is_dir=True,
        filename="traces", manifest=None, manifest_sha256="m" * 64, needs_auth=True,
        data_files=list(SHARDS),
    )  # fmt: skip


@pytest.mark.asyncio
async def test_s3_fetch_file_downloads_one_object_into_the_cache_entry() -> None:
    client = _client()
    probe = _probe()
    path = await S3SourceResolver(client=client).fetch_file(probe, FIRST)
    assert path == entry_dir(probe.ref) / "traces" / "train-00000-of-00002.parquet"
    assert path.read_bytes() == SHARDS[FIRST]
    client.get_object.assert_called_once()
    assert client.get_object.call_args.kwargs == {
        "Bucket": "bucket",
        "Key": FIRST,
        "IfMatch": ETAGS[FIRST],
    }


@pytest.mark.asyncio
async def test_s3_materialize_reuses_fetched_shard_when_etag_matches() -> None:
    client = _client()
    probe = _probe()
    resolver = S3SourceResolver(client=client)
    await resolver.fetch_file(probe, FIRST)
    source = await resolver.materialize(probe)
    assert client.get_object.call_count == 2
    assert [f.path for f in source.files] == list(SHARDS)
    assert source.files[0].sha256 == hashlib.sha256(SHARDS[FIRST]).hexdigest()


@pytest.mark.asyncio
async def test_s3_materialize_refetches_shard_whose_etag_changed() -> None:
    client = _client(head_etags={**ETAGS, FIRST: '"old"'})
    gets = iter([b"stale-bytes"])
    serve = client.get_object.side_effect
    client.get_object.side_effect = lambda **kw: (
        {"Body": _body(next(gets))} if kw["Key"] == FIRST and kw["IfMatch"] == '"old"' else serve(**kw)
    )  # fmt: skip
    probe = _probe()
    resolver = S3SourceResolver(client=client)
    fetched = await resolver.fetch_file(probe, FIRST)
    marker = s3._fetched_marker(entry_dir(probe.ref), FIRST)
    assert marker.read_text(encoding="utf-8") == '"old"'
    source = await resolver.materialize(probe)
    assert client.get_object.call_count == 3
    assert not marker.exists()
    assert fetched.read_bytes() == SHARDS[FIRST]
    assert source.files[0].sha256 == hashlib.sha256(SHARDS[FIRST]).hexdigest()


@pytest.mark.asyncio
async def test_s3_fetch_file_rejects_path_outside_the_probed_folder() -> None:
    with pytest.raises(ValueError, match="not a file under the probed folder"):
        await S3SourceResolver(client=_client()).fetch_file(
            _probe(), "elsewhere/x.parquet"
        )


@pytest.mark.asyncio
async def test_s3_fetch_file_reuses_its_earlier_copy_when_etag_matches() -> None:
    client = _client()
    probe = _probe()
    resolver = S3SourceResolver(client=client)
    first = await resolver.fetch_file(probe, FIRST)
    second = await resolver.fetch_file(probe, FIRST)
    assert first == second
    client.get_object.assert_called_once()


@pytest.mark.asyncio
async def test_s3_fetch_file_writes_under_the_entry_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    locked: list[Path] = []

    @contextlib.asynccontextmanager
    async def recording_lock(entry: Path) -> AsyncIterator[None]:
        locked.append(entry)
        events.append("acquire")
        yield
        events.append("release")

    client = _client()
    get = client.get_object.side_effect
    client.get_object.side_effect = lambda **kw: events.append("get") or get(**kw)
    monkeypatch.setattr(s3, "entry_lock", recording_lock)
    probe = _probe()
    await S3SourceResolver(client=client).fetch_file(probe, FIRST)
    assert locked == [entry_dir(probe.ref)]
    assert events == ["acquire", "get", "release"]


@pytest.mark.asyncio
async def test_s3_fetch_file_lock_timeout_is_a_transient_remote_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextlib.asynccontextmanager
    async def never_acquired(entry: Path) -> AsyncIterator[None]:
        raise Timeout(f"{entry}.lock")
        yield

    monkeypatch.setattr(s3, "entry_lock", never_acquired)
    client = _client()
    with pytest.raises(RemoteSourceError, match=r"\.lock") as exc:
        await S3SourceResolver(client=client).fetch_file(_probe(), FIRST)
    assert exc.value.kind == RemoteErrorKind.TRANSIENT
    assert exc.value.stage == RemoteStage.ACCESS
    client.get_object.assert_not_called()


@pytest.mark.asyncio
async def test_s3_fetch_file_unwritable_cache_is_a_remote_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextlib.asynccontextmanager
    async def read_only(entry: Path) -> AsyncIterator[None]:
        raise OSError(errno.EROFS, "Read-only file system")
        yield

    monkeypatch.setattr(s3, "entry_lock", read_only)
    with pytest.raises(RemoteSourceError, match="cannot be written") as exc:
        await S3SourceResolver(client=_client()).fetch_file(_probe(), FIRST)
    assert exc.value.stage == RemoteStage.ACCESS


class _BrokenBody:
    """A GetObject body whose connection drops after the first chunk."""

    def iter_chunks(self, chunk_size: int):
        yield b"PAR1"
        raise OSError("connection reset")

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_s3_fetch_file_interrupted_download_leaves_no_marker() -> None:
    probe = _probe()
    dest = await S3SourceResolver(client=_client()).fetch_file(probe, FIRST)
    marker = s3._fetched_marker(entry_dir(probe.ref), FIRST)
    assert marker.exists()
    client = _client(head_etags={**ETAGS, FIRST: '"e-new"'})
    client.get_object.side_effect = lambda **kw: {"Body": _BrokenBody()}
    with pytest.raises(RemoteSourceError):
        await S3SourceResolver(client=client).fetch_file(probe, FIRST)
    assert not marker.exists()
    assert dest.read_bytes() == SHARDS[FIRST]
    assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]


def _parquet(rows: list[dict]) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


async def _indexed(client: MagicMock, probe: ProbeResult) -> Path:
    """Materialize and index ``probe``'s entry, then forget the calls that took."""
    with patch.object(
        service, "get_resolver", return_value=S3SourceResolver(client=client)
    ):
        await service.materialize_probe(probe)
    client.reset_mock()
    return entry_dir(probe.ref) / "traces" / "train-00000-of-00002.parquet"


@pytest.mark.asyncio
async def test_remote_sniff_indexed_parquet_folder_reads_cache_without_network() -> (
    None
):
    bodies = {key: _parquet([{"timestamp": 0, "input_length": 4}]) for key in SHARDS}
    client = _client(bodies=bodies)
    probe = _probe()
    shard = await _indexed(client, probe)
    before = shard.stat().st_mtime_ns
    with patch.object(
        service, "get_resolver", return_value=S3SourceResolver(client=client)
    ):
        assert await remote_first_record_has_timestamp(probe) is True
    client.head_object.assert_not_called()
    client.get_object.assert_not_called()
    assert shard.stat().st_mtime_ns == before


@pytest.mark.asyncio
async def test_s3_fetch_file_never_rewrites_an_indexed_entry() -> None:
    client = _client()
    probe = _probe()
    shard = await _indexed(client, probe)
    before = shard.stat().st_mtime_ns
    assert await S3SourceResolver(client=client).fetch_file(probe, FIRST) == shard
    client.get_object.assert_not_called()
    assert shard.stat().st_mtime_ns == before
