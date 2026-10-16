# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3 fetch_file and materialize's reuse of the fetched shard."""

import contextlib
import errno
import hashlib
import io
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("botocore")

from botocore.response import StreamingBody  # noqa: E402
from filelock import Timeout  # noqa: E402

from aiperf.config.dataset.remote_ref import RemoteRef  # noqa: E402
from aiperf.dataset.remote import s3  # noqa: E402
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

SHARDS = {
    "set/traces/train-00000-of-00002.parquet": b"PAR1-first",
    "set/traces/train-00001-of-00002.parquet": b"PAR1-second",
}
ETAGS = {key: f'"e{i}"' for i, key in enumerate(SHARDS)}
FIRST = next(iter(SHARDS))


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(data), len(data))


def _client(head_etags: dict[str, str] = ETAGS) -> MagicMock:
    client = MagicMock()
    client.head_object.side_effect = lambda **kw: {"ETag": head_etags[kw["Key"]]}
    client.get_object.side_effect = lambda **kw: {"Body": _body(SHARDS[kw["Key"]])}
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
    probe = _probe()
    resolver = S3SourceResolver(client=client)
    await resolver.fetch_file(probe, FIRST)
    await resolver.materialize(probe)
    assert client.get_object.call_count == 3


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
