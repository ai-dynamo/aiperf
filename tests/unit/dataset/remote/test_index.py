# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.common.enums import DatasetType
from aiperf.common.environment import Environment
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import cache, index
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource


@pytest.fixture(autouse=True)
def _cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Environment.DATASET, "REMOTE_CACHE_DIR", tmp_path / "remote")


def _source(version: str, ref: str = "https://h/t.jsonl") -> ResolvedSource:
    pinned = RemoteRef(ref=ref).with_version(version)
    # Versions like "etag:1" are not valid Windows path names; cache.entry_dir sanitizes them.
    local = cache.entry_dir(pinned) / "t.jsonl"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("{}\n")
    probe = ProbeResult(
        ref=pinned, pinned_ref=pinned.display, data_relpath=None, data_is_dir=False,
        filename="t.jsonl", manifest=None, manifest_sha256=None, needs_auth=False,
    )  # fmt: skip
    return ResolvedSource(
        probe=probe,
        local_path=local,
        files=[RemoteFile(path="t.jsonl", sha256="0" * 64, size_bytes=3)],
    )


def test_write_then_read_round_trips() -> None:
    source = _source("etag:1")
    index.write_index(source)
    again = index.read_index(RemoteRef(ref="https://h/t.jsonl").with_version("etag:1"))
    assert again is not None
    assert again.local_path == source.local_path
    assert again.probe.ref.resolved_version == "etag:1"
    assert again.files == source.files


def test_read_missing_or_unresolved_is_none() -> None:
    assert (
        index.read_index(RemoteRef(ref="https://h/t.jsonl").with_version("nope"))
        is None
    )
    assert index.read_index(RemoteRef(ref="https://h/t.jsonl")) is None


def test_index_path_requires_resolved_version() -> None:
    with pytest.raises(ValueError, match="resolved_version"):
        index.index_path(RemoteRef(ref="https://h/t.jsonl"))


def test_latest_indexed_version_picks_newest() -> None:
    old, new = _source("etag:old"), _source("etag:new")
    index.write_index(old)
    index.write_index(new)
    os.utime(index.index_path(old.probe.ref), (1, 1))
    assert (
        index.latest_indexed_version(RemoteRef(ref="https://h/t.jsonl")) == "etag:new"
    )


def test_latest_indexed_version_none_without_entries() -> None:
    assert index.latest_indexed_version(RemoteRef(ref="https://h/t.jsonl")) is None


def test_presigned_index_written_redacted_is_found_by_redacted_ref() -> None:
    secret = "https://h/t.jsonl?X-Amz-Signature=secret"
    source = _source("etag:1", ref=secret)
    index.write_index(source)
    assert "secret" not in index.index_path(source.probe.ref).read_text()
    redacted = RemoteRef(ref=RemoteRef(ref=secret).display).with_version("etag:1")
    assert index.read_index(redacted) is not None


def test_describe_files_for_file_and_directory(tmp_path: Path) -> None:
    f = tmp_path / "one.jsonl"
    f.write_bytes(b"abc")
    assert index.describe_files(f, "data/one.jsonl") == [
        RemoteFile(
            path="data/one.jsonl",
            sha256=hashlib.sha256(b"abc").hexdigest(),
            size_bytes=3,
        )
    ]
    d = tmp_path / "dir"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "b.json").write_bytes(b"b")
    (d / "a.json").write_bytes(b"a")
    assert [r.path for r in index.describe_files(d, "traces")] == [
        "traces/a.json",
        "traces/sub/b.json",
    ]


def test_source_resolver_protocol_is_runtime_checkable() -> None:
    from aiperf.dataset.remote.protocols import SourceResolverProtocol

    class _Full:
        async def probe(self, ref): ...
        async def read_prefix(self, probe, nbytes): ...
        async def materialize(self, probe): ...

    assert isinstance(_Full(), SourceResolverProtocol)
    assert not isinstance(object(), SourceResolverProtocol)


_REF = "https://h/t.jsonl"


def _pinned(version: str) -> RemoteRef:
    return RemoteRef(ref=_REF).with_version(version)


def test_read_index_truncated_file_is_none(caplog: pytest.LogCaptureFixture) -> None:
    source = _source("v1")
    index.write_index(source)
    path = index.index_path(source.probe.ref)
    path.write_bytes(path.read_bytes()[:20])
    with caplog.at_level("WARNING"):
        assert index.read_index(_pinned("v1")) is None
    assert any(
        str(path) in r.getMessage() and "https://h/t.jsonl" in r.getMessage()
        for r in caplog.records
    )


def test_read_index_empty_file_is_none() -> None:
    source = _source("v1")
    index.write_index(source)
    index.index_path(source.probe.ref).write_bytes(b"")
    assert index.read_index(_pinned("v1")) is None


def test_read_index_old_schema_is_none() -> None:
    source = _source("v1")
    index.write_index(source)
    path = index.index_path(source.probe.ref)
    data = orjson.loads(path.read_bytes())
    del data["files"][0]["size_bytes"]
    path.write_bytes(orjson.dumps(data))
    assert index.read_index(_pinned("v1")) is None


def test_latest_indexed_version_skips_corrupt_newest() -> None:
    old, new = _source("v-old"), _source("v-new")
    index.write_index(old)
    index.write_index(new)
    os.utime(index.index_path(old.probe.ref), (1, 1))
    index.index_path(new.probe.ref).write_bytes(b"{not json")
    assert index.latest_indexed_version(_pinned("x")) == "v-old"


def test_latest_indexed_version_all_corrupt_is_none() -> None:
    source = _source("v1")
    index.write_index(source)
    index.index_path(source.probe.ref).write_bytes(b"")
    assert index.latest_indexed_version(_pinned("x")) is None


def test_latest_indexed_version_equal_mtimes_is_deterministic() -> None:
    a, b = _source("v-a"), _source("v-b")
    index.write_index(a)
    index.write_index(b)
    paths = {
        p: s.probe.ref.resolved_version
        for p, s in ((index.index_path(s.probe.ref), s) for s in (a, b))
    }
    for p in paths:
        os.utime(p, ns=(10**18, 10**18))
    expected = paths[max(paths, key=lambda p: p.name)]
    assert {index.latest_indexed_version(_pinned("x")) for _ in range(5)} == {expected}


@pytest.mark.parametrize(
    "ref",
    [
        param("https://user:pw@h.example/t.jsonl", id="user-and-password"),
        param("https://token@h.example/t.jsonl", id="user-only"),
        param("https://user:pw@h.example/t.jsonl?X-Amz-Signature=s", id="userinfo-and-presigned"),
    ],
)  # fmt: skip
def test_basic_auth_ref_finds_same_entry_after_run_config_round_trip(ref: str) -> None:
    pinned = RemoteRef(ref=ref).with_version("etag:1")
    ds = FileDataset.model_validate(
        {"name": "main", "type": DatasetType.FILE, "path": pinned}
    )
    serialized = ds.model_dump_json()
    assert "pw@" not in serialized
    child = FileDataset.model_validate(orjson.loads(serialized)).path
    assert index.index_path(child) == index.index_path(pinned)
    assert cache.entry_dir(child) == cache.entry_dir(pinned)
