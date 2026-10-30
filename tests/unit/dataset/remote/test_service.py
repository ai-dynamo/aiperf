# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import errno
import gzip
import hashlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import orjson
import pytest
from aiohttp import web
from filelock import Timeout
from pytest import param

from aiperf.common.enums import DatasetType
from aiperf.config.dataset import FileDataset
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import service
from aiperf.dataset.remote.cache import remote_cache_dir
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError, RemoteStage
from aiperf.dataset.remote.index import read_index
from aiperf.dataset.remote.manifest import AIPerfManifest
from aiperf.dataset.remote.models import UNVERSIONED, ProbeResult

BODY = b'{"timestamp": 1, "input_length": 4, "output_length": 2}\n'


async def _serve(http_server, tmp_path: Path, data: bytes = BODY) -> str:
    root = tmp_path / "srv"
    root.mkdir(exist_ok=True)
    (root / "t.jsonl").write_bytes(data)
    app = web.Application()
    app.router.add_static("/", root)
    return f"{await http_server(app)}/t.jsonl"


async def _serve_counting(
    http_server, gets: list[str], *, etag: str | None = '"v1"', delay_s: float = 0.0
) -> str:
    """Serve BODY at /t.jsonl, recording each full GET; HEAD is not counted."""

    async def handler(request: web.Request) -> web.Response:
        headers = {"ETag": etag} if etag else {}
        if request.method == "GET":
            gets.append(request.path)
            if delay_s:
                # asyncio.sleep is patched to a no-op in unit tests; this delay is real.
                done = asyncio.get_running_loop().create_future()
                asyncio.get_running_loop().call_later(delay_s, done.set_result, None)
                await done
        return web.Response(body=BODY, headers=headers)

    app = web.Application()
    app.router.add_get("/t.jsonl", handler)
    return f"{await http_server(app)}/t.jsonl"


@pytest.mark.asyncio
async def test_probe_materialize_cached_local_path(http_server, tmp_path: Path) -> None:
    url = await _serve(http_server, tmp_path)
    probe = await service.probe_ref(RemoteRef(ref=url))
    source = await service.materialize_probe(probe)
    assert service.cached_source(probe.ref).local_path == source.local_path
    assert service.local_path_for(probe.ref) == source.local_path
    assert service.local_path_for(tmp_path) == tmp_path


@pytest.mark.asyncio
async def test_second_materialize_is_a_cache_hit(
    http_server, tmp_path: Path, monkeypatch
) -> None:
    url = await _serve(http_server, tmp_path)
    probe = await service.probe_ref(RemoteRef(ref=url))
    first = await service.materialize_probe(probe)
    calls = []
    monkeypatch.setattr(service, "get_resolver", lambda scheme: calls.append(scheme))
    assert (await service.materialize_probe(probe)).local_path == first.local_path
    assert calls == []


@pytest.mark.asyncio
async def test_concurrent_materializations_download_once(http_server) -> None:
    gets: list[str] = []
    url = await _serve_counting(http_server, gets, delay_s=0.2)
    probe = await service.probe_ref(RemoteRef(ref=url))
    first, second = await asyncio.gather(
        service.materialize_probe(probe), service.materialize_probe(probe)
    )
    assert gets == ["/t.jsonl"]
    assert first.local_path == second.local_path
    assert first.local_path.read_bytes() == BODY


@pytest.mark.asyncio
async def test_truncated_cached_file_is_a_miss(http_server) -> None:
    gets: list[str] = []
    url = await _serve_counting(http_server, gets)
    probe = await service.probe_ref(RemoteRef(ref=url))
    first = await service.materialize_probe(probe)
    first.local_path.write_bytes(BODY[:5])
    with pytest.raises(RuntimeError, match="not in the remote dataset cache"):
        service.cached_source(probe.ref)
    again = await service.materialize_probe(probe)
    assert len(gets) == 2
    assert again.local_path.read_bytes() == BODY
    assert service.cached_source(probe.ref).local_path == again.local_path


@pytest.mark.asyncio
async def test_unversioned_source_downloads_every_time(http_server) -> None:
    gets: list[str] = []
    url = await _serve_counting(http_server, gets, etag=None)
    probe = await service.probe_ref(RemoteRef(ref=url))
    assert probe.ref.resolved_version == UNVERSIONED
    await service.materialize_probe(probe)
    await service.materialize_probe(probe)
    assert len(gets) == 2
    assert service.local_path_for(probe.ref).read_bytes() == BODY


@pytest.mark.asyncio
async def test_offline_serves_cached_unversioned_source(
    http_server, monkeypatch
) -> None:
    gets: list[str] = []
    url = await _serve_counting(http_server, gets, etag=None)
    await service.materialize_probe(await service.probe_ref(RemoteRef(ref=url)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    probe = await service.probe_ref(RemoteRef(ref=url))
    source = await service.materialize_probe(probe)
    assert source.local_path.read_bytes() == BODY
    assert len(gets) == 1


@pytest.mark.asyncio
async def test_lock_timeout_is_a_transient_remote_error(
    http_server, tmp_path: Path, monkeypatch
) -> None:
    url = await _serve(http_server, tmp_path)
    probe = await service.probe_ref(RemoteRef(ref=url))

    @contextlib.asynccontextmanager
    async def never_acquired(entry: Path) -> AsyncIterator[None]:
        raise Timeout(f"{entry}.lock")
        yield

    monkeypatch.setattr(service, "entry_lock", never_acquired)
    with pytest.raises(RemoteSourceError, match=r"\.lock") as exc:
        await service.materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.TRANSIENT
    assert exc.value.stage == RemoteStage.MATERIALIZE


@pytest.mark.asyncio
async def test_checksum_mismatch_is_reported(http_server, tmp_path: Path) -> None:
    root = tmp_path / "srv"
    root.mkdir()
    (root / "t.jsonl").write_bytes(BODY)
    (root / "t.jsonl.aiperf.json").write_bytes(
        orjson.dumps({"aiperf_manifest_version": 1, "sha256": "0" * 64})
    )
    app = web.Application()
    app.router.add_static("/", root)
    probe = await service.probe_ref(RemoteRef(ref=f"{await http_server(app)}/t.jsonl"))
    with pytest.raises(RemoteSourceError) as exc:
        await service.materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.CHECKSUM_MISMATCH
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_cache_hit_is_checked_against_a_changed_sidecar_checksum(
    http_server, tmp_path: Path
) -> None:
    root = tmp_path / "srv"
    root.mkdir()
    (root / "t.jsonl").write_bytes(BODY)
    sidecar = root / "t.jsonl.aiperf.json"
    sidecar.write_bytes(orjson.dumps({"aiperf_manifest_version": 1, "sha256": hashlib.sha256(BODY).hexdigest()}))  # fmt: skip
    app = web.Application()
    app.router.add_static("/", root)
    ref = RemoteRef(ref=f"{await http_server(app)}/t.jsonl")
    first = await service.probe_ref(ref)
    await service.materialize_probe(first)
    sidecar.write_bytes(orjson.dumps({"aiperf_manifest_version": 1, "sha256": "0" * 64}))  # fmt: skip
    second = await service.probe_ref(ref)
    assert second.ref.resolved_version == first.ref.resolved_version
    with pytest.raises(RemoteSourceError) as exc:
        await service.materialize_probe(second)
    assert exc.value.kind == RemoteErrorKind.CHECKSUM_MISMATCH


async def _serve_with_sidecar(
    http_server, tmp_path: Path, sidecar: dict
) -> tuple[RemoteRef, Path]:
    root = tmp_path / "srv"
    root.mkdir()
    (root / "t.jsonl").write_bytes(BODY)
    (root / "t.jsonl.aiperf.json").write_bytes(orjson.dumps(sidecar))
    app = web.Application()
    app.router.add_static("/", root)
    return RemoteRef(
        ref=f"{await http_server(app)}/t.jsonl"
    ), root / "t.jsonl.aiperf.json"


@pytest.mark.asyncio
async def test_cache_hit_records_the_sidecar_this_run_accepted(
    http_server, tmp_path: Path
) -> None:
    """Provenance reads the index, so a hit must not keep describing the old sidecar."""
    ref, sidecar = await _serve_with_sidecar(http_server, tmp_path, {"aiperf_manifest_version": 1, "block_size": 16})  # fmt: skip
    await service.materialize_probe(await service.probe_ref(ref))
    sidecar.write_bytes(orjson.dumps({"aiperf_manifest_version": 1, "block_size": 64}))  # fmt: skip
    second = await service.probe_ref(ref)
    source = await service.materialize_probe(second)
    assert source.probe.manifest.block_size == 64
    indexed = read_index(second.ref)
    assert indexed.probe.manifest_sha256 == second.manifest_sha256
    assert indexed.probe.manifest.block_size == 64


@pytest.mark.asyncio
async def test_cache_hit_rehashes_a_file_whose_manifest_declares_sha256(
    http_server, tmp_path: Path
) -> None:
    """A same-size corruption passes the size check; the declared sha256 catches it."""
    sha = hashlib.sha256(BODY).hexdigest()
    ref, _ = await _serve_with_sidecar(http_server, tmp_path, {"aiperf_manifest_version": 1, "sha256": sha})  # fmt: skip
    first = await service.materialize_probe(await service.probe_ref(ref))
    first.local_path.write_bytes(bytes(b ^ 0xFF for b in BODY))
    second = await service.materialize_probe(await service.probe_ref(ref))
    assert second.local_path.read_bytes() == BODY


@pytest.mark.parametrize("data_is_dir,warned", [param(True, True, id="folder"), param(False, False, id="file")])  # fmt: skip
@pytest.mark.asyncio
async def test_probe_warns_when_a_folder_manifest_declares_sha256(
    monkeypatch, caplog, data_is_dir: bool, warned: bool
) -> None:
    """sha256 covers single files only; a folder's publisher must not think it is checked."""
    ref = RemoteRef(ref="s3://bucket/set/").with_version("list:abc")
    manifest = AIPerfManifest(aiperf_manifest_version=1, format="mooncake_trace", path="data", sha256="0" * 64)  # fmt: skip
    probe = ProbeResult(
        ref=ref, pinned_ref=ref.display, data_relpath="data", data_is_dir=data_is_dir,
        filename="data", manifest=manifest, manifest_sha256="m", needs_auth=True,
    )  # fmt: skip

    class Resolver:
        async def probe(self, _ref):
            return probe

    monkeypatch.setattr(service, "get_resolver", lambda scheme: Resolver())
    with caplog.at_level("WARNING"):
        await service.probe_ref(RemoteRef(ref="s3://bucket/set/"))
    assert ("sha256 is checked for single files only" in caplog.text) is warned


@pytest.mark.asyncio
async def test_materialize_failure_writes_no_index(
    http_server, tmp_path: Path, monkeypatch
) -> None:
    url = await _serve(http_server, tmp_path)
    probe = await service.probe_ref(RemoteRef(ref=url))
    real_get_resolver = service.get_resolver

    class Broken:
        async def materialize(self, _probe):
            raise RemoteSourceError(
                ref=url,
                kind=RemoteErrorKind.NOT_FOUND,
                stage=RemoteStage.MATERIALIZE,
                what="x",
                hint="y",
            )

    monkeypatch.setattr(service, "get_resolver", lambda scheme: Broken())
    with pytest.raises(RemoteSourceError):
        await service.materialize_probe(probe)
    assert read_index(probe.ref) is None
    with pytest.raises(RuntimeError, match="not in the remote dataset cache"):
        service.cached_source(probe.ref)

    # The failure released the entry lock, so a later attempt can still succeed.
    monkeypatch.setattr(service, "get_resolver", real_get_resolver)
    source = await service.materialize_probe(probe)
    assert read_index(probe.ref) == source


@pytest.mark.asyncio
async def test_transient_errors_retry_then_succeed() -> None:
    attempts = []

    async def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RemoteSourceError(
                ref="https://h/t",
                kind=RemoteErrorKind.TRANSIENT,
                stage=RemoteStage.ACCESS,
                what="x",
                hint="y",
            )
        return "ok"

    assert await service.with_transient_retry(flaky) == "ok"
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_transient_errors_give_up_after_retry_attempts() -> None:
    attempts = []

    async def down() -> str:
        attempts.append(1)
        raise RemoteSourceError(
            ref="https://h/t",
            kind=RemoteErrorKind.TRANSIENT,
            stage=RemoteStage.ACCESS,
            what="x",
            hint="y",
        )

    with pytest.raises(RemoteSourceError):
        await service.with_transient_retry(down)
    assert len(attempts) == service.RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_non_transient_errors_do_not_retry() -> None:
    attempts = []

    async def denied() -> str:
        attempts.append(1)
        raise RemoteSourceError(
            ref="https://h/t",
            kind=RemoteErrorKind.AUTH_DENIED,
            stage=RemoteStage.ACCESS,
            what="x",
            hint="y",
        )

    with pytest.raises(RemoteSourceError):
        await service.with_transient_retry(denied)
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_offline_uses_cache_and_errors_when_empty(
    http_server, tmp_path: Path, monkeypatch
) -> None:
    url = await _serve(http_server, tmp_path)
    await service.materialize_probe(await service.probe_ref(RemoteRef(ref=url)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    probe = await service.probe_ref(RemoteRef(ref=url))
    assert service.cached_source(probe.ref).local_path.read_bytes() == BODY
    with pytest.raises(RemoteSourceError, match="offline"):
        await service.probe_ref(RemoteRef(ref="https://never.example/t.jsonl"))


def test_remote_offline_reads_setting_and_hf_env(monkeypatch) -> None:
    assert service.remote_offline() is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "TRUE")
    assert service.remote_offline() is True
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert service.remote_offline() is False
    monkeypatch.setattr(service.Environment.DATASET, "REMOTE_OFFLINE", True)
    assert service.remote_offline() is True


def test_cached_source_requires_resolution_and_materialization() -> None:
    with pytest.raises(RuntimeError, match="never resolved"):
        service.cached_source(RemoteRef(ref="https://h/t.jsonl"))
    with pytest.raises(RuntimeError, match="not in the remote dataset cache"):
        service.cached_source(RemoteRef(ref="https://h/t.jsonl").with_version("etag:x"))


@pytest.mark.asyncio
async def test_presigned_ref_serializes_redacted_and_child_finds_cache(
    http_server, tmp_path: Path
) -> None:
    url = (
        await _serve(http_server, tmp_path) + "?X-Amz-Signature=topsecret&X-Amz-Date=1"
    )
    probe = await service.probe_ref(RemoteRef(ref=url))
    await service.materialize_probe(probe)
    ds = FileDataset.model_validate(
        {"name": "main", "type": DatasetType.FILE, "path": probe.ref}
    )
    serialized = ds.model_dump_json()
    assert "topsecret" not in serialized
    child = FileDataset.model_validate(orjson.loads(serialized))
    assert "topsecret" not in child.path.ref
    assert service.local_path_for(child.path).read_bytes() == BODY
    assert (
        hashlib.sha256(BODY).hexdigest()
        == service.cached_source(child.path).files[0].sha256
    )
    # Nothing AIPerf wrote to its cache holds the signature either.
    for path in remote_cache_dir().rglob("*"):
        assert "topsecret" not in str(path)
        if path.is_file() and path.suffix == ".json":
            assert b"topsecret" not in path.read_bytes()


@pytest.mark.asyncio
async def test_gzip_content_encoding_is_cached_as_published(http_server) -> None:
    stored = gzip.compress(BODY)
    seen_encodings: list[str | None] = []

    async def handler(request: web.Request) -> web.Response:
        seen_encodings.append(request.headers.get("Accept-Encoding"))
        if request.path.endswith(".aiperf.json"):
            manifest = {
                "aiperf_manifest_version": 1,
                "sha256": hashlib.sha256(stored).hexdigest(),
            }
            return web.Response(body=orjson.dumps(manifest))
        # An object store serves its stored Content-Encoding metadata whatever the
        # client accepts.
        return web.Response(
            body=stored, headers={"ETag": '"gz"', "Content-Encoding": "gzip"}
        )

    app = web.Application()
    app.router.add_route("*", "/{name}", handler)
    url = f"{await http_server(app)}/t.jsonl.gz"
    probe = await service.probe_ref(RemoteRef(ref=url))
    prefix, _ = await service.get_resolver("http").read_prefix(probe, 4)
    source = await service.materialize_probe(probe)
    assert prefix == stored[:4]
    assert source.local_path.read_bytes() == stored
    assert source.files[0].size_bytes == len(stored)
    assert seen_encodings and set(seen_encodings) == {"identity"}


@pytest.mark.parametrize(
    "err",
    [
        param(errno.ENOSPC, id="disk-full"),
        param(errno.EACCES, id="permission-denied"),
        param(errno.EROFS, id="read-only"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_https_local_disk_error_is_not_retried_and_names_cache_dir(
    http_server, tmp_path: Path, err: int
) -> None:
    probe = await service.probe_ref(RemoteRef(ref=await _serve(http_server, tmp_path)))
    failing = MagicMock(side_effect=OSError(err, os.strerror(err)))
    with (
        patch("aiperf.dataset.remote.http_fetch.stream_to_file", failing),
        pytest.raises(RemoteSourceError, match="cannot be written") as exc,
    ):
        await service.materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert exc.value.stage == RemoteStage.MATERIALIZE
    assert str(remote_cache_dir()) in str(exc.value)
    assert failing.call_count == 1
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_unwritable_cache_dir_maps_to_cannot_be_written(
    http_server, tmp_path: Path, monkeypatch
) -> None:
    probe = await service.probe_ref(RemoteRef(ref=await _serve(http_server, tmp_path)))
    monkeypatch.setattr(
        service, "write_index", MagicMock(side_effect=OSError(errno.EROFS, "Read-only"))
    )
    with pytest.raises(RemoteSourceError, match="cannot be written") as exc:
        await service.materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert "AIPERF_DATASET_REMOTE_CACHE_DIR" in str(exc.value)
