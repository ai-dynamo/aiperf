# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import errno
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile, RepoFolder

from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.hf import HFSourceResolver
from aiperf.dataset.remote.index import read_index
from aiperf.dataset.remote.service import materialize_probe, probe_ref
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

SHA = "a" * 40
OTHER_SHA = "b" * 40


def _item(cls, path: str):
    obj = object.__new__(cls)
    obj.path = path
    return obj


class FakeRepo:
    """Serves an in-memory HF dataset repo laid out like the real hub cache.

    Files are symlinks under ``snapshots/<sha>/`` pointing at ``blobs/<n>``, which is
    what ``hf_hub_download`` and ``snapshot_download`` return paths into.
    """

    def __init__(
        self,
        tmp_path: Path,
        files: dict[str, bytes],
        *,
        private: bool = False,
        gated: bool = False,
    ):
        self.cache = tmp_path / "hf-hub"
        self.snapshot_dir = self.cache / "snapshots" / SHA
        (self.cache / "blobs").mkdir(parents=True)
        for i, (rel, data) in enumerate(files.items()):
            blob = self.cache / "blobs" / f"blob{i}"
            blob.write_bytes(data)
            link = self.snapshot_dir / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(blob)
            except (OSError, NotImplementedError):
                pytest.skip("symlinks are not available on this platform")
        siblings = [SimpleNamespace(rfilename=rel) for rel in files]
        self.info = SimpleNamespace(
            sha=SHA, private=private, gated=gated, siblings=siblings
        )
        self.download_calls: list[str] = []
        # Simulates snapshot_download falling back to a partial snapshot folder.
        self.snapshot_drops: list[str] = []

    def paths_info(self, repo_id, paths, *, repo_type, revision):
        out = []
        for rel in paths:
            target = self.snapshot_dir / rel
            if target.is_file():
                out.append(_item(RepoFile, rel))
            elif target.is_dir():
                out.append(_item(RepoFolder, rel))
        return out

    def download(self, *, repo_id, filename, repo_type, revision):
        self.download_calls.append(filename)
        target = self.snapshot_dir / filename
        if not target.is_file():
            raise EntryNotFoundError(f"{filename} missing")
        return str(target)

    def tree(self, repo_id, path_in_repo=None, *, recursive, repo_type, revision):
        base = self.snapshot_dir / (path_in_repo or "")
        if not base.is_dir():
            raise EntryNotFoundError(f"{path_in_repo} missing")
        return [
            _item(
                RepoFile if p.is_file() else RepoFolder,
                p.relative_to(self.snapshot_dir).as_posix(),
            )
            for p in sorted(base.rglob("*"))
        ]

    def snapshot(self, repo_id, *, repo_type, revision, allow_patterns):
        for rel in self.snapshot_drops:
            (self.snapshot_dir / rel).unlink(missing_ok=True)
        return str(self.snapshot_dir)

    def patches(self):
        return (
            patch("huggingface_hub.HfApi.dataset_info", return_value=self.info),
            patch("huggingface_hub.HfApi.get_paths_info", side_effect=self.paths_info),
            patch("huggingface_hub.HfApi.auth_check", return_value=None),
            patch("huggingface_hub.hf_hub_download", side_effect=self.download),
            patch("huggingface_hub.snapshot_download", side_effect=self.snapshot),
            patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
            patch("huggingface_hub.HfApi.list_repo_tree", side_effect=self.tree),
        )


async def _probe(repo: FakeRepo, ref: str):
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        resolver = HFSourceResolver()
        probe = await resolver.probe(RemoteRef(ref=ref))
        return resolver, probe, await resolver.materialize(probe)


def test_registered() -> None:
    assert (
        plugins.get_class(PluginType.DATASET_SOURCE_RESOLVER, "hf") is HFSourceResolver
    )


@pytest.mark.asyncio
async def test_file_ref_with_sidecar(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {
        "data/t.jsonl": b'{"timestamp": 1}\n',
        "data/t.jsonl.aiperf.json": orjson.dumps({"aiperf_manifest_version": 1, "format": "mooncake_trace"}),
    })  # fmt: skip
    _, probe, source = await _probe(repo, "hf://org/repo/data/t.jsonl")
    assert probe.ref.resolved_version == SHA
    assert probe.pinned_ref == f"hf://datasets/org/repo@{SHA}/data/t.jsonl"
    assert probe.manifest.format == "mooncake_trace" and not probe.data_is_dir
    assert source.local_path.read_bytes() == b'{"timestamp": 1}\n'
    # The HF cache stays the single copy: the data path is the cache symlink.
    assert source.local_path.is_symlink()


@pytest.mark.asyncio
async def test_file_ref_without_sidecar_has_no_manifest(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    _, probe, source = await _probe(repo, "hf://org/repo/t.jsonl")
    assert probe.manifest is None and probe.manifest_sha256 is None
    assert [f.path for f in source.files] == ["t.jsonl"]


@pytest.mark.asyncio
async def test_repo_root_uses_aiperf_json_and_directory_data(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {
        "aiperf.json": orjson.dumps({"aiperf_manifest_version": 1, "format": "weka_trace", "path": "traces"}),
        "traces/a.json": b"{}", "traces/b.json": b"{}",
    })  # fmt: skip
    _, probe, source = await _probe(repo, "hf://datasets/org/repo")
    assert probe.data_relpath == "traces" and probe.data_is_dir
    assert [f.path for f in source.files] == ["traces/a.json", "traces/b.json"]


@pytest.mark.asyncio
async def test_folder_ref_uses_manifest_inside_folder(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {
        "sub/aiperf.json": orjson.dumps({"aiperf_manifest_version": 1, "format": "weka_trace", "path": "t.jsonl"}),
        "sub/t.jsonl": b"{}\n",
    })  # fmt: skip
    _, probe, source = await _probe(repo, "hf://org/repo/sub")
    assert probe.data_relpath == "sub/t.jsonl" and not probe.data_is_dir
    assert source.local_path.read_bytes() == b"{}\n"


@pytest.mark.asyncio
async def test_repo_without_manifest_is_manifest_invalid(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}"})
    with pytest.raises(RemoteSourceError, match="aiperf.json") as exc:
        await _probe(repo, "hf://org/repo")
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.asyncio
async def test_missing_path_is_not_found(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}"})
    with pytest.raises(RemoteSourceError) as exc:
        await _probe(repo, "hf://org/repo/nope.jsonl")
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_manifest_pointing_at_missing_path_is_invalid(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {
        "aiperf.json": orjson.dumps({"aiperf_manifest_version": 1, "format": "weka_trace", "path": "gone"}),
    })  # fmt: skip
    with pytest.raises(RemoteSourceError, match="does not exist") as exc:
        await _probe(repo, "hf://org/repo")
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.asyncio
async def test_manifest_symlink_escaping_hf_cache_is_refused(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    outside = tmp_path / "outside.json"
    outside.write_bytes(
        orjson.dumps({"aiperf_manifest_version": 1, "format": "mooncake_trace"})
    )
    (repo.snapshot_dir / "t.jsonl.aiperf.json").symlink_to(outside)
    with pytest.raises(RemoteSourceError, match="Hugging Face cache") as exc:
        await _probe(repo, "hf://org/repo/t.jsonl")
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.asyncio
async def test_oversize_manifest_is_refused(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {
        "t.jsonl": b"{}\n",
        "t.jsonl.aiperf.json": b" " * ((1 << 20) + 1),
    })  # fmt: skip
    with pytest.raises(RemoteSourceError, match="larger than 1 MiB") as exc:
        await _probe(repo, "hf://org/repo/t.jsonl")
    assert exc.value.kind == RemoteErrorKind.MANIFEST_INVALID


@pytest.mark.parametrize(
    "error,token,kind",
    [
        (GatedRepoError("gated", response=Mock()), "hf_x", RemoteErrorKind.GATED),
        (RevisionNotFoundError("rev", response=Mock()), "hf_x", RemoteErrorKind.REVISION_NOT_FOUND),
        (RepositoryNotFoundError("401", response=Mock()), None, RemoteErrorKind.AUTH_MISSING),
        (RepositoryNotFoundError("404", response=Mock()), "hf_x", RemoteErrorKind.NOT_FOUND),
        (ConnectionError("reset"), "hf_x", RemoteErrorKind.TRANSIENT),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_error_mapping(error, token, kind) -> None:
    with (
        patch("huggingface_hub.HfApi.dataset_info", side_effect=error),
        patch("huggingface_hub.get_token", return_value=token),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    assert exc.value.kind == kind


@pytest.mark.asyncio
async def test_gated_repo_message_links_terms_page() -> None:
    info = SimpleNamespace(sha=SHA, private=False, gated="manual")
    with (
        patch("huggingface_hub.HfApi.dataset_info", return_value=info),
        patch(
            "huggingface_hub.HfApi.auth_check",
            side_effect=GatedRepoError("g", response=Mock()),
        ),
        pytest.raises(
            RemoteSourceError, match="https://huggingface.co/datasets/org/repo"
        ),
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))


@pytest.mark.asyncio
async def test_error_text_never_contains_token() -> None:
    boom = ConnectionError("failed: Authorization: Bearer hf_secret_token_value")
    with (
        patch("huggingface_hub.HfApi.dataset_info", side_effect=boom),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    # We do not echo library exception text back at all beyond its type.
    assert "hf_secret_token_value" not in str(exc.value)


@pytest.mark.asyncio
async def test_pinned_commit_that_vanished_reports_change() -> None:
    gone = RevisionNotFoundError("rev", response=Mock())
    with (
        patch("huggingface_hub.HfApi.dataset_info", side_effect=gone),
        pytest.raises(RemoteSourceError, match="changed after") as exc,
    ):
        await HFSourceResolver().probe(
            RemoteRef(ref="hf://org/repo/t.jsonl").with_version(SHA)
        )
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_private_repo_needs_auth(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"}, private=True)
    _, probe, _ = await _probe(repo, "hf://org/repo/t.jsonl")
    assert probe.needs_auth is True


@pytest.mark.asyncio
async def test_read_prefix_targets_pinned_resolve_url(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"data/t.jsonl": b"{}\n"})
    resolver, probe, _ = await _probe(repo, "hf://org/repo/data/t.jsonl")
    with patch(
        "aiperf.dataset.remote.hf.read_prefix", AsyncMock(return_value=(b"{}\n", True))
    ) as rp:
        assert await resolver.read_prefix(probe, 1024) == (b"{}\n", True)
    url = rp.call_args.args[1]
    assert f"/datasets/org/repo/resolve/{SHA}/data/t.jsonl" in url


@pytest.mark.asyncio
async def test_service_materialize_indexes_hf_symlink(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo/t.jsonl"))
        source = await materialize_probe(probe)
        # A second call is served from the index without another download.
        calls = len(repo.download_calls)
        again = await materialize_probe(probe)
    assert again.local_path == source.local_path
    assert len(repo.download_calls) == calls


class TransportError(Exception):
    """Stands in for httpx.TransportError, which the resolver matches by name through the MRO."""


class _ReadError(TransportError):
    pass


def _http_error(status: int) -> HfHubHTTPError:
    return HfHubHTTPError(f"http {status}", response=Mock(status_code=status))


@pytest.mark.asyncio
async def test_transport_error_subclasses_are_transient() -> None:
    with (
        patch(
            "huggingface_hub.HfApi.dataset_info", side_effect=_ReadError("peer reset")
        ),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    assert exc.value.kind == RemoteErrorKind.TRANSIENT
    assert exc.value.connection_failed is True


@pytest.mark.asyncio
async def test_http_503_is_transient_without_connection_failure() -> None:
    with (
        patch("huggingface_hub.HfApi.dataset_info", side_effect=_http_error(503)),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    assert exc.value.kind == RemoteErrorKind.TRANSIENT
    assert exc.value.connection_failed is False


@pytest.mark.parametrize(
    "status,token,kind",
    [
        (401, None, RemoteErrorKind.AUTH_MISSING),
        (401, "hf_x", RemoteErrorKind.AUTH_DENIED),
        (403, None, RemoteErrorKind.AUTH_DENIED),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_http_auth_statuses(status, token, kind) -> None:
    with (
        patch("huggingface_hub.HfApi.dataset_info", side_effect=_http_error(status)),
        patch("huggingface_hub.get_token", return_value=token),
        pytest.raises(RemoteSourceError) as exc,
    ):
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    assert exc.value.kind == kind


def _local_miss(cause: BaseException | None) -> LocalEntryNotFoundError:
    err = LocalEntryNotFoundError("cannot reach the Hub and nothing is cached")
    err.__cause__ = cause
    return err


@pytest.mark.parametrize(
    "cause,token,kind",
    [
        (None, "hf_x", RemoteErrorKind.TRANSIENT),
        (_http_error(429), "hf_x", RemoteErrorKind.TRANSIENT),
        (_ReadError("reset"), "hf_x", RemoteErrorKind.TRANSIENT),
        (_http_error(401), None, RemoteErrorKind.AUTH_MISSING),
        (_http_error(403), "hf_x", RemoteErrorKind.AUTH_DENIED),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_failed_sidecar_fetch_is_not_a_missing_sidecar(
    tmp_path: Path, cause, token, kind
) -> None:
    """hf_hub_download reports a failed metadata call as LocalEntryNotFoundError, not as a 404."""
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    p = repo.patches()
    with (
        p[0], p[1], p[2], p[4], p[5], p[6],
        patch("huggingface_hub.hf_hub_download", side_effect=_local_miss(cause)),
        patch("huggingface_hub.get_token", return_value=token),
        pytest.raises(RemoteSourceError) as exc,
    ):  # fmt: skip
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo/t.jsonl"))
    assert exc.value.kind == kind


@pytest.mark.asyncio
async def test_failed_root_manifest_fetch_is_transient(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"aiperf.json": b"{}"})
    p = repo.patches()
    with (
        p[0], p[1], p[2], p[4], p[5], p[6],
        patch("huggingface_hub.hf_hub_download", side_effect=_local_miss(_http_error(429))),
        pytest.raises(RemoteSourceError) as exc,
    ):  # fmt: skip
        await HFSourceResolver().probe(RemoteRef(ref="hf://org/repo"))
    assert exc.value.kind == RemoteErrorKind.TRANSIENT


_DIR_MANIFEST = orjson.dumps(
    {"aiperf_manifest_version": 1, "format": "weka_trace", "path": "traces"}
)


@pytest.mark.asyncio
async def test_partial_snapshot_is_rejected_and_not_indexed(tmp_path: Path) -> None:
    repo = FakeRepo(
        tmp_path,
        {"aiperf.json": _DIR_MANIFEST, "traces/a.json": b"{}", "traces/b.json": b"{}"},
    )
    repo.snapshot_drops = ["traces/b.json"]
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo"))
        with pytest.raises(RemoteSourceError, match="traces/b.json") as exc:
            await materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.TRANSIENT
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_empty_snapshot_folder_is_rejected(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"aiperf.json": _DIR_MANIFEST, "traces/a.json": b"{}"})
    repo.info.siblings = [SimpleNamespace(rfilename="aiperf.json")]
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo"))
        with pytest.raises(RemoteSourceError, match="no files"):
            await materialize_probe(probe)
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_disk_full_is_not_retried_and_names_cache_dir(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo/t.jsonl"))
        full = Mock(side_effect=OSError(errno.ENOSPC, "No space left on device"))
        with (
            patch("huggingface_hub.hf_hub_download", full),
            pytest.raises(RemoteSourceError, match=re.escape(str(repo.cache))) as exc,
        ):
            await materialize_probe(probe)
    assert exc.value.kind == RemoteErrorKind.NOT_FOUND
    assert full.call_count == 1
    assert read_index(probe.ref) is None


@pytest.mark.asyncio
async def test_partial_snapshot_under_nested_folder_is_rejected(tmp_path: Path) -> None:
    manifest = orjson.dumps(
        {"aiperf_manifest_version": 1, "format": "weka_trace", "path": "a/b"}
    )
    repo = FakeRepo(
        tmp_path, {"aiperf.json": manifest, "a/b/x.json": b"{}", "a/b/y.json": b"{}"}
    )
    repo.snapshot_drops = ["a/b/y.json"]
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5], p[6]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo"))
        with pytest.raises(RemoteSourceError, match="a/b/y.json"):
            await materialize_probe(probe)
