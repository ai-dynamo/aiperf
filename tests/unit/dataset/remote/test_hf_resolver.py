# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import orjson
import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.hf_api import RepoFile, RepoFolder

from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.errors import RemoteErrorKind, RemoteSourceError
from aiperf.dataset.remote.hf import HFSourceResolver
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
            link.symlink_to(blob)
        self.info = SimpleNamespace(sha=SHA, private=private, gated=gated)
        self.download_calls: list[str] = []

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

    def snapshot(self, repo_id, *, repo_type, revision, allow_patterns):
        return str(self.snapshot_dir)

    def patches(self):
        return (
            patch("huggingface_hub.HfApi.dataset_info", return_value=self.info),
            patch("huggingface_hub.HfApi.get_paths_info", side_effect=self.paths_info),
            patch("huggingface_hub.HfApi.auth_check", return_value=None),
            patch("huggingface_hub.hf_hub_download", side_effect=self.download),
            patch("huggingface_hub.snapshot_download", side_effect=self.snapshot),
            patch("huggingface_hub.constants.HF_HUB_CACHE", str(self.cache)),
        )


async def _probe(repo: FakeRepo, ref: str):
    p = repo.patches()
    with p[0], p[1], p[2], p[3], p[4], p[5]:
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
async def test_pinned_ref_that_moved_reports_change(tmp_path: Path) -> None:
    repo = FakeRepo(tmp_path, {"t.jsonl": b"{}\n"})
    repo.info = SimpleNamespace(sha=OTHER_SHA, private=False, gated=False)
    p = repo.patches()
    with (
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
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
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        probe = await probe_ref(RemoteRef(ref="hf://org/repo/t.jsonl"))
        source = await materialize_probe(probe)
        # A second call is served from the index without another download.
        calls = len(repo.download_calls)
        again = await materialize_probe(probe)
    assert again.local_path == source.local_path
    assert len(repo.download_calls) == calls
