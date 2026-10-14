# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolver for hf:// references to Hugging Face dataset repos."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar

from aiperf.common.redact import redact_string
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.cache import local_fs_error
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
    changed_after_pin,
)
from aiperf.dataset.remote.http_fetch import open_session, read_prefix
from aiperf.dataset.remote.index import describe_files
from aiperf.dataset.remote.manifest import (
    ROOT_MANIFEST_NAME,
    SIDECAR_SUFFIX,
    AIPerfManifest,
    parse_manifest,
)
from aiperf.dataset.remote.models import ProbeResult, ResolvedSource

T = TypeVar("T")
_MANIFEST_MAX_BYTES = 1 << 20


def _error(
    ref: RemoteRef,
    *,
    kind: RemoteErrorKind,
    stage: RemoteStage,
    what: str,
    hint: str,
) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref.display, kind=kind, stage=stage, what=what, hint=hint
    )


def _http_error(
    ref: RemoteRef, e: BaseException, stage: RemoteStage, *, has_token: bool
) -> RemoteSourceError:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 401 and not has_token:
        return _error(
            ref, kind=RemoteErrorKind.AUTH_MISSING, stage=stage,
            what="Hugging Face requires authentication (HTTP 401)",
            hint="Set HF_TOKEN or run `hf auth login`, then retry",
        )  # fmt: skip
    if status in (401, 403):
        return _error(
            ref, kind=RemoteErrorKind.AUTH_DENIED, stage=stage,
            what=f"Hugging Face denied access (HTTP {status})",
            hint="Check your token's permissions",
        )  # fmt: skip
    if status == 429 or (isinstance(status, int) and status >= 500):
        return _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=f"Hugging Face is unavailable (HTTP {status})", hint="Retry later",
        )  # fmt: skip
    return _error(
        ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what=f"Hugging Face returned HTTP {status}", hint="Check the reference",
    )  # fmt: skip


def _is_transport_error(e: BaseException) -> bool:
    # huggingface_hub>=1.0 uses httpx, whose transport errors are not OSError
    # subclasses; matching by name avoids importing httpx.
    return any(cls.__name__ == "TransportError" for cls in type(e).__mro__)


def _io_error(
    ref: RemoteRef, e: Exception, stage: RemoteStage
) -> RemoteSourceError | None:
    from huggingface_hub import constants

    local = local_fs_error(
        e, ref_display=ref.display, stage=stage, env_var="HF_HOME",
        cache=f"the Hugging Face cache at {constants.HF_HUB_CACHE}",
    )  # fmt: skip
    if local is not None:
        return local
    if isinstance(e, OSError | TimeoutError) or _is_transport_error(e):
        return _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=redact_string(f"the connection failed ({type(e).__name__}: {e})"),
            hint="Check network access and retry",
        )  # fmt: skip
    return None


def _translate(
    ref: RemoteRef, e: Exception, stage: RemoteStage, *, has_token: bool
) -> RemoteSourceError | None:
    """The user-facing error for a huggingface_hub failure; None when it is not one we map."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    repo = ref.parsed.location
    # Order matters: GatedRepoError subclasses RepositoryNotFoundError, and every
    # one of these subclasses HfHubHTTPError. LocalEntryNotFoundError subclasses
    # EntryNotFoundError but means the metadata call failed (network, 429, 5xx, 403),
    # not that the file is missing. RemoteEntryNotFoundError (hub>=1.0) subclasses
    # EntryNotFoundError, so one check covers both.
    if isinstance(e, LocalEntryNotFoundError):
        cause = e.__cause__
        mapped = (
            _translate(ref, cause, stage, has_token=has_token)
            if isinstance(cause, Exception)
            else None
        )
        return mapped or _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what="Hugging Face could not be reached",
            hint="Check network access and retry",
        )  # fmt: skip
    if isinstance(e, GatedRepoError):
        return _error(
            ref, kind=RemoteErrorKind.GATED, stage=stage,
            what=f"{repo} is gated and your account has not accepted its terms",
            hint=f"Accept them at https://huggingface.co/datasets/{repo}, then retry",
        )  # fmt: skip
    if isinstance(e, RevisionNotFoundError):
        if ref.resolved_version is not None:
            # A pinned commit that no longer resolves was force-pushed away.
            return changed_after_pin(ref.display, stage, "the repo")
        return _error(
            ref, kind=RemoteErrorKind.REVISION_NOT_FOUND, stage=stage,
            what=f"{repo} has no revision '{ref.parsed.revision}'",
            hint="Check the @revision in the reference",
        )  # fmt: skip
    if isinstance(e, RepositoryNotFoundError):
        # HF answers both "missing" and "private, not yours" with this error.
        if not has_token:
            return _error(
                ref, kind=RemoteErrorKind.AUTH_MISSING, stage=stage,
                what=f"{repo} was not found; it may be private",
                hint="Set HF_TOKEN or run `hf auth login`, then retry",
            )  # fmt: skip
        return _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
            what=f"{repo} was not found, or your token can't access it",
            hint="Check the repo name and your token's permissions",
        )  # fmt: skip
    if isinstance(e, EntryNotFoundError):
        return _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
            what=f"a file it needs is missing from {repo}",
            hint="Check the path in the reference",
        )  # fmt: skip
    if isinstance(e, HfHubHTTPError):
        return _http_error(ref, e, stage, has_token=has_token)
    return _io_error(ref, e, stage)


async def _hf_call(
    ref: RemoteRef,
    fn: Callable[[], T],
    *,
    stage: RemoteStage = RemoteStage.ACCESS,
    missing_ok: bool = False,
) -> T | None:
    """Run a blocking huggingface_hub call in a thread, mapping its errors to ``RemoteSourceError``.

    ``missing_ok`` returns None only for a genuine remote 404; a failed metadata
    call (``LocalEntryNotFoundError``) is an error, not a missing file.
    """
    from huggingface_hub import get_token
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return await asyncio.to_thread(fn)
    except Exception as e:
        if (
            missing_ok
            and isinstance(e, EntryNotFoundError)
            and not isinstance(e, LocalEntryNotFoundError)
        ):
            return None
        # get_token reads files and env; keep it off the event loop.
        has_token = await asyncio.to_thread(get_token) is not None
        mapped = _translate(ref, e, stage, has_token=has_token)
        if mapped is None:
            raise
        raise mapped from e


def _join(root: str, rel: str) -> str:
    return str(PurePosixPath(root) / rel) if root else rel


def _read_hf_cached(ref: RemoteRef, local: str) -> bytes:
    """Read a small file the HF client placed in its hub cache.

    HF cache entries are symlinks into ``blobs/``, which ``safe_read_template_path``
    rejects, so containment is checked on the resolved target instead.
    """
    from huggingface_hub import constants

    cache_root = Path(constants.HF_HUB_CACHE).resolve()
    target = Path(local).resolve()
    if not target.is_relative_to(cache_root):
        raise _error(
            ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=RemoteStage.ACCESS,
            what="a manifest file resolves outside the Hugging Face cache",
            hint="Clear the Hugging Face cache for this repo and retry",
        )  # fmt: skip
    if target.stat().st_size > _MANIFEST_MAX_BYTES:
        raise _error(
            ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=RemoteStage.ACCESS,
            what="its manifest is larger than 1 MiB",
            hint="A manifest is a small JSON document; check the file in the repo",
        )  # fmt: skip
    return target.read_bytes()


class _RepoReader:
    """Metadata and small-file reads against one pinned commit of a dataset repo."""

    def __init__(self, ref: RemoteRef, api: Any, sha: str) -> None:
        self._ref = ref
        self._api = api
        self._sha = sha

    async def paths(self, items: list[str]) -> list[Any]:
        repo, sha = self._ref.parsed.location, self._sha
        found = await _hf_call(
            self._ref,
            lambda: self._api.get_paths_info(
                repo, items, repo_type="dataset", revision=sha
            ),
        )
        return list(found or [])

    async def fetch(self, relpath: str) -> bytes | None:
        """Bytes of a small repo file; None when the repo has no such file."""
        from huggingface_hub import hf_hub_download

        repo, sha = self._ref.parsed.location, self._sha
        local = await _hf_call(
            self._ref,
            lambda: hf_hub_download(
                repo_id=repo, filename=relpath, repo_type="dataset", revision=sha
            ),
            missing_ok=True,
        )
        if local is None:
            return None
        return await asyncio.to_thread(_read_hf_cached, self._ref, local)


def _parse(
    ref: RemoteRef, raw: bytes, kind: Literal["root", "sidecar"]
) -> tuple[AIPerfManifest, str]:
    manifest = parse_manifest(raw, ref_display=ref.display, kind=kind)
    return manifest, hashlib.sha256(raw).hexdigest()


async def _pin(ref: RemoteRef, api: Any) -> tuple[str, bool]:
    """Resolve the commit SHA of ``ref``; also whether reading it needs credentials."""
    parsed = ref.parsed
    repo = parsed.location
    info: Any = await _hf_call(
        ref,
        lambda: api.dataset_info(
            repo, revision=ref.resolved_version or parsed.revision
        ),
    )
    if info.gated:
        await _hf_call(ref, lambda: api.auth_check(repo, repo_type="dataset"))
    sha: str = info.sha
    return sha, bool(info.private) or bool(info.gated)


async def _manifest_for_root(
    ref: RemoteRef, reader: _RepoReader, root: str
) -> tuple[AIPerfManifest, str, str, bool]:
    """Manifest, its digest, and the data path and kind for a repo root or folder ref."""
    from huggingface_hub.hf_api import RepoFolder

    repo, path = ref.parsed.location, ref.parsed.path
    raw = await reader.fetch(_join(root, ROOT_MANIFEST_NAME))
    if raw is None:
        where = f"'{path}'" if path else "its root"
        raise _error(
            ref,
            kind=RemoteErrorKind.NOT_FOUND
            if path
            else RemoteErrorKind.MANIFEST_INVALID,
            stage=RemoteStage.ACCESS,
            what=f"{repo} has no file at {where} and no {ROOT_MANIFEST_NAME} describing it",
            hint="Point at a data file (hf://<org>/<repo>/<file>) or ask the publisher to add aiperf.json",
        )
    manifest, manifest_sha = _parse(ref, raw, "root")
    data_relpath = _join(root, manifest.path or "")
    found = await reader.paths([data_relpath])
    if not found:
        raise _error(
            ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=RemoteStage.ACCESS,
            what=f"its manifest points at '{manifest.path}', which does not exist",
            hint="Ask the publisher to fix aiperf.json",
        )  # fmt: skip
    return manifest, manifest_sha, data_relpath, isinstance(found[0], RepoFolder)


async def _expected_files(ref: RemoteRef, relpath: str) -> list[str]:
    """Repo paths under the folder ``relpath`` at the pinned commit."""
    from huggingface_hub import HfApi

    repo, sha = ref.parsed.location, ref.resolved_version
    info: Any = await _hf_call(
        ref,
        lambda: HfApi().dataset_info(repo, revision=sha),
        stage=RemoteStage.MATERIALIZE,
    )
    prefix = f"{relpath}/"
    expected = [
        sibling.rfilename
        for sibling in info.siblings or []
        if sibling.rfilename.startswith(prefix)
    ]
    if not expected:
        raise _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=RemoteStage.MATERIALIZE,
            what=f"{repo} has no files under '{relpath}' at the pinned version",
            hint="Ask the publisher to fix aiperf.json",
        )  # fmt: skip
    return expected


def _require_complete(
    ref: RemoteRef, snapshot: Path, local: Path, expected: list[str]
) -> None:
    missing = [rel for rel in expected if not (snapshot / rel).is_file()]
    if local.is_dir() and not missing:
        return
    shown = ", ".join(missing[:3]) if missing else str(local)
    raise _error(
        ref, kind=RemoteErrorKind.TRANSIENT, stage=RemoteStage.MATERIALIZE,
        what=f"the download is incomplete (missing {shown})",
        hint="Check network access and retry",
    )  # fmt: skip


class HFSourceResolver:
    """Resolve hf:// references to Hugging Face dataset repos."""

    async def probe(self, ref: RemoteRef) -> ProbeResult:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile

        parsed = ref.parsed
        api = HfApi()
        sha, needs_auth = await _pin(ref, api)
        reader = _RepoReader(ref, api, sha)

        manifest: AIPerfManifest | None = None
        manifest_sha: str | None = None
        data_relpath = ""
        data_is_dir = False
        root: str | None = "" if parsed.path is None else None
        if parsed.path is not None:
            found = await reader.paths([parsed.path])
            if found and isinstance(found[0], RepoFile):
                data_relpath = parsed.path
                raw = await reader.fetch(parsed.path + SIDECAR_SUFFIX)
                if raw is not None:
                    manifest, manifest_sha = _parse(ref, raw, "sidecar")
            else:
                root = parsed.path
        if root is not None:
            (
                manifest,
                manifest_sha,
                data_relpath,
                data_is_dir,
            ) = await _manifest_for_root(ref, reader, root)
        repo = parsed.location
        return ProbeResult(
            ref=ref.with_version(sha),
            pinned_ref=f"hf://datasets/{repo}@{sha}"
            + (f"/{parsed.path}" if parsed.path else ""),
            data_relpath=data_relpath,
            data_is_dir=data_is_dir,
            filename=PurePosixPath(data_relpath).name,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            needs_auth=needs_auth,
        )

    async def read_prefix(self, probe: ProbeResult, nbytes: int) -> tuple[bytes, bool]:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        url = hf_hub_url(
            probe.ref.parsed.location, probe.data_relpath or "",
            repo_type="dataset", revision=probe.ref.resolved_version,
        )  # fmt: skip
        headers = await asyncio.to_thread(build_hf_headers)
        async with open_session() as session:
            return await read_prefix(
                session,
                url,
                nbytes,
                headers=headers,
                ref_display=probe.ref.display,
            )

    async def materialize(self, probe: ProbeResult) -> ResolvedSource:
        from huggingface_hub import hf_hub_download, snapshot_download

        ref = probe.ref
        repo, sha = ref.parsed.location, ref.resolved_version
        relpath = probe.data_relpath or ""
        stage = RemoteStage.MATERIALIZE
        if probe.data_is_dir:
            # snapshot_download returns the existing snapshot folder, unverified, when
            # its own repo-info call fails, so the expected file set comes from our own.
            expected = await _expected_files(ref, relpath)
            root = await _hf_call(
                ref,
                lambda: snapshot_download(
                    repo,
                    repo_type="dataset",
                    revision=sha,
                    allow_patterns=[f"{relpath}/**"],
                ),
                stage=stage,
            )
            snapshot = Path(str(root))
            local = snapshot / relpath
            await asyncio.to_thread(_require_complete, ref, snapshot, local, expected)
        else:
            downloaded = await _hf_call(
                ref,
                lambda: hf_hub_download(
                    repo_id=repo, filename=relpath, repo_type="dataset", revision=sha
                ),
                stage=stage,
            )
            local = Path(str(downloaded))
        files = await asyncio.to_thread(describe_files, local, relpath)
        return ResolvedSource(probe=probe, local_path=local, files=files)
