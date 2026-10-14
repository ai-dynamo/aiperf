# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolver for s3:// references, using botocore's default credential chain."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import threading
from collections.abc import Awaitable, Callable, Iterable, Iterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, NamedTuple, NoReturn, TypeVar
from urllib.parse import quote

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.optional_dependencies import (
    AWS_EXTRA,
    install_optional_dependency_hint,
)
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote.cache import (
    digest,
    entry_dir,
    remote_cache_dir,
    write_atomically,
)
from aiperf.dataset.remote.errors import (
    RemoteErrorKind,
    RemoteSourceError,
    RemoteStage,
    changed_after_pin,
)
from aiperf.dataset.remote.http_fetch import content_range_total
from aiperf.dataset.remote.manifest import (
    ROOT_MANIFEST_NAME,
    SIDECAR_SUFFIX,
    AIPerfManifest,
    parse_manifest,
)
from aiperf.dataset.remote.models import ProbeResult, RemoteFile, ResolvedSource

T = TypeVar("T")

_CHUNK = 1 << 20
_CONNECT_TIMEOUT_S = 10
_READ_TIMEOUT_S = 30
_MANIFEST_MAX_BYTES = 1 << 20
# Local filesystem failures a retry cannot fix (and re-downloading GBs would not help).
_LOCAL_FS_ERRNOS = frozenset(
    {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM}
)
# HEAD responses carry no body, so their error code is the bare HTTP status.
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_AUTH_MISSING_CODES = frozenset(
    {"ExpiredToken", "ExpiredTokenException", "InvalidToken", "TokenRefreshRequired"}
)
_AUTH_DENIED_CODES = frozenset(
    {"AccessDenied", "403", "Forbidden", "AllAccessDisabled", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
)  # fmt: skip
_TRANSIENT_CODES = frozenset(
    {"SlowDown", "InternalError", "ServiceUnavailable", "RequestTimeout", "Throttling", "ThrottlingException"}
)  # fmt: skip


class _Object(NamedTuple):
    """One S3 object that makes up the data, as listed when it was pinned."""

    key: str
    etag: str
    version_id: str | None


def _error(
    ref: RemoteRef, *, kind: RemoteErrorKind, stage: RemoteStage, what: str, hint: str
) -> RemoteSourceError:
    return RemoteSourceError(
        ref=ref.display, kind=kind, stage=stage, what=what, hint=hint
    )


def _error_code(exc: BaseException) -> tuple[str, int | None]:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    return code, response.get("ResponseMetadata", {}).get("HTTPStatusCode")


def _version_id(response: dict[str, Any]) -> str | None:
    vid = response.get("VersionId")
    # A bucket with versioning suspended reports the literal "null".
    return vid if vid and vid != "null" else None


def _object_version(head: dict[str, Any]) -> str:
    vid = _version_id(head)
    return f"v:{vid}" if vid else f"etag:{head['ETag']}"


def _version_params(version: str | None) -> dict[str, str]:
    if version and version.startswith("v:"):
        return {"VersionId": version[len("v:") :]}
    if version and version.startswith("etag:"):
        return {"IfMatch": version[len("etag:") :]}
    return {}


def _listing_version(manifest_sha: str | None, objects: list[_Object]) -> str:
    listed = "".join(f"{o.key}\0{o.etag}\0" for o in objects)
    return "list:" + digest((manifest_sha or "") + listed, 32)


def _object_params(bucket: str, obj: _Object) -> dict[str, str]:
    params = {"Bucket": bucket, "Key": obj.key, "IfMatch": obj.etag}
    if obj.version_id:
        params["VersionId"] = obj.version_id
    return params


def _join(root: str, rel: str) -> str:
    return str(PurePosixPath(root) / rel) if root else rel


def _is_safe_relpath(rel: str) -> bool:
    """Whether a key below the data prefix maps to a file inside the cache entry.

    Checked per segment. ``:`` is a legal file-name character on POSIX, but on
    Windows a drive in any later segment (``x/D:y``) resets the joined path, and any
    ``:`` names an NTFS alternate data stream.
    """
    return all(
        part not in ("", ".", "..")
        and "\\" not in part
        and not (IS_WINDOWS and (":" in part or PureWindowsPath(part).anchor))
        for part in rel.split("/")
    )


def _is_within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _revision_missing(ref: RemoteRef, stage: RemoteStage) -> RemoteSourceError:
    if ref.resolved_version is not None:
        return changed_after_pin(ref.display, stage, "the object")
    return _error(
        ref, kind=RemoteErrorKind.REVISION_NOT_FOUND, stage=stage,
        what=f"S3 has no version '{ref.parsed.revision}' of that key",
        hint="Check the versionId in the reference",
    )  # fmt: skip


def _classify_client_error(
    ref: RemoteRef, exc: BaseException, stage: RemoteStage
) -> RemoteSourceError:
    code, status = _error_code(exc)
    if code in _AUTH_MISSING_CODES:
        return _error(
            ref, kind=RemoteErrorKind.AUTH_MISSING, stage=stage,
            what=f"the AWS credentials are expired or invalid ({code})",
            hint="Refresh them (e.g. `aws sso login`) and retry",
        )  # fmt: skip
    if code in _AUTH_DENIED_CODES or status == 403:
        return _error(
            ref, kind=RemoteErrorKind.AUTH_DENIED, stage=stage,
            what=f"S3 denied access ({code})",
            hint="Check the IAM permissions for this bucket and key; without s3:ListBucket, "
            "S3 also answers 403 for a key that does not exist",
        )  # fmt: skip
    if code == "PreconditionFailed" or status == 412:
        return changed_after_pin(ref.display, stage, "the object")
    if code == "NoSuchVersion" or (
        ref.parsed.revision and code in ("InvalidArgument", "400")
    ):
        return _revision_missing(ref, stage)
    if code in _TRANSIENT_CODES or status == 429 or (status or 0) >= 500:
        return _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=f"S3 is unavailable ({code})", hint="Retry later",
        )  # fmt: skip
    if code in _MISSING_CODES or code == "NoSuchBucket" or status == 404:
        return _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
            what=f"S3 has no such bucket or key ({code})", hint="Check the reference",
        )  # fmt: skip
    return _error(
        ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
        what=f"S3 rejected the request ({code or 'no error code'}, HTTP {status})",
        hint="Check the reference; temporary AWS credentials may also have expired",
    )  # fmt: skip


def _classify(
    ref: RemoteRef, exc: BaseException, stage: RemoteStage
) -> RemoteSourceError | None:
    """The user-facing error for a botocore or local-disk failure; None when it is not one we map."""
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        CredentialRetrievalError,
        HTTPClientError,
        IncompleteReadError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        SSOError,
        TokenRetrievalError,
    )
    from botocore.exceptions import ConnectionError as BotoConnectionError

    credential_errors = (
        NoCredentialsError, PartialCredentialsError, ProfileNotFound,
        CredentialRetrievalError, SSOError, TokenRetrievalError,
    )  # fmt: skip
    if isinstance(exc, credential_errors):
        # The botocore message is not shown: it can name profiles and credential files.
        return _error(
            ref, kind=RemoteErrorKind.AUTH_MISSING, stage=stage,
            what=f"no usable AWS credentials were found ({type(exc).__name__})",
            hint="Configure AWS credentials (environment variables, ~/.aws, `aws sso login`, "
            "or an instance/pod role) and retry",
        )  # fmt: skip
    # Checked before OSError: botocore's ReadTimeoutError is also an OSError.
    if isinstance(exc, HTTPClientError | BotoConnectionError | IncompleteReadError):
        return _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=f"the connection to S3 failed ({type(exc).__name__})",
            hint="Check network access and retry",
        )  # fmt: skip
    if isinstance(exc, ClientError):
        return _classify_client_error(ref, exc, stage)
    if isinstance(exc, OSError) and exc.errno in _LOCAL_FS_ERRNOS:
        return _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
            what=f"the remote dataset cache at {remote_cache_dir()} cannot be written ({exc.strerror})",
            hint="Free disk space or fix permissions there, or set AIPERF_DATASET_REMOTE_CACHE_DIR "
            "to another location, then retry",
        )  # fmt: skip
    if isinstance(exc, OSError | TimeoutError):
        return _error(
            ref, kind=RemoteErrorKind.TRANSIENT, stage=stage,
            what=f"the connection to S3 failed ({type(exc).__name__})",
            hint="Check network access and retry",
        )  # fmt: skip
    if isinstance(exc, BotoCoreError):
        return _error(
            ref, kind=RemoteErrorKind.NOT_FOUND, stage=stage,
            what=f"the S3 request failed ({type(exc).__name__}: {exc})",
            hint="Check the reference and your AWS configuration",
        )  # fmt: skip
    return None


def _is_missing(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    return isinstance(exc, ClientError) and _error_code(exc)[0] in _MISSING_CODES


def _reraise(ref: RemoteRef, exc: Exception, stage: RemoteStage) -> NoReturn:
    if not isinstance(exc, RemoteSourceError):
        mapped = _classify(ref, exc, stage)
        if mapped is not None:
            raise mapped from exc
    raise exc


async def _call(ref: RemoteRef, work: Awaitable[T], *, stage: RemoteStage) -> T:
    """Await ``work`` (a blocking botocore call in a thread), mapping its errors."""
    try:
        return await work
    except Exception as e:
        _reraise(ref, e, stage)


async def _call_if_present(
    ref: RemoteRef, work: Awaitable[T], *, stage: RemoteStage
) -> T | None:
    """Like ``_call``, but None when S3 says the key does not exist."""
    try:
        return await work
    except Exception as e:
        if not isinstance(e, RemoteSourceError) and _is_missing(e):
            return None
        _reraise(ref, e, stage)


class _Stopped(Exception):
    """The awaiting coroutine was cancelled, so the download thread gave up."""


def _until_stopped(stop: threading.Event, chunks: Iterable[bytes]) -> Iterator[bytes]:
    """``chunks``, checking ``stop`` before each read so a cancel never waits on one."""
    iterator = iter(chunks)
    while not stop.is_set():
        chunk = next(iterator, None)
        if chunk is None:
            return
        yield chunk
    raise _Stopped


def _download_object(
    client: Any, params: dict[str, str], dest: Path, stop: threading.Event
) -> tuple[str, int]:
    # The thread may start after the caller was cancelled; don't open a request then.
    if stop.is_set():
        raise _Stopped
    body = client.get_object(**params)["Body"]
    try:
        # write_atomically removes its partial file when the iterator raises.
        chunks = body.iter_chunks(chunk_size=_CHUNK)
        return write_atomically(_until_stopped(stop, chunks), dest)
    finally:
        body.close()


async def _in_stoppable_thread(fn: Callable[[threading.Event], T]) -> T:
    """Run ``fn`` in a worker thread; on cancellation, stop it and wait for it to exit.

    A cancelled ``asyncio.to_thread`` leaves its thread running, which would keep
    writing a partial file after the caller has moved on.
    """
    stop = threading.Event()
    task = asyncio.ensure_future(asyncio.to_thread(fn, stop))
    # After a cancellation nobody reads the outcome; retrieving it keeps asyncio quiet.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stop.set()
        await asyncio.wait([task])
        raise


def _create_client(ref: RemoteRef) -> Any:
    try:
        import botocore.session
        from botocore.config import Config
    except ImportError as e:
        raise _error(
            ref, kind=RemoteErrorKind.MISSING_DEPENDENCY, stage=RemoteStage.ACCESS,
            what="s3:// references need the optional botocore dependency",
            hint=install_optional_dependency_hint(AWS_EXTRA),
        ) from e  # fmt: skip
    # The service already retries transient failures; botocore's default of 5
    # attempts with 60 s timeouts under it would stall a dead endpoint for minutes.
    config = Config(
        connect_timeout=_CONNECT_TIMEOUT_S,
        read_timeout=_READ_TIMEOUT_S,
        retries={"mode": "standard", "max_attempts": 1},
    )
    return botocore.session.get_session().create_client("s3", config=config)


class S3SourceResolver:
    """Resolve s3:// references."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    async def _s3(self, ref: RemoteRef) -> Any:
        # Building a client resolves the credential chain, which can read files and
        # call the instance metadata service, so it stays off the event loop.
        if self._client is None:
            self._client = await _call(
                ref, asyncio.to_thread(_create_client, ref), stage=RemoteStage.ACCESS
            )
        return self._client

    async def _head(
        self, ref: RemoteRef, params: dict[str, str], stage: RemoteStage
    ) -> dict[str, Any] | None:
        client = await self._s3(ref)
        return await _call_if_present(
            ref, asyncio.to_thread(lambda: client.head_object(**params)), stage=stage
        )

    async def _manifest(
        self, ref: RemoteRef, bucket: str, key: str, *, kind: Literal["root", "sidecar"]
    ) -> tuple[AIPerfManifest, str] | None:
        """The parsed manifest at ``key`` and its sha256; None when there is none."""
        client = await self._s3(ref)

        def fetch() -> bytes:
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            try:
                return body.read(_MANIFEST_MAX_BYTES + 1)
            finally:
                body.close()

        try:
            raw = await _call_if_present(
                ref, asyncio.to_thread(fetch), stage=RemoteStage.ACCESS
            )
        except RemoteSourceError as e:
            # A readable object with no sidecar must not fail because the caller
            # lacks s3:ListBucket (S3 then answers 403 for the missing sidecar).
            if kind == "sidecar" and e.kind == RemoteErrorKind.AUTH_DENIED:
                return None
            raise
        if raw is None:
            return None
        if len(raw) > _MANIFEST_MAX_BYTES:
            raise _error(
                ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=RemoteStage.ACCESS,
                what=f"its manifest s3://{bucket}/{key} is larger than 1 MiB",
                hint="A manifest is a small JSON document; check the object",
            )  # fmt: skip
        manifest = parse_manifest(raw, ref_display=ref.display, kind=kind)
        return manifest, hashlib.sha256(raw).hexdigest()

    async def _list(
        self, ref: RemoteRef, bucket: str, prefix: str, stage: RemoteStage
    ) -> list[_Object]:
        """Every downloadable object under ``prefix``, sorted by key."""
        client = await self._s3(ref)

        def collect() -> list[_Object]:
            out: list[_Object] = []
            pages = client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix
            )
            for page in pages:
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(prefix) :]
                    if not rel or rel.endswith("/"):
                        continue  # a folder marker, not data
                    # Skipping a key would replay a subset of the dataset unnoticed.
                    if not _is_safe_relpath(rel):
                        raise _error(
                            ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=stage,
                            what=f"S3 key {obj['Key']!r} under the data prefix cannot be "
                            "stored as a local file on this system",
                            hint="Ask the publisher to rename that object",
                        )  # fmt: skip
                    out.append(_Object(obj["Key"], obj["ETag"], _version_id(obj)))
            return sorted(out)

        return await _call(ref, asyncio.to_thread(collect), stage=stage)

    async def _data_objects(
        self, ref: RemoteRef, bucket: str, data_key: str, *, is_dir: bool | None,
        stage: RemoteStage,
    ) -> tuple[list[_Object], bool]:  # fmt: skip
        """The objects behind a manifest's data path, and whether it is a directory.

        ``is_dir=None`` decides by checking for an object at ``data_key`` first.
        """
        if data_key and not is_dir:
            head = await self._head(ref, {"Bucket": bucket, "Key": data_key}, stage)
            if head is not None or is_dir is False:
                objects = [] if head is None else [_Object(data_key, head["ETag"], _version_id(head))]  # fmt: skip
                return objects, False
        prefix = f"{data_key}/" if data_key else ""
        return await self._list(ref, bucket, prefix, stage), True

    def _check_unchanged(self, ref: RemoteRef, version: str) -> None:
        if ref.resolved_version is not None and ref.resolved_version != version:
            raise changed_after_pin(ref.display, RemoteStage.ACCESS, "the data")

    async def probe(self, ref: RemoteRef) -> ProbeResult:
        parsed = ref.parsed
        bucket, key = parsed.location, parsed.path or ""
        if parsed.revision and parsed.is_container:
            raise _error(
                ref, kind=RemoteErrorKind.NOT_FOUND, stage=RemoteStage.ACCESS,
                what="a versionId names one object version, not a prefix",
                hint="Point at an object key, or drop ?versionId=",
            )  # fmt: skip
        if parsed.is_container:
            return await self._probe_root(ref, bucket, key)
        params = {"Bucket": bucket, "Key": key}
        if parsed.revision:
            params["VersionId"] = parsed.revision
        head = await self._head(ref, params, RemoteStage.ACCESS)
        if head is not None:
            return await self._probe_object(ref, bucket, key, head)
        if parsed.revision:
            raise _revision_missing(ref, RemoteStage.ACCESS)
        return await self._probe_root(ref, bucket, f"{key}/")

    async def _probe_object(
        self, ref: RemoteRef, bucket: str, key: str, head: dict[str, Any]
    ) -> ProbeResult:
        version = _object_version(head)
        self._check_unchanged(ref, version)
        found = await self._manifest(ref, bucket, key + SIDECAR_SUFFIX, kind="sidecar")
        manifest, manifest_sha = found if found is not None else (None, None)
        pinned = ref.with_version(version)
        vid = _version_id(head)
        return ProbeResult(
            ref=pinned,
            pinned_ref=f"s3://{bucket}/{key}?versionId={quote(vid, safe='')}"
            if vid
            else pinned.display,
            data_relpath=key,
            data_is_dir=False,
            filename=PurePosixPath(key).name,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            needs_auth=True,
        )

    async def _probe_root(self, ref: RemoteRef, bucket: str, root: str) -> ProbeResult:
        stage = RemoteStage.ACCESS
        found = await self._manifest(
            ref, bucket, root + ROOT_MANIFEST_NAME, kind="root"
        )
        if found is None:
            raise _error(
                ref,
                kind=RemoteErrorKind.MANIFEST_INVALID
                if ref.parsed.is_container
                else RemoteErrorKind.NOT_FOUND,
                stage=stage,
                what=f"there is no object at that key and no {ROOT_MANIFEST_NAME} under it",
                hint="Point at a data object, or add aiperf.json to the prefix",
            )
        manifest, manifest_sha = found
        data_key = _join(root, manifest.path or "")
        objects, data_is_dir = await self._data_objects(
            ref, bucket, data_key, is_dir=None, stage=stage
        )
        if not objects:
            raise _error(
                ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=stage,
                what=f"its manifest points at '{manifest.path}', which does not exist",
                hint="Ask the publisher to fix aiperf.json",
            )  # fmt: skip
        version = _listing_version(manifest_sha, objects)
        self._check_unchanged(ref, version)
        pinned = ref.with_version(version)
        return ProbeResult(
            ref=pinned, pinned_ref=pinned.display, data_relpath=data_key,
            data_is_dir=data_is_dir, filename=PurePosixPath(data_key).name,
            manifest=manifest, manifest_sha256=manifest_sha, needs_auth=True,
        )  # fmt: skip

    async def read_prefix(self, probe: ProbeResult, nbytes: int) -> tuple[bytes, bool]:
        if nbytes < 1:
            raise ValueError(f"nbytes must be at least 1, got {nbytes}")
        ref = probe.ref
        client = await self._s3(ref)
        params = {
            "Bucket": ref.parsed.location,
            "Key": probe.data_relpath or "",
            "Range": f"bytes=0-{nbytes - 1}",
            **_version_params(ref.resolved_version),
        }

        def fetch() -> tuple[bytes, bool]:
            from botocore.exceptions import ClientError

            try:
                response = client.get_object(**params)
            except ClientError as e:
                # S3 rejects any range on an empty object.
                if _error_code(e)[0] == "InvalidRange":
                    return b"", True
                raise
            body = response["Body"]
            try:
                # A store that ignores Range sends the whole object; read only the prefix.
                data = body.read(nbytes)
            finally:
                body.close()
            total = content_range_total(response.get("ContentRange"))
            if total is None:
                total = response.get("ContentLength")
            whole = total <= nbytes if total is not None else len(data) < nbytes
            return data, whole

        return await _call(ref, asyncio.to_thread(fetch), stage=RemoteStage.ACCESS)

    async def _download(
        self, ref: RemoteRef, params: dict[str, str], dest: Path
    ) -> RemoteFile:
        client = await self._s3(ref)
        sha, size = await _call(
            ref,
            _in_stoppable_thread(
                lambda stop: _download_object(client, params, dest, stop)
            ),
            stage=RemoteStage.MATERIALIZE,
        )
        return RemoteFile(path=params["Key"], sha256=sha, size_bytes=size)

    async def materialize(self, probe: ProbeResult) -> ResolvedSource:
        ref = probe.ref
        bucket = ref.parsed.location
        data_key = probe.data_relpath or ""
        version = ref.resolved_version or ""
        entry = entry_dir(ref)
        if not version.startswith("list:"):
            # One object pinned by VersionId or ETag; S3 enforces the pin on the GET.
            dest = entry / probe.filename
            params = {"Bucket": bucket, "Key": data_key, **_version_params(version)}
            return ResolvedSource(
                probe=probe, local_path=dest, files=[await self._download(ref, params, dest)]
            )  # fmt: skip
        # Manifest data is pinned by a digest of its listing: list again, compare,
        # then make each GET conditional on the listed ETag.
        stage = RemoteStage.MATERIALIZE
        objects, _ = await self._data_objects(
            ref, bucket, data_key, is_dir=probe.data_is_dir, stage=stage
        )
        if _listing_version(probe.manifest_sha256, objects) != version:
            raise changed_after_pin(ref.display, stage, "the data")
        local = entry / probe.filename
        prefix = f"{data_key}/" if data_key else ""
        files: list[RemoteFile] = []
        for obj in objects:
            dest = (
                local.joinpath(*obj.key[len(prefix) :].split("/"))
                if probe.data_is_dir
                else local
            )
            if not await asyncio.to_thread(_is_within, dest, entry):
                raise _error(
                    ref, kind=RemoteErrorKind.MANIFEST_INVALID, stage=stage,
                    what=f"S3 key {obj.key!r} would be stored outside the cache entry",
                    hint="Ask the publisher to rename that object",
                )  # fmt: skip
            files.append(await self._download(ref, _object_params(bucket, obj), dest))
        return ResolvedSource(probe=probe, local_path=local, files=files)
