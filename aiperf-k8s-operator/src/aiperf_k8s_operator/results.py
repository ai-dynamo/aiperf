# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor-confined durable storage for native Kubernetes results."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import secrets
import stat
import threading
import time
from collections.abc import AsyncIterable, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from jsonschema import Draft202012Validator

from .contract import _schema

MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
_IDENTITY_NAME = ".aiperf-result-identity.json"
_MANIFEST_NAME = "results-manifest.json"
_MANIFEST_VALIDATOR = Draft202012Validator(_schema("results-manifest.schema.json"))
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class UploadConflict(Exception):
    """The upload conflicts with already stored immutable content."""


class UploadInvalid(Exception):
    """The upload is malformed or does not match its declared metadata."""


class UploadTooLarge(Exception):
    """The upload exceeds a storage or wire-contract limit."""


class ResultsExpired(Exception):
    """The completed result incarnation expired and cannot be recreated."""


@dataclass(frozen=True)
class ResultIdentity:
    """The complete durable-storage identity of one Kubernetes result set."""

    namespace: str
    job_id: str
    run_id: str
    object_uid: str

    def __post_init__(self) -> None:
        for label, value, maximum in (
            ("namespace", self.namespace, 253),
            ("job ID", self.job_id, 253),
            ("run ID", self.run_id, 512),
            ("object UID", self.object_uid, 128),
        ):
            if (
                not value
                or len(value.encode()) > maximum
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
            ):
                raise UploadInvalid(f"{label} is invalid")


@dataclass(frozen=True)
class StorageLimits:
    """Hard admission and retention limits for staged and published results."""

    max_staging_runs: int = 128
    max_staging_bytes: int = 4 * 1024 * 1024 * 1024
    max_run_bytes: int = 1024 * 1024 * 1024
    max_artifacts_per_run: int = 256
    staging_ttl_seconds: float = 24 * 60 * 60
    max_published_runs: int = 1024
    max_published_bytes: int = 8 * 1024 * 1024 * 1024
    published_ttl_seconds: float = 7 * 24 * 60 * 60

    def __post_init__(self) -> None:
        if (
            min(
                self.max_staging_runs,
                self.max_staging_bytes,
                self.max_run_bytes,
                self.max_artifacts_per_run,
                self.max_published_runs,
                self.max_published_bytes,
            )
            <= 0
            or self.staging_ttl_seconds <= 0
            or self.published_ttl_seconds <= 0
        ):
            raise ValueError("result storage limits must be positive")


@dataclass
class RunRecord:
    """One indexed identity and its readiness-gated manifest."""

    status: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None


@dataclass
class ArtifactHandle:
    """A verified regular artifact retained by descriptor until streamed."""

    file: BinaryIO
    length: int
    content_type: str

    def read(self, size: int = -1) -> bytes:
        """Read from the verified descriptor."""
        return self.file.read(size)

    def close(self) -> None:
        """Close the verified descriptor."""
        self.file.close()

    def chunks(self, size: int = 64 * 1024) -> Iterator[bytes]:
        """Yield bounded chunks and close on every exit path."""
        try:
            while chunk := self.file.read(size):
                yield chunk
        finally:
            self.file.close()


@dataclass
class _PendingUpload:
    key: str
    parent_fd: int
    temporary_name: str
    destination_name: str
    output: BinaryIO
    digest: Any
    declared_digest: str
    declared_length: int
    length: int = 0
    released: bool = False


class ResultsIndex:
    """Index backed by a retained root and componentwise no-follow operations."""

    def __init__(
        self,
        root: Path,
        *,
        limits: StorageLimits | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        if os.name != "posix" or not all(
            hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        ):
            raise OSError("durable result storage requires POSIX no-follow descriptors")
        self._root = Path(root)
        self._root_fd = self._open_root(self._root)
        self._lock = threading.RLock()
        self._limits = limits or StorageLimits()
        self._now = now
        self._runs: dict[ResultIdentity, RunRecord] = {}
        self._active: dict[str, int] = {}
        self._reserved: dict[str, int] = {}
        try:
            self._staging_fd = self._ensure_dir(self._root_fd, ".staging")
            self._published_fd = self._ensure_dir(self._root_fd, "runs")
            self._tombstone_fd = self._ensure_dir(self._root_fd, ".expired")
        except BaseException:
            os.close(self._root_fd)
            raise

    def __del__(self) -> None:
        for name in ("_tombstone_fd", "_published_fd", "_staging_fd", "_root_fd"):
            descriptor = getattr(self, name, None)
            if isinstance(descriptor, int):
                with suppress(OSError):
                    os.close(descriptor)
                setattr(self, name, None)

    @staticmethod
    def _open_root(path: Path) -> int:
        absolute = path if path.is_absolute() else Path.cwd() / path
        descriptor = os.open("/", _DIR_FLAGS)
        try:
            for component in absolute.parts:
                if component == "/":
                    continue
                if component in {"", ".", ".."}:
                    raise OSError("result root must be canonical")
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _ensure_dir(parent_fd: int, name: str) -> int:
        try:
            return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
            os.fsync(descriptor)
            return descriptor

    @staticmethod
    def _open_dir(parent_fd: int, name: str) -> int:
        descriptor = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise UploadConflict("result component is not a directory")
        return descriptor

    @staticmethod
    def _key(identity: ResultIdentity) -> str:
        message = bytearray(b"AIPERF-RESULTS-STORAGE-IDENTITY\x01")
        for value in (
            identity.namespace,
            identity.job_id,
            identity.run_id,
            identity.object_uid,
        ):
            encoded = value.encode()
            message.extend(len(encoded).to_bytes(8, "big"))
            message.extend(encoded)
        return hashlib.sha256(message).hexdigest()

    @staticmethod
    def _safe_name(name: str) -> tuple[str, ...]:
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or str(path) != name
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise UploadInvalid("artifact path is not canonical and relative")
        return path.parts

    @staticmethod
    def _safe_content_type(content_type: str) -> bool:
        return bool(content_type) and not any(
            ord(character) < 32 or ord(character) == 127 for character in content_type
        )

    @staticmethod
    def _read_file(parent_fd: int, name: str, maximum: int) -> bytes:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
                raise UploadInvalid("result is not a bounded regular file")
            body = bytearray()
            while chunk := os.read(descriptor, min(64 * 1024, maximum + 1 - len(body))):
                body.extend(chunk)
                if len(body) > maximum:
                    raise UploadTooLarge("result file exceeds its limit")
            return bytes(body)
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_write(parent_fd: int, name: str, body: bytes) -> None:
        temporary = f".upload-{secrets.token_hex(16)}"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        try:
            offset = 0
            while offset < len(body):
                offset += os.write(descriptor, body[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent_fd)

    @staticmethod
    def _identity_body(identity: ResultIdentity, created: float) -> bytes:
        return json.dumps(
            {
                "namespace": identity.namespace,
                "jobId": identity.job_id,
                "runId": identity.run_id,
                "objectUid": identity.object_uid,
                "created": created,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _read_identity(self, run_fd: int) -> tuple[ResultIdentity, float]:
        try:
            document = json.loads(
                self._read_file(run_fd, _IDENTITY_NAME, MAX_MANIFEST_BYTES)
            )
            if set(document) != {
                "namespace",
                "jobId",
                "runId",
                "objectUid",
                "created",
            }:
                raise ValueError
            identity = ResultIdentity(
                document["namespace"],
                document["jobId"],
                document["runId"],
                document["objectUid"],
            )
            created = float(document["created"])
            if not 0 <= created < float("inf"):
                raise ValueError
            return identity, created
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise UploadInvalid("stored result identity is invalid") from error

    def _is_expired_locked(self, identity: ResultIdentity) -> bool:
        key = self._key(identity)
        try:
            body = self._read_file(self._tombstone_fd, key, MAX_MANIFEST_BYTES)
        except FileNotFoundError:
            return False
        try:
            document = json.loads(body)
            stored = ResultIdentity(
                document["namespace"],
                document["jobId"],
                document["runId"],
                document["objectUid"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise UploadConflict("stored expiry tombstone is invalid") from error
        if stored != identity:
            raise UploadConflict("stored expiry tombstone identity is inconsistent")
        return True

    def _refuse_expired_locked(self, identity: ResultIdentity) -> None:
        if self._is_expired_locked(identity):
            raise ResultsExpired("completed results have expired")

    def _expire_published_if_due_locked(self, identity: ResultIdentity) -> None:
        key = self._key(identity)
        try:
            run_fd = self._open_dir(self._published_fd, key)
        except FileNotFoundError:
            return
        try:
            stored, created = self._read_identity(run_fd)
            if stored != identity:
                raise UploadConflict("published identity is inconsistent")
            is_expired = self._now() - created > self._limits.published_ttl_seconds
        finally:
            os.close(run_fd)
        if not is_expired:
            return
        self._atomic_write(
            self._tombstone_fd,
            key,
            self._identity_body(identity, self._now()),
        )
        self._remove_tree(self._published_fd, key)
        self._runs.pop(identity, None)
        raise ResultsExpired("completed results have expired")

    def release_identity(self, identity: ResultIdentity) -> bool:
        """Purge one exact identity after its Kubernetes authority is deleted."""
        with self._lock:
            key = self._key(identity)
            if self._active.get(key):
                raise UploadConflict("result upload is still in progress")
            removed = self._runs.pop(identity, None) is not None
            for parent_fd, label in (
                (self._staging_fd, "staging"),
                (self._published_fd, "published"),
            ):
                try:
                    run_fd = self._open_dir(parent_fd, key)
                except FileNotFoundError:
                    continue
                try:
                    stored, _ = self._read_identity(run_fd)
                    if stored != identity:
                        raise UploadConflict(f"{label} identity is inconsistent")
                finally:
                    os.close(run_fd)
                self._remove_tree(parent_fd, key)
                removed = True
            if self._is_expired_locked(identity):
                os.unlink(key, dir_fd=self._tombstone_fd)
                os.fsync(self._tombstone_fd)
                removed = True
            return removed

    def _artifact_parent(
        self, run_fd: int, parts: tuple[str, ...], *, create: bool
    ) -> int:
        descriptor = os.dup(run_fd)
        try:
            for component in parts[:-1]:
                child = (
                    self._ensure_dir(descriptor, component)
                    if create
                    else self._open_dir(descriptor, component)
                )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _matches(parent_fd: int, name: str, digest: str, length: int) -> bool:
        try:
            descriptor = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
        except OSError:
            return False
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != length:
                return False
            actual = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    return False
                remaining -= len(chunk)
                actual.update(chunk)
            return not os.read(descriptor, 1) and actual.hexdigest() == digest
        finally:
            os.close(descriptor)

    def _walk(
        self, directory_fd: int, prefix: str = ""
    ) -> tuple[dict[str, int], int, int]:
        files: dict[str, int] = {}
        temporary_count = 0
        temporary_bytes = 0
        for name in os.listdir(directory_fd):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            relative = f"{prefix}/{name}" if prefix else name
            if stat.S_ISDIR(metadata.st_mode):
                child = self._open_dir(directory_fd, name)
                try:
                    nested, nested_count, nested_bytes = self._walk(child, relative)
                    files.update(nested)
                    temporary_count += nested_count
                    temporary_bytes += nested_bytes
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode):
                if name.startswith(".upload-"):
                    temporary_count += 1
                    temporary_bytes += metadata.st_size
                elif relative not in {_IDENTITY_NAME, _MANIFEST_NAME}:
                    files[relative] = metadata.st_size
            else:
                raise UploadConflict("result storage contains a non-regular entry")
        return files, temporary_count, temporary_bytes

    def _remove_tree(self, parent_fd: int, name: str) -> None:
        descriptor = self._open_dir(parent_fd, name)
        try:
            for child in os.listdir(descriptor):
                metadata = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    self._remove_tree(descriptor, child)
                else:
                    os.unlink(child, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)

    def _usage_locked(self) -> tuple[int, int, dict[str, tuple[int, int]]]:
        usage: dict[str, tuple[int, int]] = {}
        for key in os.listdir(self._staging_fd):
            run_fd = self._open_dir(self._staging_fd, key)
            try:
                identity, created = self._read_identity(run_fd)
                if self._key(identity) != key:
                    raise UploadConflict("staging identity is inconsistent")
                if (
                    self._now() - created > self._limits.staging_ttl_seconds
                    and not self._active.get(key)
                ):
                    expired = True
                    files = {}
                else:
                    expired = False
                    files, temporary_count, temporary_bytes = self._walk(run_fd)
            finally:
                os.close(run_fd)
            if expired:
                self._remove_tree(self._staging_fd, key)
            else:
                usage[key] = (
                    len(files) + temporary_count,
                    sum(files.values()) + temporary_bytes,
                )
        return len(usage), sum(item[1] for item in usage.values()), usage

    def staging_stats(self) -> dict[str, int]:
        """Return current unpublished usage after bounded expiry collection."""
        with self._lock:
            runs, stored, _ = self._usage_locked()
            return {"runs": runs, "bytes": stored}

    def _published_usage_locked(self) -> tuple[int, int]:
        usage: dict[str, int] = {}
        expired_identities: list[ResultIdentity] = []
        for key in os.listdir(self._published_fd):
            run_fd = self._open_dir(self._published_fd, key)
            try:
                identity, created = self._read_identity(run_fd)
                if self._key(identity) != key:
                    raise UploadConflict("published identity is inconsistent")
                files, temporary_count, _ = self._walk(run_fd)
                if temporary_count:
                    raise UploadConflict("published result contains temporary files")
                if self._now() - created > self._limits.published_ttl_seconds:
                    is_expired = True
                    expired_identities.append(identity)
                else:
                    is_expired = False
            finally:
                os.close(run_fd)
            if is_expired:
                self._atomic_write(
                    self._tombstone_fd,
                    key,
                    self._identity_body(identity, self._now()),
                )
                self._remove_tree(self._published_fd, key)
            else:
                usage[key] = sum(files.values())
        for identity in expired_identities:
            self._runs.pop(identity, None)
        return len(usage), sum(usage.values())

    def update_status(self, identity: ResultIdentity, status: dict[str, Any]) -> None:
        """Store status for one complete identity."""
        self._runs.setdefault(identity, RunRecord()).status = status

    def publish_manifest(
        self, identity: ResultIdentity, manifest: dict[str, Any]
    ) -> None:
        """Publish one validated manifest for one complete identity."""
        if not isinstance(manifest.get("artifacts"), list) or manifest.get(
            "runId"
        ) != identity.run_id:
            raise ValueError("results manifest does not match its complete identity")
        self._runs.setdefault(identity, RunRecord()).manifest = manifest

    def ready_manifest(self, identity: ResultIdentity) -> dict[str, Any] | None:
        """Return one complete identity's ready manifest."""
        with self._lock:
            self._refuse_expired_locked(identity)
            self._expire_published_if_due_locked(identity)
            record = self._runs.get(identity)
            return record.manifest if record else None

    def _manifest_at(self, run_fd: int, identity: ResultIdentity) -> dict[str, Any]:
        try:
            manifest = json.loads(self._read_file(run_fd, _MANIFEST_NAME, MAX_MANIFEST_BYTES))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UploadInvalid("stored manifest is invalid") from error
        if not self._is_ready_manifest(identity, manifest):
            raise UploadInvalid("stored manifest identity is invalid")
        return manifest

    def _published_status(
        self,
        identity: ResultIdentity,
        parts: tuple[str, ...],
        digest: str,
        length: int,
    ) -> bool | None:
        self._refuse_expired_locked(identity)
        self._expire_published_if_due_locked(identity)
        try:
            run_fd = self._open_dir(self._published_fd, self._key(identity))
        except FileNotFoundError:
            return None
        try:
            stored, _ = self._read_identity(run_fd)
            manifest = self._manifest_at(run_fd, identity)
            name = "/".join(parts)
            entry = next(
                (item for item in manifest["artifacts"] if item["path"] == name), None
            )
            if (
                stored != identity
                or entry is None
                or entry["sha256"] != digest
                or entry["bytes"] != length
            ):
                raise UploadConflict("run is already published with different content")
            parent_fd = self._artifact_parent(run_fd, parts, create=False)
            try:
                if not self._matches(parent_fd, parts[-1], digest, length):
                    raise UploadConflict("published artifact is corrupt")
            finally:
                os.close(parent_fd)
            return False
        finally:
            os.close(run_fd)

    def _begin_stage(
        self,
        identity: ResultIdentity,
        name: str,
        declared_sha256: str,
        declared_length: int,
    ) -> _PendingUpload | None:
        parts = self._safe_name(name)
        if declared_length < 0:
            raise UploadInvalid("content length must be non-negative")
        if declared_length > MAX_ARTIFACT_BYTES:
            raise UploadTooLarge("artifact exceeds the upload limit")
        if len(declared_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in declared_sha256
        ):
            raise UploadInvalid("artifact digest is not canonical SHA-256")
        key = self._key(identity)
        with self._lock:
            self._refuse_expired_locked(identity)
            published = self._published_status(
                identity, parts, declared_sha256, declared_length
            )
            if published is not None:
                return None
            run_count, stored_bytes, usage = self._usage_locked()
            created_run = False
            try:
                run_fd = self._open_dir(self._staging_fd, key)
                stored, _ = self._read_identity(run_fd)
                if stored != identity:
                    raise UploadConflict("staging identity is inconsistent")
            except FileNotFoundError:
                if run_count >= self._limits.max_staging_runs:
                    raise UploadTooLarge("staging run quota is exhausted") from None
                run_fd = self._ensure_dir(self._staging_fd, key)
                self._atomic_write(
                    run_fd, _IDENTITY_NAME, self._identity_body(identity, self._now())
                )
                usage[key] = (0, 0)
                created_run = True
            try:
                parent_fd = self._artifact_parent(run_fd, parts, create=True)
            finally:
                os.close(run_fd)
            try:
                if self._matches(parent_fd, parts[-1], declared_sha256, declared_length):
                    os.close(parent_fd)
                    return None
                try:
                    os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise UploadConflict("artifact path already contains different bytes")
                artifacts, run_bytes = usage[key]
                reserved_run = self._reserved.get(key, 0)
                reserved_total = sum(self._reserved.values())
                if artifacts >= self._limits.max_artifacts_per_run:
                    raise UploadTooLarge("staging artifact quota is exhausted")
                if run_bytes + reserved_run + declared_length > self._limits.max_run_bytes:
                    raise UploadTooLarge("staging run byte quota is exhausted")
                if stored_bytes + reserved_total + declared_length > self._limits.max_staging_bytes:
                    raise UploadTooLarge("global staging byte quota is exhausted")
                temporary = f".upload-{secrets.token_hex(16)}"
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
                self._active[key] = self._active.get(key, 0) + 1
                self._reserved[key] = reserved_run + declared_length
                return _PendingUpload(
                    key,
                    parent_fd,
                    temporary,
                    parts[-1],
                    os.fdopen(descriptor, "wb"),
                    hashlib.sha256(),
                    declared_sha256,
                    declared_length,
                )
            except BaseException:
                os.close(parent_fd)
                if created_run:
                    self._remove_tree(self._staging_fd, key)
                raise

    @staticmethod
    def _append_stage(upload: _PendingUpload, chunk: bytes) -> None:
        upload.length += len(chunk)
        if upload.length > upload.declared_length:
            raise UploadInvalid("artifact exceeds its declared length")
        upload.digest.update(chunk)
        upload.output.write(chunk)

    def _release_stage(self, upload: _PendingUpload, *, unlink: bool) -> None:
        if upload.released:
            return
        upload.released = True
        try:
            if not upload.output.closed:
                upload.output.close()
            if unlink:
                try:
                    os.unlink(upload.temporary_name, dir_fd=upload.parent_fd)
                    os.fsync(upload.parent_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(upload.parent_fd)
            with self._lock:
                active = self._active.get(upload.key, 0) - 1
                if active:
                    self._active[upload.key] = active
                else:
                    self._active.pop(upload.key, None)
                reserved = self._reserved.get(upload.key, 0) - upload.declared_length
                if reserved:
                    self._reserved[upload.key] = reserved
                else:
                    self._reserved.pop(upload.key, None)

    def _finish_stage(self, upload: _PendingUpload) -> bool:
        try:
            upload.output.flush()
            os.fsync(upload.output.fileno())
            if (
                upload.length != upload.declared_length
                or upload.digest.hexdigest() != upload.declared_digest
            ):
                raise UploadInvalid("artifact body does not match digest and length")
            try:
                os.link(
                    upload.temporary_name,
                    upload.destination_name,
                    src_dir_fd=upload.parent_fd,
                    dst_dir_fd=upload.parent_fd,
                    follow_symlinks=False,
                )
                created = True
            except FileExistsError:
                if not self._matches(
                    upload.parent_fd,
                    upload.destination_name,
                    upload.declared_digest,
                    upload.declared_length,
                ):
                    raise UploadConflict("artifact path contains different bytes") from None
                created = False
            os.fsync(upload.parent_fd)
            return created
        finally:
            self._release_stage(upload, unlink=True)

    async def stage_artifact(
        self,
        identity: ResultIdentity,
        name: str,
        chunks: AsyncIterable[bytes],
        declared_sha256: str,
        declared_length: int,
    ) -> bool:
        """Stream one artifact with all file IO and hashing off the event loop."""
        upload = await asyncio.to_thread(
            self._begin_stage, identity, name, declared_sha256, declared_length
        )
        if upload is None:
            return False
        try:
            async for chunk in chunks:
                await asyncio.to_thread(self._append_stage, upload, chunk)
            return await asyncio.to_thread(self._finish_stage, upload)
        except BaseException:
            await asyncio.shield(
                asyncio.to_thread(self._release_stage, upload, unlink=True)
            )
            raise

    def _declared(self, manifest: dict[str, Any]) -> dict[str, tuple[str, int]]:
        declared: dict[str, tuple[str, int]] = {}
        for artifact in manifest["artifacts"]:
            name = "/".join(self._safe_name(artifact["path"]))
            if name in declared:
                raise UploadInvalid("manifest artifact paths must be unique")
            if not self._safe_content_type(artifact["contentType"]):
                raise UploadInvalid("artifact content type contains control characters")
            if artifact["bytes"] > MAX_ARTIFACT_BYTES:
                raise UploadTooLarge("manifest artifact exceeds the upload limit")
            declared[name] = (artifact["sha256"], artifact["bytes"])
        if len(declared) > self._limits.max_artifacts_per_run:
            raise UploadTooLarge("manifest artifact count exceeds the run limit")
        if sum(length for _, length in declared.values()) > self._limits.max_run_bytes:
            raise UploadTooLarge("manifest bytes exceed the run limit")
        return declared

    def _verify_set(
        self, run_fd: int, declared: dict[str, tuple[str, int]]
    ) -> None:
        actual, temporary_count, _ = self._walk(run_fd)
        if temporary_count:
            raise UploadConflict("result upload is still in progress")
        if set(actual) != set(declared):
            raise UploadInvalid("stored artifacts do not exactly match the manifest")
        for name, (digest, length) in declared.items():
            parts = self._safe_name(name)
            parent_fd = self._artifact_parent(run_fd, parts, create=False)
            try:
                if not self._matches(parent_fd, parts[-1], digest, length):
                    raise UploadInvalid(f"stored artifact does not match manifest: {name}")
            finally:
                os.close(parent_fd)

    def commit_manifest(
        self,
        identity: ResultIdentity,
        body: bytes,
        declared_sha256: str,
        declared_length: int,
    ) -> bool:
        """Publish exact staged content and fsync both rename parents before ACK."""
        if declared_length > MAX_MANIFEST_BYTES:
            raise UploadTooLarge("manifest exceeds the upload limit")
        if (
            len(body) != declared_length
            or hashlib.sha256(body).hexdigest() != declared_sha256
        ):
            raise UploadInvalid("manifest body does not match digest and length")
        try:
            manifest = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UploadInvalid("manifest is not valid UTF-8 JSON") from error
        errors = sorted(_MANIFEST_VALIDATOR.iter_errors(manifest), key=str)
        if errors:
            raise UploadInvalid(errors[0].message)
        if manifest["runId"] != identity.run_id:
            raise UploadInvalid("manifest runId does not match upload identity")
        declared = self._declared(manifest)
        key = self._key(identity)

        with self._lock:
            self._refuse_expired_locked(identity)
            self._expire_published_if_due_locked(identity)
            if self._active.get(key):
                raise UploadConflict("result upload is still in progress")
            try:
                final_fd = self._open_dir(self._published_fd, key)
            except FileNotFoundError:
                final_fd = None
            if final_fd is not None:
                try:
                    stored, _ = self._read_identity(final_fd)
                    existing = self._read_file(
                        final_fd, _MANIFEST_NAME, MAX_MANIFEST_BYTES
                    )
                    if stored != identity or existing != body:
                        raise UploadConflict("run already has a different publication")
                    self._verify_set(final_fd, declared)
                    self.publish_manifest(identity, manifest)
                    return False
                finally:
                    os.close(final_fd)

            published_runs, published_bytes = self._published_usage_locked()
            if published_runs >= self._limits.max_published_runs:
                raise UploadTooLarge("published run quota is exhausted")
            declared_bytes = sum(length for _, length in declared.values())
            if published_bytes + declared_bytes > self._limits.max_published_bytes:
                raise UploadTooLarge("published byte quota is exhausted")

            try:
                staging_fd = self._open_dir(self._staging_fd, key)
            except FileNotFoundError as error:
                raise UploadInvalid("no staged artifacts exist for this identity") from error
            try:
                stored, _ = self._read_identity(staging_fd)
                if stored != identity:
                    raise UploadConflict("staging identity is inconsistent")
                self._verify_set(staging_fd, declared)
                self._atomic_write(staging_fd, _MANIFEST_NAME, body)
                self._atomic_write(
                    staging_fd,
                    _IDENTITY_NAME,
                    self._identity_body(identity, self._now()),
                )
                os.fsync(staging_fd)
            finally:
                os.close(staging_fd)
            try:
                os.rename(
                    key,
                    key,
                    src_dir_fd=self._staging_fd,
                    dst_dir_fd=self._published_fd,
                )
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                raise UploadConflict("run was published concurrently") from error
            os.fsync(self._staging_fd)
            os.fsync(self._published_fd)
            self.publish_manifest(identity, manifest)
            return True

    def open_artifact(self, identity: ResultIdentity, name: str) -> ArtifactHandle:
        """Open and verify one declared artifact without loading it into memory."""
        manifest = self.ready_manifest(identity)
        if manifest is None:
            raise FileNotFoundError("results are not ready")
        entry = next(
            (item for item in manifest["artifacts"] if item.get("path") == name), None
        )
        if entry is None:
            raise FileNotFoundError("artifact is not declared")
        parts = self._safe_name(name)
        run_fd = self._open_dir(self._published_fd, self._key(identity))
        try:
            parent_fd = self._artifact_parent(run_fd, parts, create=False)
        finally:
            os.close(run_fd)
        try:
            descriptor = os.open(parts[-1], _READ_FLAGS, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        try:
            metadata = os.fstat(descriptor)
            length = entry["bytes"]
            if (
                not stat.S_ISREG(metadata.st_mode)
                or length > MAX_ARTIFACT_BYTES
                or metadata.st_size != length
            ):
                raise FileNotFoundError("artifact is not a bounded regular file")
            digest = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    raise FileNotFoundError("artifact is truncated")
                remaining -= len(chunk)
                digest.update(chunk)
            if os.read(descriptor, 1) or digest.hexdigest() != entry["sha256"]:
                raise FileNotFoundError("artifact digest mismatch")
            os.lseek(descriptor, 0, os.SEEK_SET)
            return ArtifactHandle(
                os.fdopen(descriptor, "rb"), length, entry["contentType"]
            )
        except BaseException:
            os.close(descriptor)
            raise

    def stats(self) -> dict[str, int]:
        """Return result counts without exposing identities or content."""
        return {
            "runs": len(self._runs),
            "readyRuns": sum(
                record.manifest is not None for record in self._runs.values()
            ),
        }

    def rebuild(self) -> None:
        """Rebuild readiness using only descriptor-confined trusted identities."""
        self._runs.clear()
        with self._lock:
            for key in os.listdir(self._published_fd):
                try:
                    run_fd = self._open_dir(self._published_fd, key)
                except OSError:
                    continue
                try:
                    identity, _ = self._read_identity(run_fd)
                    if self._key(identity) != key:
                        continue
                    manifest = self._manifest_at(run_fd, identity)
                    self._verify_set(run_fd, self._declared(manifest))
                    self.publish_manifest(identity, manifest)
                except (OSError, UploadConflict, UploadInvalid, UploadTooLarge):
                    continue
                finally:
                    os.close(run_fd)

    @staticmethod
    def _is_ready_manifest(identity: ResultIdentity, manifest: Any) -> bool:
        if not isinstance(manifest, dict):
            return False
        if (
            any(_MANIFEST_VALIDATOR.iter_errors(manifest))
            or manifest.get("runId") != identity.run_id
        ):
            return False
        try:
            paths = [
                "/".join(ResultsIndex._safe_name(item["path"]))
                for item in manifest["artifacts"]
            ]
        except UploadInvalid:
            return False
        return len(paths) == len(set(paths)) and all(
            ResultsIndex._safe_content_type(item["contentType"])
            for item in manifest["artifacts"]
        )
