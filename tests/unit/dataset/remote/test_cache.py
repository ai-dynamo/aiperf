# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from aiperf.common.environment import Environment
from aiperf.config.dataset.remote_ref import RemoteRef
from aiperf.dataset.remote import cache


@pytest.fixture(autouse=True)
def _cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Environment.DATASET, "REMOTE_CACHE_DIR", tmp_path / "remote")
    return tmp_path / "remote"


def test_remote_cache_dir_defaults_under_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Environment.DATASET, "REMOTE_CACHE_DIR", None)
    assert cache.remote_cache_dir() == Path.home() / ".cache" / "aiperf" / "remote"


def test_entry_dir_is_keyed_by_canonical_and_version(_cache_root: Path) -> None:
    a = cache.entry_dir(RemoteRef(ref="hf://org/repo/x.jsonl").with_version("sha1"))
    b = cache.entry_dir(
        RemoteRef(ref="hf://datasets/org/repo/x.jsonl").with_version("sha1")
    )
    c = cache.entry_dir(RemoteRef(ref="hf://org/repo/x.jsonl").with_version("sha2"))
    assert a == b != c
    assert a.is_relative_to(_cache_root / "hf")


def test_entry_dir_requires_version() -> None:
    with pytest.raises(ValueError, match="resolved_version"):
        cache.entry_dir(RemoteRef(ref="s3://bucket/k"))


@pytest.mark.parametrize(
    "version",
    ['"abc:def"', 'W/"etag/with*odd?chars"', "a b", "..", ".", "x" * 500],
)
def test_entry_dir_sanitizes_version_names(version: str, _cache_root: Path) -> None:
    entry = cache.entry_dir(RemoteRef(ref="https://h/t.jsonl").with_version(version))
    name = entry.name
    assert entry.parent.parent == _cache_root / "https"
    assert all(c.isalnum() or c in "._-" for c in name)
    assert name not in {".", ".."}
    assert len(name) < 100


def test_entry_dir_distinct_versions_never_collide() -> None:
    ref = RemoteRef(ref="https://h/t.jsonl")
    # Both sanitize to the same readable prefix; the digest must tell them apart.
    a = cache.entry_dir(ref.with_version("a:b"))
    b = cache.entry_dir(ref.with_version("a/b"))
    c = cache.entry_dir(ref.with_version("a_b"))
    assert len({a, b, c}) == 3


@pytest.mark.asyncio
async def test_stream_to_file_writes_and_hashes(tmp_path: Path) -> None:
    async def chunks():
        yield b"ab"
        yield b"cd"

    dest = tmp_path / "out.jsonl"
    sha, size = await cache.stream_to_file(chunks(), dest)
    assert dest.read_bytes() == b"abcd"
    assert (sha, size) == (hashlib.sha256(b"abcd").hexdigest(), 4)


@pytest.mark.asyncio
async def test_stream_to_file_failure_leaves_no_partial(tmp_path: Path) -> None:
    async def chunks():
        yield b"partial"
        raise ConnectionResetError("dropped")

    dest = tmp_path / "out.jsonl"
    with pytest.raises(ConnectionResetError):
        await cache.stream_to_file(chunks(), dest)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_stream_to_file_cancellation_leaves_no_partial(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def chunks() -> AsyncIterator[bytes]:
        yield b"partial"
        started.set()
        await asyncio.sleep(3600)
        yield b"never"

    dest = tmp_path / "out.jsonl"
    task = asyncio.create_task(cache.stream_to_file(chunks(), dest))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_stream_to_file_failure_keeps_previous_dest(tmp_path: Path) -> None:
    dest = tmp_path / "out.jsonl"
    dest.write_bytes(b"old")

    async def chunks():
        yield b"new"
        raise ConnectionResetError("dropped")

    with pytest.raises(ConnectionResetError):
        await cache.stream_to_file(chunks(), dest)
    assert dest.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


def test_write_atomically_writes_and_hashes(tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "out.jsonl"
    sha, size = cache.write_atomically(iter([b"ab", b"cd"]), dest)
    assert dest.read_bytes() == b"abcd"
    assert (sha, size) == (hashlib.sha256(b"abcd").hexdigest(), 4)


def test_write_atomically_failure_leaves_no_partial(tmp_path: Path) -> None:
    def chunks():
        yield b"partial"
        raise OSError("disk full")

    with pytest.raises(OSError):
        cache.write_atomically(chunks(), tmp_path / "out.jsonl")
    assert list(tmp_path.iterdir()) == []


def _lock_is_free(entry: Path, timeout: float) -> bool:
    probe = FileLock(f"{entry}.lock", thread_local=False)
    try:
        probe.acquire(timeout=timeout)
    except Timeout:
        return False
    probe.release()
    return True


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


async def _enter_and_hold(entry: Path) -> None:
    async with cache.entry_lock(entry):
        await asyncio.Event().wait()


@pytest.fixture
def held_entry(tmp_path: Path) -> Iterator[tuple[Path, FileLock]]:
    """A cache entry whose lock another holder owns until the test releases it."""
    entry = tmp_path / "a" / "b"
    entry.parent.mkdir(parents=True)
    holder = FileLock(f"{entry}.lock", thread_local=False)
    holder.acquire()
    yield entry, holder
    if holder.is_locked:
        holder.release(force=True)


@pytest.mark.asyncio
async def test_entry_lock_cancelled_waiter_stops_waiting_on_its_thread(
    held_entry: tuple[Path, FileLock], monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, holder = held_entry
    active: list[int] = []

    class SpyLock(FileLock):
        def acquire(self, *args, **kwargs):
            active.append(1)
            try:
                return super().acquire(*args, **kwargs)
            finally:
                active.pop()

    monkeypatch.setattr(cache, "FileLock", SpyLock)
    try:
        waiter = asyncio.create_task(_enter_and_hold(entry))
        done, _ = await asyncio.wait({waiter}, timeout=0.3)
        assert not done and active
        started = time.monotonic()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert time.monotonic() - started < 1.0
        # While the other holder still has the lock, no worker thread may stay blocked
        # in acquire: one would hang interpreter shutdown for the full lock timeout.
        assert await asyncio.to_thread(_wait_until, lambda: not active, 3.0)
    finally:
        # Released here, not by the fixture: the event loop joins its worker threads
        # before fixture teardown, so a stranded acquire would deadlock it.
        holder.release()
    assert await asyncio.to_thread(_lock_is_free, entry, 5.0)


@pytest.mark.asyncio
async def test_entry_lock_times_out_at_the_deadline(
    held_entry: tuple[Path, FileLock], monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, holder = held_entry
    monkeypatch.setattr(cache, "_LOCK_TIMEOUT_S", 0.3)
    started = time.monotonic()
    try:
        with pytest.raises(Timeout):
            async with cache.entry_lock(entry):
                pass
        assert time.monotonic() - started < 3.0
    finally:
        holder.release()


def test_lock_attempt_abandoned_after_acquiring_releases(tmp_path: Path) -> None:
    lock = FileLock(f"{tmp_path / 'e'}.lock", thread_local=False)
    attempt = cache._LockAttempt(lock)
    assert attempt.run(timeout=1.0)
    attempt.abandon()
    assert not lock.is_locked


def test_lock_attempt_abandoned_while_waiting_releases_on_acquire(
    held_entry: tuple[Path, FileLock],
) -> None:
    entry, holder = held_entry
    lock = FileLock(f"{entry}.lock", thread_local=False)
    attempt = cache._LockAttempt(lock)
    results: list[bool] = []
    thread = threading.Thread(target=lambda: results.append(attempt.run(timeout=5.0)))
    thread.start()
    attempt.abandon()
    holder.release()
    thread.join(timeout=10.0)
    assert results == [False]
    assert not lock.is_locked
    assert _lock_is_free(entry, 1.0)


@pytest.mark.asyncio
async def test_entry_lock_creates_parent_and_releases(tmp_path: Path) -> None:
    entry = tmp_path / "a" / "b"
    async with cache.entry_lock(entry):
        assert entry.parent.is_dir()
    async with cache.entry_lock(entry):
        pass
