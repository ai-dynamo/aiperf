# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orphaned graph build artifacts are reclaimed on the next graph run.

``DatasetManager._cleanup`` reclaims the store and sidecar dirs on the stop
path, but that path does not run when the process dies abruptly -- the
``os._exit`` force-kill in ``cli_runner``, a SIGKILL from the service manager,
or a hard crash. Those runs orphan multi-GB dirs under the temp root.

The owner lock is what makes a later sweep safe: the OS drops a file lock when
the holding process dies by ANY means, so a lock that can still be acquired
proves its run is gone, while a contended one proves a concurrent run is still
using the dir.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import pytest
from filelock import FileLock

from aiperf.dataset.graph.artifact_gc import (
    FOREIGN_HOST_GRACE_SECONDS,
    acquire_owner_lock,
    current_boot_id,
    owner_lock_path,
    read_owner_identity,
    sweep_orphaned_graph_artifacts,
    write_owner_identity,
)


def _artifact_dir(base: Path, name: str, *, age_seconds: float = 0.0) -> Path:
    """Create an artifact dir with a payload file, optionally backdated."""
    d = base / name
    d.mkdir()
    (d / "content.blob").write_bytes(b"x" * 64)
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(d, (old, old))
    return d


def test_owner_lock_path_is_outside_the_artifact_dir(tmp_path: Path) -> None:
    """Windows must be able to remove the artifact dir while its lock is held."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_lock_location")

    assert owner_lock_path(d).parent == tmp_path


def test_sweep_removes_a_dir_whose_owner_process_is_gone(tmp_path: Path) -> None:
    """An artifact dir with a free owner lock belonged to a dead run and is removed."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_dead1", age_seconds=3600)
    # A crashed run leaves the lock FILE behind; the kernel released the lock.
    owner_lock_path(d).touch()

    removed = sweep_orphaned_graph_artifacts(tmp_path)

    assert removed == [d]
    assert not d.exists()


def test_sweep_removes_a_dir_that_never_got_a_lock_file(tmp_path: Path) -> None:
    """A dir from before the owner-lock convention is reclaimable on age alone.

    This is the ONLY case age decides. A current run locks its dirs before the
    build starts, so a live build is never lock-less -- see
    ``test_a_build_in_progress_survives_a_sweep_regardless_of_age``.
    """
    d = _artifact_dir(tmp_path, "aiperf_graph_meta_dead2", age_seconds=3600)

    removed = sweep_orphaned_graph_artifacts(tmp_path)

    assert removed == [d]
    assert not d.exists()


def test_sweep_skips_a_dir_held_by_a_live_run(tmp_path: Path) -> None:
    """A contended owner lock means a concurrent run owns the dir; leave it alone."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_live1", age_seconds=3600)
    holder = FileLock(str(owner_lock_path(d)))
    holder.acquire()
    try:
        removed = sweep_orphaned_graph_artifacts(tmp_path)
    finally:
        holder.release()

    assert removed == []
    assert d.is_dir(), "a live run's store must survive a concurrent sweep"


def test_sweep_skips_a_dir_inside_the_startup_grace_window(tmp_path: Path) -> None:
    """A just-created dir may belong to a run that has not yet taken its lock."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_starting")

    removed = sweep_orphaned_graph_artifacts(tmp_path)

    assert removed == []
    assert d.is_dir(), "the mkdir-before-lock window must not be swept"


def test_sweep_ignores_unrelated_directories(tmp_path: Path) -> None:
    """Only the two graph artifact prefixes are eligible."""
    keep = _artifact_dir(tmp_path, "aiperf_child_something", age_seconds=3600)
    other = _artifact_dir(tmp_path, "pytest-of-someone", age_seconds=3600)

    removed = sweep_orphaned_graph_artifacts(tmp_path)

    assert removed == []
    assert keep.is_dir()
    assert other.is_dir()


def test_acquired_owner_lock_blocks_a_sweep_of_that_dir(tmp_path: Path) -> None:
    """The lock a run holds for its own dir is exactly what the sweep tests."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_owned", age_seconds=3600)

    lock = acquire_owner_lock(d)
    assert lock is not None, "acquiring the owner lock of a fresh dir must succeed"
    try:
        assert sweep_orphaned_graph_artifacts(tmp_path) == []
        assert d.is_dir()
    finally:
        lock.release()

    assert sweep_orphaned_graph_artifacts(tmp_path) == [d]


def test_acquire_owner_lock_returns_none_when_another_run_holds_it(
    tmp_path: Path,
) -> None:
    """A second run must not believe it owns a dir another run already locked."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_contended")
    first = acquire_owner_lock(d)
    assert first is not None
    try:
        assert acquire_owner_lock(d) is None
    finally:
        first.release()


def test_sweep_reclaims_a_dir_from_an_earlier_boot_immediately(tmp_path: Path) -> None:
    """No process survives a reboot, so an earlier boot's dir needs no age wait.

    PostgreSQL reached the same conclusion after an OOM kill stranded 1.9 TB of
    temp files: namespace by boot id so orphans are identifiable rather than
    inferred from age.
    """
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_lastboot")
    owner_lock_path(d).touch()
    write_owner_identity(d, boot_id="a-previous-boot")

    # Young, and its lock file is present -- neither the age grace nor the lock
    # would reclaim it. The boot id is what proves it dead.
    assert sweep_orphaned_graph_artifacts(tmp_path) == [d]


def test_sweep_reclaims_an_earlier_boot_dir_on_a_foreign_host_too(
    tmp_path: Path,
) -> None:
    """Cross-host age-only fallback is skipped when the boot id already settles it."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_oldpodboot")
    write_owner_identity(d, host="some-other-pod", boot_id="a-previous-boot")

    assert sweep_orphaned_graph_artifacts(tmp_path) == [d]


def test_sweep_leaves_a_live_run_from_this_boot_alone(tmp_path: Path) -> None:
    """The boot-id shortcut must not bypass the lock for the current boot."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_thisboot", age_seconds=3600)
    lock = acquire_owner_lock(d)
    assert lock is not None
    try:
        assert sweep_orphaned_graph_artifacts(tmp_path) == []
        assert d.is_dir()
    finally:
        lock.release()


def test_acquire_owner_lock_records_this_boot(tmp_path: Path) -> None:
    """Claiming a dir stamps the boot a later sweep compares against."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_bootstamp")
    lock = acquire_owner_lock(d)
    assert lock is not None
    try:
        assert read_owner_identity(d)["boot_id"] == current_boot_id()
    finally:
        lock.release()


def test_sweep_never_reclaims_a_dir_owned_by_another_host(tmp_path: Path) -> None:
    """A shared PVC is a supported layout, and flock does not cross hosts there.

    MMAP_BASE_PATH is documented for "Kubernetes mounted volumes", and services
    may run on different hosts. NFS-backed volumes commonly scope flock to one
    host, so a peer's sweep can acquire a lock its owner still holds. Reclaiming
    on that basis would delete a LIVE run's store, so ownership by a foreign
    host disqualifies a dir outright.
    """
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_otherpod", age_seconds=3600)
    owner_lock_path(d).touch()
    write_owner_identity(d, host="some-other-pod")

    removed = sweep_orphaned_graph_artifacts(tmp_path)

    assert removed == []
    assert d.is_dir(), "another host's dir must never be reclaimed on liveness"


def test_sweep_reclaims_a_foreign_host_dir_only_when_ancient(tmp_path: Path) -> None:
    """Foreign-host dirs still get a safety valve, far beyond any live run."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_oldpod")
    owner_lock_path(d).touch()
    write_owner_identity(d, host="some-other-pod")
    ancient = time.time() - (FOREIGN_HOST_GRACE_SECONDS + 3600)
    os.utime(d, (ancient, ancient))

    assert sweep_orphaned_graph_artifacts(tmp_path) == [d]


def test_sweep_still_reclaims_this_hosts_own_dead_dir(tmp_path: Path) -> None:
    """The host guard must not disable same-host reclamation."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_mine", age_seconds=3600)
    owner_lock_path(d).touch()
    write_owner_identity(d)

    assert sweep_orphaned_graph_artifacts(tmp_path) == [d]


def test_acquire_owner_lock_records_this_host(tmp_path: Path) -> None:
    """Claiming a dir stamps the identity a later sweep reads."""
    d = _artifact_dir(tmp_path, "aiperf_graph_segments_stamped")
    lock = acquire_owner_lock(d)
    assert lock is not None
    try:
        assert read_owner_identity(d)["host"] == socket.gethostname()
    finally:
        lock.release()


def test_sweep_fails_closed_when_the_filesystem_has_no_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without flock, liveness is unprovable, so a locked dir is never reclaimed."""
    import aiperf.dataset.graph.artifact_gc as gc_mod

    d = _artifact_dir(tmp_path, "aiperf_graph_segments_noflock", age_seconds=3600)
    owner_lock_path(d).touch()

    def _no_flock(*args, **kwargs):
        raise NotImplementedError("use SoftFileLock instead")

    monkeypatch.setattr(gc_mod.FileLock, "acquire", _no_flock)

    assert sweep_orphaned_graph_artifacts(tmp_path) == []
    assert d.is_dir(), "an unprovable dir must leak, not be deleted"
    assert acquire_owner_lock(d) is None


@pytest.mark.asyncio
async def test_a_build_in_progress_survives_a_sweep_regardless_of_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run must lock its dirs BEFORE building, not after.

    The unlocked window would otherwise be the whole build -- measured at
    26-70s on single-file production corpora, and growing with corpus size
    while the grace stays constant. Once it exceeds the grace, a concurrent
    run's sweep deletes a live build's store out from under it.
    """
    from aiperf.common.environment import Environment
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.dataset_manager import DatasetManager
    from aiperf.dataset.graph import store_build
    from tests.unit.conftest import make_run_from_cli

    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    graph_min = (
        Path(__file__).parents[1]
        / "graph"
        / "adapters"
        / "fixtures"
        / "dynamo_nested"
        / "nested_2_level.jsonl.gz"
    )
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(graph_min),
            tokenizer_name="builtin",
        )
    )
    manager = DatasetManager(run=run, service_id="dm-inflight-test")
    store_dir = tmp_path / f"aiperf_graph_segments_{run.benchmark_id}"

    survived: dict[str, bool] = {}
    real_build = store_build.GraphStoreBuilder.build

    async def _build_with_concurrent_sweep(self, graph_path):
        # Mid-build, this dir is as old as the build is long. Backdate past the
        # grace and let a concurrent run's sweep have its shot at it.
        old = time.time() - 3600
        os.utime(store_dir, (old, old))
        sweep_orphaned_graph_artifacts(tmp_path)
        survived["store_dir"] = store_dir.is_dir()
        return await real_build(self, graph_path)

    monkeypatch.setattr(
        store_build.GraphStoreBuilder, "build", _build_with_concurrent_sweep
    )
    try:
        await manager._build_graph_store(graph_min)
        assert survived["store_dir"], (
            "a sweep during the build deleted the live run's store dir"
        )
    finally:
        await manager._cleanup()


@pytest.mark.asyncio
async def test_orphans_are_swept_before_the_build_allocates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reclaim must free the device BEFORE the new build writes to it.

    Sweeping afterwards cannot prevent the OSError(28) during dataset
    configuration that motivated the reclaim in the first place.
    """
    from aiperf.common.environment import Environment
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.dataset_manager import DatasetManager
    from aiperf.dataset.graph import store_build
    from tests.unit.conftest import make_run_from_cli

    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    graph_min = (
        Path(__file__).parents[1]
        / "graph"
        / "adapters"
        / "fixtures"
        / "dynamo_nested"
        / "nested_2_level.jsonl.gz"
    )
    orphan = _artifact_dir(tmp_path, "aiperf_graph_segments_old", age_seconds=3600)
    owner_lock_path(orphan).touch()

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(graph_min),
            tokenizer_name="builtin",
        )
    )
    manager = DatasetManager(run=run, service_id="dm-order-test")

    seen: dict[str, bool] = {}
    real_build = store_build.GraphStoreBuilder.build

    async def _record_then_build(self, graph_path):
        seen["orphan_gone_at_build_time"] = not orphan.exists()
        return await real_build(self, graph_path)

    monkeypatch.setattr(store_build.GraphStoreBuilder, "build", _record_then_build)
    try:
        await manager._build_graph_store(graph_min)
        assert seen["orphan_gone_at_build_time"], (
            "the orphan was still consuming the device when the build started"
        )
    finally:
        await manager._cleanup()


@pytest.mark.asyncio
async def test_dataset_manager_holds_the_owner_lock_for_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live graph run's artifacts survive a concurrent sweep, then vanish at stop."""
    from aiperf.common.environment import Environment
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.dataset.dataset_manager import DatasetManager
    from tests.unit.conftest import make_run_from_cli

    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)
    graph_min = (
        Path(__file__).parents[1]
        / "graph"
        / "adapters"
        / "fixtures"
        / "dynamo_nested"
        / "nested_2_level.jsonl.gz"
    )
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(graph_min),
            tokenizer_name="builtin",
        )
    )
    manager = DatasetManager(run=run, service_id="dm-gc-test")
    result = await manager._build_graph_store(graph_min)

    store_dir = tmp_path / f"aiperf_graph_segments_{run.benchmark_id}"
    sidecar_dir = result.sidecar_path.parent
    # Backdate past the grace window so only the lock decides.
    for d in (store_dir, sidecar_dir):
        old = time.time() - 3600
        os.utime(d, (old, old))

    assert sweep_orphaned_graph_artifacts(tmp_path) == []
    assert store_dir.is_dir() and sidecar_dir.is_dir()

    await manager._cleanup()

    assert not store_dir.exists()
    assert not sidecar_dir.exists()
