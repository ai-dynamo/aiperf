# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash-recovery cache for cluster-scoped chaos mutations.

The per-test ``async with faults.inject(...)`` block undoes mutations on
normal teardown, and the session sweeper handles namespace-scoped leftovers
from test crashes (see ``conftest.py``). Cluster-scoped mutations (RBAC,
ResourceQuota, NetworkPolicy at non-test namespaces, ...) cannot be discovered
by a namespace sweep, so concrete injectors record them here at inject time;
``pytest --chaos-sweep`` reads the JSON and reverses them.

Per spec §5: the file lives at ``~/.cache/aiperf/chaos-sweep.json`` so it
survives process crashes and can be inspected by hand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger

logger = AIPerfLogger(__name__)

CHAOS_SWEEP_CACHE_PATH: Path = Path.home() / ".cache" / "aiperf" / "chaos-sweep.json"
"""Append-only JSON log of cluster-scoped chaos mutations awaiting cleanup."""


@dataclass
class ClusterScopedMutation:
    """One cluster-scoped mutation that the session sweeper cannot undo.

    Concrete injectors construct this and call :py:func:`record_mutation`
    BEFORE performing the irreversible apiserver call (so a crash between the
    record and the call leaves a no-op entry rather than a missing one).

    ``kind``/``api_version``/``name`` together address the cluster resource.
    ``op`` is one of ``"create"`` (delete the resource to undo) or ``"patch"``
    (``payload`` carries the inverse patch).
    """

    kind: str
    api_version: str
    name: str
    op: str
    payload: dict[str, Any] = field(default_factory=dict)
    namespace: str | None = None


def record_mutation(mutation: ClusterScopedMutation) -> None:
    """Append a mutation to the on-disk cache.

    The file is rewritten atomically (write-temp + rename) so a crash mid-write
    leaves either the old contents or the new contents, never a truncated file.
    """
    CHAOS_SWEEP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_cache()
    existing.append(asdict(mutation))
    _write_cache(existing)


def _read_cache() -> list[dict[str, Any]]:
    if not CHAOS_SWEEP_CACHE_PATH.exists():
        return []
    try:
        data = orjson.loads(CHAOS_SWEEP_CACHE_PATH.read_bytes())
    except orjson.JSONDecodeError as exc:
        # Corrupt cache (mid-write crash on an older format) should not
        # block the test run; quarantine it and start fresh.
        backup = CHAOS_SWEEP_CACHE_PATH.with_suffix(".corrupt.json")
        CHAOS_SWEEP_CACHE_PATH.rename(backup)
        logger.warning(
            lambda exc=exc, bk=backup: (
                f"chaos-sweep cache was corrupt ({exc!r}); quarantined to {bk}"
            )
        )
        return []
    if not isinstance(data, list):
        return []
    return data


def _write_cache(entries: list[dict[str, Any]]) -> None:
    tmp = CHAOS_SWEEP_CACHE_PATH.with_suffix(".tmp")
    tmp.write_bytes(orjson.dumps(entries, option=orjson.OPT_INDENT_2))
    tmp.replace(CHAOS_SWEEP_CACHE_PATH)


def load_pending_mutations() -> list[ClusterScopedMutation]:
    """Return every mutation in the cache, oldest first."""
    return [ClusterScopedMutation(**entry) for entry in _read_cache()]


def clear_cache() -> None:
    """Wipe the cache. Called after a successful sweep."""
    if CHAOS_SWEEP_CACHE_PATH.exists():
        CHAOS_SWEEP_CACHE_PATH.unlink()


async def reverse_cluster_scoped_mutations() -> list[ClusterScopedMutation]:
    """Best-effort: undo every cached cluster-scoped mutation.

    Walks the cache in reverse (LIFO), invokes the inverse apiserver call,
    and clears the cache only on full success. Failures are logged but do
    not raise; the user can retry by re-running ``pytest --chaos-sweep``.

    Returns the list of mutations that could not be reversed (empty on full
    success).
    """
    # Concrete unwind logic intentionally deferred: Phase 1 only wires the
    # plumbing (cache shape + CLI hook). Phase 2+ injectors that actually
    # create cluster-scoped mutations will land the per-kind reverse calls
    # alongside their inject() methods.
    pending = load_pending_mutations()
    if not pending:
        return []
    n_pending = len(pending)
    logger.warning(
        lambda n=n_pending: (
            f"chaos-sweep: {n} cluster-scoped mutation(s) recorded but no "
            "reverse handlers are registered yet (Phase 1 stub); leaving cache "
            "intact for manual review"
        )
    )
    return pending


def pytest_addoption(parser: Any) -> None:
    """Register ``--chaos-sweep`` on the pytest CLI.

    Re-exported from ``conftest.py``; this lives here so the cache + CLI hook
    stay together.
    """
    group = parser.getgroup("chaos")
    group.addoption(
        "--chaos-sweep",
        action="store_true",
        default=False,
        help=(
            "Reverse cluster-scoped chaos mutations recorded in "
            f"{CHAOS_SWEEP_CACHE_PATH} and exit without collecting tests."
        ),
    )


def pytest_configure(config: Any) -> None:
    """If ``--chaos-sweep`` was passed, run the unwind and exit.

    Implemented via ``pytest.exit`` so collection never starts.
    """
    if not config.getoption("--chaos-sweep", default=False):
        return
    import asyncio

    import pytest

    failed = asyncio.run(reverse_cluster_scoped_mutations())
    if failed:
        pytest.exit(
            f"chaos-sweep: {len(failed)} mutation(s) could not be reversed; "
            f"see {CHAOS_SWEEP_CACHE_PATH}",
            returncode=2,
        )
    else:
        clear_cache()
        pytest.exit("chaos-sweep: complete", returncode=0)
