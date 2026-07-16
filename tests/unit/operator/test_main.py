# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for operator main module (kopf handlers)."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from unittest.mock import patch as mock_patch

import aiohttp
import kopf
import orjson
import pytest
import zstandard
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator.client_cache import (
    _progress_clients,
)
from aiperf.operator.client_cache import (
    close_progress_client as _close_progress_client,
)
from aiperf.operator.client_cache import (
    get_or_create_progress_client as _get_or_create_progress_client,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers.completion import (
    fetch_results_with_retry as _fetch_results_with_retry,
)
from aiperf.operator.handlers.monitor import (
    _get_elapsed_seconds,
    _get_job_timeout,
)
from aiperf.operator.health import check_endpoint_health as _check_endpoint_health
from aiperf.operator.main import configure
from aiperf.operator.models import (
    ControllerFetchResult,
    EndpointHealthResult,
    OwnerReference,
)
from aiperf.operator.progress_client import JobProgress
from aiperf.operator.status import Phase

# Body fixture with a deterministic creationTimestamp so results_layout
# functions (epoch_key_from_body) have a valid epoch in unit tests.
# 2024-04-25T17:02:03Z -> epoch 1714064523.
_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_BODY: dict[str, Any] = {
    "metadata": {"creationTimestamp": _FIXTURE_CREATION_TS}
}


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_zero_fetch_raises_temporary_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The field handler retries when the sidecar returns no aggregate files."""
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )
    from aiperf.operator.main import on_aiperfsweep_aggregation_complete

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=0, listed=0))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError):
        await on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_missing_aggregate_keeps_jobset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fetch that reports files but never landed aggregate.json must NOT
    delete the JobSet — the emptyDir-backed controller pod holds the only
    other copy, so deleting on a partial download destroys it. The handler
    retries the harvest instead.
    """
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    # files fetched, but no aggregate.json on disk
    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=3, listed=3))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError, match=r"aggregate harvest incomplete"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    fetch.assert_awaited_once()
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_partial_download_keeps_jobset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """downloaded < listed keeps the JobSet even when aggregate.json landed.

    The missing sibling artifacts (children.json, per-cell exports) exist
    only on the controller pod's emptyDir; deleting the JobSet on a partial
    harvest destroys them permanently.
    """
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b"{}")

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=2, listed=3))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError, match=r"harvest partial \(2/3"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    delete.assert_not_awaited()
    # The sentinel is full-success-only evidence; a partial harvest must
    # never mint it, or a later zero-download tick would treat-as-done.
    assert not (epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME).exists()


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_full_fetch_deletes_jobset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A full harvest writes the harvest sentinel, then reaps the JobSet."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b"{}")

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=3, listed=3))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    await operator_main.on_aiperfsweep_aggregation_complete(
        body={},
        status={"runEpoch": "1714064523"},
        name="latency-sweep",
        namespace="benchmarks",
    )

    delete.assert_awaited_once_with("benchmarks", "aiperf-latency-sweep")
    sentinel = epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME
    assert orjson.loads(sentinel.read_bytes()) == {
        "harvestComplete": True,
        "downloaded": 3,
        "listed": 3,
    }


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_truncated_aggregate_keeps_jobset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A crash-truncated aggregate.json must NOT pass the on-disk gate.

    exists() alone would accept the truncated file and delete the JobSet,
    permanently losing the only intact copy on the controller pod's emptyDir.
    """
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succee')  # truncated

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=3, listed=3))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError, match=r"aggregate harvest incomplete"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_zero_fetch_with_sentinel_treats_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A re-fire against a dead sidecar with parseable aggregate AND harvest
    sentinel on disk is treated as an already-finished harvest, not an
    endless retry loop. The aggregate alone is NOT enough (see the
    no-sentinel tests below)."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succeeded"}')
    (epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME).write_bytes(
        orjson.dumps({"harvestComplete": True, "downloaded": 3, "listed": 3})
    )

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=0, listed=0))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    await operator_main.on_aiperfsweep_aggregation_complete(
        body={},
        status={"runEpoch": "1714064523"},
        name="latency-sweep",
        namespace="benchmarks",
    )

    delete.assert_awaited_once_with("benchmarks", "aiperf-latency-sweep")


@pytest.mark.asyncio
async def test_sweep_aggregation_partial_then_zero_download_keeps_jobset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: the two-tick data-loss sequence must keep retrying.

    Tick 1: partial harvest (5/6 files, aggregate.json landed) raises
    TemporaryError and keeps the JobSet. Tick 2: the sidecar is transiently
    unreachable (downloaded=0/listed=0) — the old code saw the parseable
    aggregate.json, 'treated as done', and deleted the JobSet, destroying the
    emptyDir-only copy of the 6th file. Now the missing harvest sentinel plus
    the still-existing JobSet must keep the handler on the retry path.
    """
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succeeded"}')

    fetch = AsyncMock(
        side_effect=[
            SweepAggregateFetchResult(downloaded=5, listed=6),
            SweepAggregateFetchResult(downloaded=0, listed=0),
        ]
    )
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    jobset_exists = AsyncMock(return_value=True)
    monkeypatch.setattr(operator_main, "_sweep_jobset_exists", jobset_exists)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    # Tick 1: partial harvest — retry, no sentinel minted.
    with pytest.raises(kopf.TemporaryError, match=r"harvest partial \(5/6"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )
    assert not (epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME).exists()

    # Tick 2: sidecar transiently unreachable — still retry, still no delete.
    with pytest.raises(kopf.TemporaryError, match=r"without\s+harvest sentinel"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    delete.assert_not_awaited()
    jobset_exists.assert_awaited_once_with("benchmarks", "aiperf-latency-sweep")


@pytest.mark.asyncio
async def test_sweep_aggregation_full_harvest_then_crash_then_zero_deletes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An operator that crashed AFTER a full harvest but BEFORE the JobSet
    delete must still converge: the sentinel written on the full-success tick
    lets the zero-download re-fire delete without re-reaching the sidecar."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succeeded"}')

    fetch = AsyncMock(
        side_effect=[
            SweepAggregateFetchResult(downloaded=6, listed=6),
            SweepAggregateFetchResult(downloaded=0, listed=0),
        ]
    )
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    # Simulate the crash-before-delete: tick 1's delete raises after the
    # sentinel is already on disk, exactly the state a restarted operator sees.
    delete = AsyncMock(side_effect=[RuntimeError("operator crashed"), None])
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(RuntimeError, match="operator crashed"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )
    assert (epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME).is_file()

    # Re-fire on the restarted operator: dead-or-alive sidecar, zero files.
    await operator_main.on_aiperfsweep_aggregation_complete(
        body={},
        status={"runEpoch": "1714064523"},
        name="latency-sweep",
        namespace="benchmarks",
    )

    assert delete.await_count == 2
    delete.assert_awaited_with("benchmarks", "aiperf-latency-sweep")


@pytest.mark.asyncio
async def test_sweep_aggregation_zero_download_with_listed_files_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """downloaded=0 with listed>0 is a LIVE sidecar whose downloads all
    failed — never an already-finished harvest. Must retry, even when both
    the aggregate and the sentinel are already on disk from a prior epoch
    state, and must never delete the JobSet."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succeeded"}')
    (epoch_dir / operator_main.SWEEP_HARVEST_SENTINEL_NAME).write_bytes(
        orjson.dumps({"harvestComplete": True, "downloaded": 6, "listed": 6})
    )

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=0, listed=6))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError, match=r"listed 6 file\(s\) but none"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_aggregation_presentinel_harvest_jobset_gone_treats_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Back-compat: a PVC harvested by pre-sentinel operator versions has
    aggregate.json but no sentinel. With the JobSet confirmed gone (already
    reaped by the old code) there is nothing left to recover, so the handler
    completes without retry — and without a pointless delete."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succeeded"}')

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=0, listed=0))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    jobset_exists = AsyncMock(return_value=False)
    monkeypatch.setattr(operator_main, "_sweep_jobset_exists", jobset_exists)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    await operator_main.on_aiperfsweep_aggregation_complete(
        body={},
        status={"runEpoch": "1714064523"},
        name="latency-sweep",
        namespace="benchmarks",
    )

    jobset_exists.assert_awaited_once_with("benchmarks", "aiperf-latency-sweep")
    delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_aggregation_complete_zero_fetch_truncated_aggregate_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zero-fetch + truncated on-disk aggregate stays on the re-fetch path."""
    from aiperf.operator import main as operator_main
    from aiperf.operator.handlers.sweep import _aggregate_fetch
    from aiperf.operator.handlers.sweep._aggregate_fetch import (
        SweepAggregateFetchResult,
    )

    epoch_dir = tmp_path / "benchmarks" / "sweeps" / "latency-sweep" / "1714064523"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(b'{"phase": "Succee')  # truncated

    fetch = AsyncMock(return_value=SweepAggregateFetchResult(downloaded=0, listed=0))
    monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)
    delete = AsyncMock()
    monkeypatch.setattr(operator_main, "_delete_sweep_jobset", delete)
    monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)

    with pytest.raises(kopf.TemporaryError, match=r"sidecar returned no files"):
        await operator_main.on_aiperfsweep_aggregation_complete(
            body={},
            status={"runEpoch": "1714064523"},
            name="latency-sweep",
            namespace="benchmarks",
        )

    delete.assert_not_awaited()


# =============================================================================
# Helpers
# =============================================================================


async def _async_pod_list(*pods):
    """Create an async generator yielding pods, for mocking Pod.list."""
    for pod in pods:
        yield pod


def _fake_k8s_client(mock_api):
    """Async context manager helper that yields the given mock ApiClient."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _ctx():
        yield mock_api

    return _ctx()


class _V1ContainerStateWaiting:
    """Minimal stand-in for ``V1ContainerStateWaiting``."""

    def __init__(self, reason: str | None = None, message: str | None = None) -> None:
        self.reason = reason
        self.message = message


class _V1ContainerStateTerminated:
    """Minimal stand-in for ``V1ContainerStateTerminated``."""

    def __init__(
        self,
        reason: str | None = None,
        exit_code: int | None = None,
        message: str | None = None,
    ) -> None:
        self.reason = reason
        self.exit_code = exit_code
        self.message = message


class _V1ContainerState:
    def __init__(self, waiting=None, terminated=None, running=None) -> None:
        self.waiting = waiting
        self.terminated = terminated
        self.running = running


class _V1ContainerStatus:
    def __init__(
        self,
        name: str = "",
        restart_count: int = 0,
        state: _V1ContainerState | None = None,
        last_state: _V1ContainerState | None = None,
        ready: bool = False,
    ) -> None:
        self.name = name
        self.restart_count = restart_count
        self.state = state or _V1ContainerState()
        self.last_state = last_state or _V1ContainerState()
        self.ready = ready


class _V1PodStatus:
    def __init__(
        self,
        phase: str | None = None,
        container_statuses: list | None = None,
        init_container_statuses: list | None = None,
        conditions: list | None = None,
        pod_ip: str | None = None,
    ) -> None:
        self.phase = phase
        self.container_statuses = container_statuses
        self.init_container_statuses = init_container_statuses
        self.conditions = conditions
        self.pod_ip = pod_ip


class _V1ObjectMeta:
    def __init__(
        self,
        name: str = "",
        namespace: str = "",
        labels: dict | None = None,
        annotations: dict | None = None,
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.labels = labels
        self.annotations = annotations


class _V1Pod:
    def __init__(
        self,
        metadata: _V1ObjectMeta | None = None,
        status: _V1PodStatus | None = None,
        spec=None,
    ) -> None:
        self.metadata = metadata or _V1ObjectMeta()
        self.status = status or _V1PodStatus()
        self.spec = spec


class _V1PodList:
    def __init__(self, items: list) -> None:
        self.items = items


def _make_pod(
    *,
    name: str = "p",
    namespace: str = "default",
    labels: dict | None = None,
    annotations: dict | None = None,
    phase: str | None = None,
    container_statuses_raw: list | None = None,
    init_container_statuses_raw: list | None = None,
) -> _V1Pod:
    """Build a minimal typed V1Pod mock from raw-shape dicts."""

    def _cs_from_raw(d: dict) -> _V1ContainerStatus:
        state_d = d.get("state") or {}
        last_state_d = d.get("lastState") or {}
        waiting_d = state_d.get("waiting") or None
        terminated_d = state_d.get("terminated") or None
        last_term_d = last_state_d.get("terminated") or None

        waiting = (
            _V1ContainerStateWaiting(
                reason=waiting_d.get("reason"),
                message=waiting_d.get("message"),
            )
            if waiting_d is not None
            else None
        )
        terminated = (
            _V1ContainerStateTerminated(
                reason=terminated_d.get("reason"),
                exit_code=terminated_d.get("exitCode"),
                message=terminated_d.get("message"),
            )
            if terminated_d is not None
            else None
        )
        last_state = _V1ContainerState(
            terminated=(
                _V1ContainerStateTerminated(reason=last_term_d.get("reason"))
                if last_term_d is not None
                else None
            )
        )
        return _V1ContainerStatus(
            name=d.get("name", ""),
            restart_count=d.get("restartCount", 0),
            state=_V1ContainerState(waiting=waiting, terminated=terminated),
            last_state=last_state,
            ready=d.get("ready", False),
        )

    cs_list = [_cs_from_raw(d) for d in (container_statuses_raw or [])]
    ics_list = [_cs_from_raw(d) for d in (init_container_statuses_raw or [])]

    return _V1Pod(
        metadata=_V1ObjectMeta(
            name=name, namespace=namespace, labels=labels, annotations=annotations
        ),
        status=_V1PodStatus(
            phase=phase,
            container_statuses=cs_list,
            init_container_statuses=ics_list,
        ),
    )


async def _fake_pod_list(*pods) -> _V1PodList:
    return _V1PodList(items=list(pods))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_progress_client():
    """Create a mock ProgressClient."""
    client = AsyncMock()
    client.get_metrics = AsyncMock(return_value={"metrics": {"throughput": 100}})
    client.get_progress = AsyncMock()
    client.get_server_metrics = AsyncMock(return_value={})
    client.download_all_results = AsyncMock(return_value=["profile_export_aiperf.json"])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def temp_results_dir():
    """Create a temporary results directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Test _create_owner_reference
# =============================================================================


class TestCreateOwnerReference:
    """Tests for _create_owner_reference function."""

    @pytest.mark.parametrize(
        "name,uid",
        [
            param("my-job", "abc-123-uid", id="simple"),
            param("job-with-dashes", "uid-with-dashes-123", id="dashes"),
            param("a", "x", id="minimal"),
        ],
    )  # fmt: skip
    def test_creates_owner_reference_with_correct_fields(
        self, name: str, uid: str
    ) -> None:
        """Verify owner reference has all required fields."""
        ref = OwnerReference.for_aiperf_job(name, uid)

        assert ref.api_version == "aiperf.nvidia.com/v1alpha1"
        assert ref.kind == "AIPerfJob"
        assert ref.name == name
        assert ref.uid == uid
        assert ref.controller is True
        assert ref.block_owner_deletion is True

    def test_to_k8s_dict_serialization(self) -> None:
        """Verify to_k8s_dict produces correct camelCase keys."""
        ref = OwnerReference.for_aiperf_job("my-job", "uid-123")
        d = ref.to_k8s_dict()

        assert d == {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "name": "my-job",
            "uid": "uid-123",
            "controller": True,
            "blockOwnerDeletion": True,
        }


# =============================================================================
# Test _check_endpoint_health
# =============================================================================


class TestCheckEndpointHealth:
    """Tests for _check_endpoint_health async function."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code,expected_reachable",
        [
            param(200, True, id="ok"),
            param(201, True, id="created"),
            param(400, True, id="bad_request"),
            param(401, True, id="unauthorized"),
            param(404, True, id="not_found"),
            param(500, False, id="server_error"),
            param(503, False, id="unavailable"),
        ],
    )  # fmt: skip
    async def test_reachability_based_on_status_code(
        self, status_code: int, expected_reachable: bool
    ) -> None:
        """Verify reachability based on HTTP status codes."""
        mock_response = AsyncMock()
        mock_response.status = status_code
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with mock_patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_endpoint_health("http://test:8000")

        assert result.reachable is expected_reachable

    @pytest.mark.asyncio
    async def test_returns_false_when_all_endpoints_fail(self) -> None:
        """Verify returns False when no health endpoints respond."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with mock_patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_endpoint_health("http://test:8000", timeout=1.0)

        assert result.reachable is False
        assert "unreachable" in result.error.lower()

    @pytest.mark.asyncio
    async def test_returns_false_with_unexpected_error(self) -> None:
        """Verify returns False with error message for unexpected exceptions."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=RuntimeError("Unexpected"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with mock_patch("aiohttp.ClientSession", return_value=mock_session):
            result = await _check_endpoint_health("http://test:8000")

        assert result.reachable is False
        assert "Unexpected" in result.error


# =============================================================================
# Test _fetch_results_with_retry
# =============================================================================


class TestFetchResultsWithRetry:
    """Tests for _fetch_results_with_retry async function."""

    @pytest.fixture(autouse=True)
    def _clear_client_cache(self):
        """Clear the ProgressClient cache between tests."""
        _progress_clients.clear()
        yield
        _progress_clients.clear()

    @pytest.mark.asyncio
    async def test_returns_metrics_and_files_on_success(
        self, mock_progress_client: AsyncMock, temp_results_dir: Path
    ) -> None:
        """Verify returns both metrics and downloaded files."""
        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_progress_client,
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=1,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.metrics == {"metrics": {"throughput": 100}}
        assert result.downloaded == ["profile_export_aiperf.json"]
        assert result.checkpoints == []

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, temp_results_dir: Path) -> None:
        """Verify retries when fetch fails."""
        mock_client = AsyncMock()
        mock_client.get_metrics = AsyncMock(
            side_effect=[Exception("First fail"), {"metrics": {"ok": True}}]
        )
        mock_client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )

        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_client,
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=2,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.metrics == {"metrics": {"ok": True}}
        assert result.checkpoints == []

    @pytest.mark.asyncio
    async def test_returns_partial_results_after_max_retries(
        self, temp_results_dir: Path
    ) -> None:
        """Verify returns partial results if retries exhausted."""
        mock_client = AsyncMock()
        mock_client.get_metrics = AsyncMock(return_value={"metrics": {"partial": True}})
        mock_client.download_all_results = AsyncMock(
            side_effect=Exception("Download failed")
        )

        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_client,
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=1,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.metrics == {"metrics": {"partial": True}}
        assert result.downloaded == []
        assert result.checkpoints == []

    @pytest.mark.asyncio
    async def test_skips_download_when_results_dir_missing(
        self, mock_progress_client: AsyncMock
    ) -> None:
        """Verify skips download if RESULTS_DIR doesn't exist."""
        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_progress_client,
            ),
            mock_patch.object(
                OperatorEnvironment.RESULTS, "DIR", Path("/nonexistent/path")
            ),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=0,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.downloaded == []
        mock_progress_client.download_all_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_results_sidecar_when_primary_has_no_files(
        self, temp_results_dir: Path
    ) -> None:
        """Verify sidecar file-serving path is used when the main API cannot serve files."""
        mock_client = AsyncMock()
        mock_client.get_metrics = AsyncMock(return_value=None)
        mock_client.download_all_results = AsyncMock(return_value=[])

        sidecar_client = AsyncMock()
        sidecar_client.download_all_results = AsyncMock(
            return_value=["profile_export_aiperf.json"]
        )
        sidecar_client.__aenter__ = AsyncMock(return_value=sidecar_client)
        sidecar_client.__aexit__ = AsyncMock(return_value=None)

        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.ProgressClient",
                return_value=sidecar_client,
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=0,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.downloaded == ["profile_export_aiperf.json"]
        assert result.checkpoints == []
        sidecar_client.download_all_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracks_checkpoint_downloads_separately(
        self, temp_results_dir: Path
    ) -> None:
        """Verify checkpoint artifacts are kept separate from final exports."""
        mock_client = AsyncMock()
        mock_client.get_metrics = AsyncMock(return_value=None)
        mock_client.download_all_results = AsyncMock(
            return_value=["checkpoints/profile_export_aiperf_partial.json"]
        )

        sidecar_client = AsyncMock()
        sidecar_client.download_all_results = AsyncMock(return_value=[])
        sidecar_client.__aenter__ = AsyncMock(return_value=sidecar_client)
        sidecar_client.__aexit__ = AsyncMock(return_value=None)

        with (
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.get_or_create_progress_client",
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers._completion_fetch.ProgressClient",
                return_value=sidecar_client,
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            result = await _fetch_results_with_retry(
                "controller-host",
                "default",
                "job-123",
                max_retries=0,
                retry_delay=0.01,
                body=_FIXTURE_BODY,
            )

        assert result.downloaded == []
        assert result.checkpoints == ["checkpoints/profile_export_aiperf_partial.json"]


# =============================================================================
# Test configure
# =============================================================================


class TestConfigure:
    """Tests for configure kopf startup handler."""

    @pytest.mark.asyncio
    async def test_apiserver_proxy_login_disables_kopf_tls_verification(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Kopf has no TLS server-name field, so C15 must use insecure kopf transport."""
        from kopf._cogs.structs.credentials import ConnectionInfo

        from aiperf.operator.main import login_for_apiserver_proxy

        base_connection = ConnectionInfo(
            server="https://toxiproxy.aiperf-chaos-toxiproxy.svc.cluster.local:20000",
            ca_path="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
            insecure=False,
            priority=30,
        )
        login = AsyncMock(return_value=base_connection)
        monkeypatch.setenv(
            "AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE", "kubernetes.default.svc"
        )

        settings = kopf.OperatorSettings()
        logger = MagicMock()
        with mock_patch("aiperf.operator.main.kopf.login_via_async_client", login):
            result = await login_for_apiserver_proxy(logger=logger, settings=settings)

        login.assert_awaited_once_with(logger=logger, settings=settings)
        assert result is not None
        assert result.server == base_connection.server
        assert result.ca_path == base_connection.ca_path
        assert result.insecure is True

    def test_sets_finalizer(self) -> None:
        """Verify configures kopf finalizer."""
        settings = kopf.OperatorSettings()
        with mock_patch("aiperf.operator.main.start_metrics_server"):
            configure(settings)

        assert settings.persistence.finalizer == "aiperf.nvidia.com/finalizer"

    def test_sets_posting_level(self) -> None:
        """Verify sets posting log level."""
        import logging

        settings = kopf.OperatorSettings()
        with mock_patch("aiperf.operator.main.start_metrics_server"):
            configure(settings)

        assert settings.posting.level == logging.INFO


# =============================================================================
class TestOnCreateHandler:
    """Tests for on_create kopf handler."""

    @pytest.fixture(autouse=True)
    def _mock_persistence(self):
        """Persistence runs before JobSet create (H1); stub it out for tests.

        Without this, save_job_spec_file hits /data and raises TemporaryError,
        which masks the actual resource-creation assertions.
        """
        with (
            mock_patch(
                "aiperf.operator.handlers.create.save_job_spec_file",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.runs_index",
                new=MagicMock(upsert_run_created=AsyncMock()),
            ),
        ):
            yield

    @pytest.fixture
    def mock_all_events(self):
        """Mock all event functions to avoid kopf context issues."""
        with (
            mock_patch("aiperf.operator.events.spec_valid"),
            mock_patch("aiperf.operator.events.spec_invalid"),
            mock_patch("aiperf.operator.events.endpoint_reachable"),
            mock_patch("aiperf.operator.events.endpoint_unreachable"),
            mock_patch("aiperf.operator.events.resources_created"),
            mock_patch("aiperf.operator.events.created"),
            mock_patch("aiperf.operator.events.failed"),
            mock_patch("aiperf.operator.events.preflight_passed"),
            mock_patch("aiperf.operator.events.preflight_failed"),
            mock_patch("aiperf.operator.events.preflight_warning"),
        ):
            yield

    @pytest.mark.asyncio
    async def test_fails_with_invalid_spec(self, mock_all_events: None) -> None:
        """Verify fails permanently with invalid spec."""
        from aiperf.operator.main import on_create

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        spec = {"endpoint": {}}  # Missing required fields
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        with pytest.raises(kopf.PermanentError, match="Invalid spec"):
            await on_create(
                body=body,
                spec=spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

    @pytest.mark.asyncio
    async def test_creates_resources_successfully(
        self,
        mock_all_events: None,
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """Verify creates ConfigMap and JobSet on valid spec."""
        from aiperf.operator.main import on_create

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()
        AsyncMock()
        AsyncMock()

        mock_preflight = MagicMock()
        mock_preflight.run_all = AsyncMock(
            return_value=MagicMock(passed=True, checks=[]),
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=mock_preflight,
            ),
        ):
            result = await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        assert "jobSetName" in result
        assert "workers" in result

    @pytest.mark.asyncio
    async def test_handles_unreachable_endpoint_as_warning(
        self,
        mock_all_events: None,
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """Verify unreachable endpoint is a warning, not failure."""
        from aiperf.operator.main import on_create

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()
        AsyncMock()
        AsyncMock()

        mock_preflight = MagicMock()
        mock_preflight.run_all = AsyncMock(
            return_value=MagicMock(passed=True, checks=[]),
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=False, error="Connection refused"),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=mock_preflight,
            ),
        ):
            result = await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        assert result is not None


class TestOnDeleteHandler:
    """Tests for on_delete kopf handler."""

    @pytest.mark.asyncio
    async def test_delegates_to_lifecycle_on_delete(self) -> None:
        """Verify on_delete forwards name/namespace/status to lifecycle.on_delete."""
        from aiperf.operator.main import on_delete

        with mock_patch(
            "aiperf.operator.main.lifecycle.on_delete", new_callable=AsyncMock
        ) as mock_lifecycle_delete:
            await on_delete(name="test-job", namespace="default", status={})

        mock_lifecycle_delete.assert_awaited_once_with(
            name="test-job", namespace="default", status={}
        )


class TestOnCancelHandler:
    """Tests for on_cancel kopf handler."""

    @pytest.fixture
    def mock_cancel_events(self):
        """Mock events for cancel handler."""
        with mock_patch("aiperf.operator.events.cancelled"):
            yield

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "spec,status,should_cancel",
        [
            param({"cancel": False}, {"phase": Phase.RUNNING}, False, id="cancel_false"),
            param({"cancel": True}, {"phase": Phase.COMPLETED}, False, id="already_completed"),
            param({"cancel": True}, {"phase": Phase.FAILED}, False, id="already_failed"),
            param({"cancel": True}, {"phase": Phase.CANCELLED}, False, id="already_cancelled"),
        ],
    )  # fmt: skip
    async def test_ignores_when_cancel_not_applicable(
        self,
        spec: dict[str, Any],
        status: dict[str, Any],
        should_cancel: bool,
    ) -> None:
        """Verify does nothing when cancel is not applicable."""
        from aiperf.operator.main import on_cancel

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        await on_cancel(
            body=_FIXTURE_BODY,
            spec=spec,
            status=status,
            name="test-job",
            namespace="default",
            patch=kopf_patch,
        )

        assert kopf_patch.status.get("phase") != Phase.CANCELLED

    @pytest.mark.asyncio
    async def test_cancels_running_job(
        self,
        mock_cancel_events: None,
    ) -> None:
        """Verify cancels running job and deletes JobSet."""
        from aiperf.operator.main import on_cancel

        spec = {"cancel": True}
        status = {
            "phase": Phase.RUNNING,
            "jobId": "job-123",
            "jobSetName": "jobset-123",
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_delete = AsyncMock(return_value={})
        mock_custom = MagicMock(delete_namespaced_custom_object=mock_delete)

        with (
            mock_patch(
                "aiperf.operator.handlers.lifecycle.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            await on_cancel(
                body=_FIXTURE_BODY,
                spec=spec,
                status=status,
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        mock_delete.assert_awaited_once()
        assert kopf_patch.status["phase"] == Phase.CANCELLED


class TestMonitorProgressHandler:
    """Tests for monitor_progress kopf timer handler."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        # on_delete (TestOnDeleteHandler) sets a STICKY cancellation flag for
        # default/test-job in client_cache._cancellation_events that is not
        # cleared by _close_unlocked; without a reset it leaks here and
        # monitor_progress short-circuits on the cancellation check before
        # stamping status.phase. Mirror the reset the other monitor test
        # classes (TestMonitorProgressAdvanced, ...) already use.
        from aiperf.operator.client_cache import _reset_for_testing

        _reset_for_testing()
        yield
        _reset_for_testing()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param(Phase.COMPLETED, id="completed"),
            param(Phase.FAILED, id="failed"),
            param(Phase.CANCELLED, id="cancelled"),
        ],
    )  # fmt: skip
    async def test_skips_terminal_jobs(self, phase: str) -> None:
        """Verify skips monitoring for terminal jobs."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        await monitor_progress(
            body=_FIXTURE_BODY,
            status={"phase": phase},
            spec={},
            name="test-job",
            namespace="default",
            patch=kopf_patch,
        )

        assert kopf_patch.status == {}

    @pytest.mark.asyncio
    async def test_skips_when_no_jobset(self) -> None:
        """Verify skips when jobSetName is missing."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        await monitor_progress(
            body=_FIXTURE_BODY,
            status={"phase": Phase.PENDING},
            spec={},
            name="test-job",
            namespace="default",
            patch=kopf_patch,
        )

        assert kopf_patch.status == {}

    @pytest.mark.asyncio
    async def test_handles_jobset_not_found(self) -> None:
        """Verify sets Failed phase when JobSet is gone and the CR is a genuine orphan.

        The reconciler now requires positive evidence that the CR is NOT
        a victim of the JobSet-not-found phase-stomp race (claim annotation
        absent on cached body, fresh re-read confirms a non-terminal phase
        with no claim) before stamping FAILED. See
        ``test_monitor_jobset_not_found_race.py`` for the regression suite
        on the success-path race.
        """
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        # Two get_namespaced_custom_object calls: JobSet 404, then fresh
        # AIPerfJob re-read returning a non-terminal phase with no claim.
        mock_get = AsyncMock(
            side_effect=[
                ApiException(status=404, reason="not found"),
                {
                    "metadata": {"annotations": {}},
                    "status": {"phase": str(Phase.RUNNING)},
                },
            ]
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(get_namespaced_custom_object=mock_get),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.FAILED

    @pytest.mark.asyncio
    async def test_transitions_pending_to_initializing(self) -> None:
        """Verify transitions from Pending to Initializing when workers start."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [{"name": "workers", "ready": 1}],
            }
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.PENDING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                    "workers": {"total": 2, "ready": 0},
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.INITIALIZING


class TestMonitorCompletedClaimsShutdownKey:
    """K1 regression: monitor's ``Completed`` condition branch must claim
    the job key in ``_shutdown_sent`` BEFORE awaiting handle_completion so
    a concurrent ``on_benchmark_complete`` handler sees the claim and
    returns early instead of racing into a second results fetch."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        from aiperf.operator.client_cache import _reset_for_testing

        _reset_for_testing()
        yield
        _reset_for_testing()

    @pytest.mark.asyncio
    async def test_key_is_claimed_before_handle_completion_awaits(self) -> None:
        from aiperf.operator.client_cache import _shutdown_sent
        from aiperf.operator.main import monitor_progress

        observed_claim: list[bool] = []

        async def fake_handle_completion(*args, **kwargs) -> None:
            observed_claim.append("default/job-123" in _shutdown_sent)

        async def fake_claim(namespace, name, body):
            _shutdown_sent.add(f"{namespace}/job-123")
            return True

        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [{"type": "Completed", "status": "True"}],
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                side_effect=fake_claim,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                side_effect=fake_handle_completion,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert observed_claim == [True], (
            "handle_completion must be invoked AFTER the key is added to "
            "_shutdown_sent; otherwise a concurrent on_benchmark_complete "
            "watch handler can also pass its check and run in parallel"
        )

    @pytest.mark.asyncio
    async def test_concurrent_watch_handler_returns_after_monitor_claims(
        self,
    ) -> None:
        """If monitor claims and is mid-handle_completion, a concurrent
        on_benchmark_complete for the same key must NOT also call
        handle_completion."""
        import asyncio

        from aiperf.operator.handlers.lifecycle import on_benchmark_complete
        from aiperf.operator.main import monitor_progress

        release_monitor = asyncio.Event()
        completion_calls: list[str] = []

        async def fake_handle_completion(
            body, namespace, jobset_name, job_id, status, sb, result=None
        ) -> None:
            completion_calls.append(job_id)
            await release_monitor.wait()

        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [{"type": "Completed", "status": "True"}],
            }
        }
        kopf_patch_monitor = MagicMock()
        kopf_patch_monitor.status = {}
        kopf_patch_watch = MagicMock()
        kopf_patch_watch.status = {}

        call_count = {"claim": 0}

        async def fake_claim(namespace, name, body):
            from aiperf.operator.client_cache import _shutdown_sent

            call_count["claim"] += 1
            key = f"{namespace}/job-123"
            if key in _shutdown_sent:
                return False
            _shutdown_sent.add(key)
            return True

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                side_effect=fake_claim,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.try_claim_completion",
                side_effect=fake_claim,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                side_effect=fake_handle_completion,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.handle_completion",
                side_effect=fake_handle_completion,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.get_or_create_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.lifecycle.close_progress_client",
                new_callable=AsyncMock,
            ),
        ):
            monitor_task = asyncio.create_task(
                monitor_progress(
                    body=_FIXTURE_BODY,
                    status={
                        "phase": Phase.RUNNING,
                        "jobSetName": "test-jobset",
                        "jobId": "job-123",
                    },
                    spec={},
                    name="test-job",
                    namespace="default",
                    patch=kopf_patch_monitor,
                )
            )

            # Let the monitor reach the handle_completion await and block.
            for _ in range(50):
                await asyncio.sleep(0)
                if completion_calls:
                    break
            assert completion_calls == ["job-123"]

            # Now fire the watch handler for the same CR. It must see the
            # claim and return early without invoking handle_completion.
            await on_benchmark_complete(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                name="test-job",
                namespace="default",
                patch=kopf_patch_watch,
            )

            release_monitor.set()
            await monitor_task

        assert completion_calls == ["job-123"], (
            "Only the monitor's handle_completion should have run; the "
            "concurrent watch handler must have returned early because "
            "the key was already in _shutdown_sent"
        )


class TestMonitorProgressAdvanced:
    """Additional tests for monitor_progress handler edge cases."""

    @pytest.fixture(autouse=True)
    def _reset_state(self):
        from aiperf.operator.client_cache import _reset_for_testing

        _reset_for_testing()
        yield
        _reset_for_testing()

    @pytest.mark.asyncio
    async def test_preserves_bootstrap_fallback_when_controller_progress_unavailable(
        self,
    ) -> None:
        """Verify JobSet readiness still bootstraps Initializing when progress is unavailable."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [{"name": "workers", "ready": 1, "active": 1}],
            }
        }
        mock_client = AsyncMock()
        mock_client.get_progress = AsyncMock(
            return_value=JobProgress(connection_error="controller unavailable")
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._maybe_recover_terminated_controller",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.PENDING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.INITIALIZING
        assert kopf_patch.status["workers"] == {"ready": 1, "total": 2}
        assert "currentPhase" not in kopf_patch.status

    @pytest.mark.asyncio
    async def test_controller_jobset_failure_remains_fatal(self) -> None:
        """Verify controller JobSet failure still fails the benchmark."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "message": "controller crashed",
                    }
                ],
                "replicatedJobsStatus": [
                    {"name": "workers", "failed": 2, "ready": 248, "active": 0},
                    {"name": "controller", "failed": 1, "ready": 0, "active": 0},
                ],
            }
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch("aiperf.operator.events.failed") as mock_failed,
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.FAILED
        assert "controller crashed" in kopf_patch.status["error"]
        mock_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_only_jobset_failure_is_non_fatal(self) -> None:
        """Verify worker-only JobSet failures do not fail the benchmark."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [
                    {
                        "type": "Failed",
                        "status": "True",
                        "message": "jobset failed due to worker failures",
                    }
                ],
                "replicatedJobsStatus": [
                    {"name": "workers", "failed": 2, "ready": 248, "active": 0},
                    {"name": "controller", "failed": 0, "ready": 1, "active": 1},
                ],
            }
        }

        mock_client = AsyncMock()
        mock_client.get_progress = AsyncMock(return_value=None)
        mock_client.get_metrics = AsyncMock(return_value={})
        mock_client.get_server_metrics = AsyncMock(return_value={})
        mock_client.get_worker_startup_states = AsyncMock(return_value=None)

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.get_or_create_progress_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor._fetch_progress",
                new_callable=AsyncMock,
                return_value=False,
            ),
            mock_patch("aiperf.operator.events.failed") as mock_failed,
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                    "workers": {"total": 250, "ready": 250},
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status.get("phase") != Phase.FAILED
        assert kopf_patch.status["workers"]["total"] == 250
        mock_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_generic_api_exception(self) -> None:
        """Verify handles non-404 API exceptions gracefully."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        side_effect=ApiException(status=500, reason="server error")
                    )
                ),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

    @pytest.mark.asyncio
    async def test_completed_jobset_with_failed_workers_still_completes(self) -> None:
        """Verify controller completion still triggers completion handling with failed workers."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [{"type": "Completed", "status": "True"}],
                "replicatedJobsStatus": [
                    {"name": "workers", "failed": 5, "ready": 0, "succeeded": 245},
                    {"name": "controller", "failed": 0, "ready": 0, "succeeded": 1},
                ],
            }
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_completion,
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        mock_completion.assert_awaited_once()


class TestHandleCompletion:
    """Tests for _handle_completion function."""

    @pytest.fixture(autouse=True)
    def _clear_client_cache(self):
        """Clear operator client-cache state between tests."""
        from aiperf.operator.client_cache import _reset_for_testing

        _reset_for_testing()
        yield
        _reset_for_testing()

    @pytest.mark.asyncio
    async def test_sets_conditions_and_phase(self, temp_results_dir: Path) -> None:
        """Verify sets conditions and phase on completion."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 2}})

        # Mock metrics with proper structure for MetricsSummary (list of metric objects)
        mock_metrics = {
            "metrics": [
                {"tag": "request_throughput", "avg": 100.0},
                {"tag": "request_latency", "avg": 50.0},
            ]
        }

        AsyncMock()
        AsyncMock()

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed"),
            mock_patch("aiperf.operator.events.results_stored"),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.client.CustomObjectsApi",
                return_value=MagicMock(
                    delete_namespaced_custom_object=AsyncMock(return_value={})
                ),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-123",
                status={"workers": {"total": 2}},
                sb=sb,
                result=ControllerFetchResult(
                    metrics=mock_metrics, downloaded=["profile_export_aiperf.json"]
                ),
            )

        assert kopf_patch.status["phase"] == Phase.COMPLETED
        assert "completionTime" in kopf_patch.status

    @pytest.mark.asyncio
    async def test_inputs_json_only_does_not_count_as_results_stored(
        self, temp_results_dir: Path
    ) -> None:
        """Verify inputs.json alone does not mark the job as ResultsStored."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import ConditionType, StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 2}})

        AsyncMock()
        mock_js = AsyncMock()

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed"),
            mock_patch("aiperf.operator.events.results_stored") as mock_results_stored,
            mock_patch("aiperf.operator.events.results_failed") as mock_results_failed,
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.client.CustomObjectsApi",
                return_value=MagicMock(
                    delete_namespaced_custom_object=AsyncMock(return_value={})
                ),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-123",
                status={"workers": {"total": 2}},
                sb=sb,
                result=ControllerFetchResult(
                    metrics={"metrics": [{"tag": "request_throughput", "avg": 100.0}]},
                    downloaded=["inputs.json"],
                ),
            )

        results_available = next(
            c
            for c in kopf_patch.status["conditions"]
            if c["type"] == ConditionType.RESULTS_AVAILABLE
        )
        assert results_available["status"] == "False"
        assert results_available["reason"] == "ResultsFetchFailed"
        mock_results_stored.assert_not_called()
        mock_results_failed.assert_called_once()
        mock_js.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_missing_metrics(self, temp_results_dir: Path) -> None:
        """Verify handles case when metrics fetch fails but files are available."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 2}})

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed"),
            mock_patch("aiperf.operator.events.results_stored"),
            mock_patch("aiperf.operator.events.results_failed"),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.client.CustomObjectsApi",
                return_value=MagicMock(
                    delete_namespaced_custom_object=AsyncMock(return_value={})
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-123",
                status={"workers": {"total": 2}},
                sb=sb,
                result=ControllerFetchResult(
                    metrics=None,
                    downloaded=["profile_export_aiperf.json"],
                ),
            )

        assert kopf_patch.status["phase"] == Phase.COMPLETED

    @pytest.mark.asyncio
    async def test_calculates_duration(self, temp_results_dir: Path) -> None:
        """Verify calculates duration from startTime."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import StatusBuilder, format_timestamp

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        start_time = format_timestamp()
        sb = StatusBuilder(
            kopf_patch, {"workers": {"total": 1}, "startTime": start_time}
        )

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed") as mock_completed,
            mock_patch("aiperf.operator.events.results_stored"),
            mock_patch("aiperf.operator.events.results_failed"),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.client.CustomObjectsApi",
                return_value=MagicMock(
                    delete_namespaced_custom_object=AsyncMock(return_value={})
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.completion.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-123",
                status={"workers": {"total": 1}, "startTime": start_time},
                sb=sb,
                result=ControllerFetchResult(
                    metrics={"metrics": {}},
                    downloaded=["profile_export_aiperf.json"],
                ),
            )

        mock_completed.assert_called_once()


class TestFetchProgress:
    """Tests for _fetch_progress function."""

    @pytest.mark.asyncio
    async def test_handles_connection_error(self) -> None:
        """Verify handles connection errors gracefully."""
        from aiperf.operator.handlers.monitor import _fetch_progress
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch)

        mock_progress = MagicMock()
        mock_progress.connection_error = True

        mock_client = AsyncMock()
        mock_client.get_progress = AsyncMock(return_value=mock_progress)

        await _fetch_progress(
            "default", "test-jobset", kopf_patch, sb, mock_client, "job-1"
        )

        # Should return early without updating liveMetrics
        assert "liveMetrics" not in kopf_patch.status

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self) -> None:
        """Verify handles exceptions without crashing."""
        from aiperf.operator.handlers.monitor import _fetch_progress
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch)

        mock_client = AsyncMock()
        mock_client.get_progress = AsyncMock(side_effect=Exception("Network error"))

        # Should not raise
        await _fetch_progress(
            "default", "test-jobset", kopf_patch, sb, mock_client, "job-1"
        )


class TestCleanupOldResultsTimer:
    """Tests for cleanup_old_results kopf timer handler."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param(Phase.PENDING, id="pending"),
            param(Phase.RUNNING, id="running"),
            param(Phase.FAILED, id="failed"),
        ],
    )  # fmt: skip
    async def test_skips_non_completed_jobs(self, phase: str) -> None:
        """Verify skips jobs that aren't completed."""
        from aiperf.operator.handlers.cleanup import cleanup_old_results

        await cleanup_old_results(
            body=_FIXTURE_BODY,
            status={"phase": phase},
            name="test-job",
        )

    @pytest.mark.asyncio
    async def test_skips_when_no_results_path(self) -> None:
        """Verify skips when resultsPath is not set."""
        from aiperf.operator.handlers.cleanup import cleanup_old_results

        await cleanup_old_results(
            body=_FIXTURE_BODY,
            status={"phase": Phase.COMPLETED},
            name="test-job",
        )

    @pytest.mark.asyncio
    async def test_skips_when_results_dir_not_exists(self) -> None:
        """Verify skips when results directory doesn't exist."""
        from aiperf.operator.handlers.cleanup import cleanup_old_results

        await cleanup_old_results(
            body=_FIXTURE_BODY,
            status={
                "phase": Phase.COMPLETED,
                "jobId": "job-123",
                "resultsPath": "/nonexistent/path",
            },
            name="test-job",
        )

    @pytest.mark.asyncio
    async def test_cleans_up_old_results(self, temp_results_dir: Path) -> None:
        """Verify cleans up results older than TTL."""
        import os

        from aiperf.operator.handlers.cleanup import cleanup_old_results

        results_dir = temp_results_dir / "job-123"
        results_dir.mkdir()
        old_time = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(results_dir, (old_time, old_time))

        with (
            mock_patch("aiperf.operator.events.results_cleaned"),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            await cleanup_old_results(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-123",
                    "resultsPath": str(results_dir),
                    "resultsTtlDays": 30,
                },
                name="test-job",
            )

        assert not results_dir.exists()

    @pytest.mark.asyncio
    async def test_keeps_recent_results(self, temp_results_dir: Path) -> None:
        """Verify keeps results newer than TTL."""
        from aiperf.operator.handlers.cleanup import cleanup_old_results

        results_dir = temp_results_dir / "job-123"
        results_dir.mkdir()

        with mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir):
            await cleanup_old_results(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-123",
                    "resultsPath": str(results_dir),
                    "resultsTtlDays": 30,
                },
                name="test-job",
            )

        assert results_dir.exists()

    @pytest.mark.asyncio
    async def test_handles_cleanup_exception(self, temp_results_dir: Path) -> None:
        """Verify handles exceptions during cleanup gracefully."""
        import os

        from aiperf.operator.handlers.cleanup import cleanup_old_results

        results_dir = temp_results_dir / "job-123"
        results_dir.mkdir()
        old_time = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(results_dir, (old_time, old_time))

        with (
            mock_patch("shutil.rmtree", side_effect=OSError("Permission denied")),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
        ):
            # Should not raise
            await cleanup_old_results(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-123",
                    "resultsPath": str(results_dir),
                    "resultsTtlDays": 30,
                },
                name="test-job",
            )


# =============================================================================
# Test _get_elapsed_seconds
# =============================================================================


class TestGetElapsedSeconds:
    """Tests for _get_elapsed_seconds helper."""

    def test_returns_none_when_no_start_time(self) -> None:
        """Verify returns None when startTime is missing."""
        assert _get_elapsed_seconds({}) is None

    def test_returns_none_when_start_time_empty(self) -> None:
        """Verify returns None when startTime is empty string."""
        assert _get_elapsed_seconds({"startTime": ""}) is None

    def test_returns_positive_elapsed_seconds(self) -> None:
        """Verify returns positive elapsed seconds for a past startTime."""

        past_time = "2020-01-01T00:00:00Z"
        result = _get_elapsed_seconds({"startTime": past_time})
        assert result is not None
        assert result > 0

    def test_returns_small_elapsed_for_recent_start(self) -> None:
        """Verify returns small elapsed for a just-set startTime."""
        from aiperf.operator.status import format_timestamp

        now_ts = format_timestamp()
        result = _get_elapsed_seconds({"startTime": now_ts})
        assert result is not None
        assert result < 5.0

    @pytest.mark.parametrize(
        "bad_value",
        [
            param("not-a-timestamp", id="invalid-format"),
            param("2026-99-99T00:00:00Z", id="invalid-date"),
        ],
    )  # fmt: skip
    def test_returns_none_for_invalid_timestamps(self, bad_value: str) -> None:
        """Verify returns None for unparsable timestamps."""
        assert _get_elapsed_seconds({"startTime": bad_value}) is None


# =============================================================================
# Test _get_job_timeout
# =============================================================================


class TestGetJobTimeout:
    """Tests for _get_job_timeout helper."""

    def test_returns_spec_timeout_when_present(self) -> None:
        """Verify returns timeoutSeconds from spec."""
        assert _get_job_timeout({"timeoutSeconds": 300}) == 300.0

    def test_returns_global_default_when_not_in_spec(self) -> None:
        """Verify falls back to OperatorEnvironment.JOB_TIMEOUT_SECONDS default."""
        assert _get_job_timeout({}) == OperatorEnvironment.JOB_TIMEOUT_SECONDS

    def test_returns_zero_when_spec_is_zero(self) -> None:
        """Verify spec value of 0 means no timeout."""
        assert _get_job_timeout({"timeoutSeconds": 0}) == 0.0

    def test_converts_string_to_float(self) -> None:
        """Verify string values are converted to float."""
        assert _get_job_timeout({"timeoutSeconds": "600"}) == 600.0


# =============================================================================
# Test _get_or_create_progress_client / _close_progress_client
# =============================================================================


class TestProgressClientCache:
    """Tests for _get_or_create_progress_client and _close_progress_client."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear the module-level progress client cache before and after each test."""
        _progress_clients.clear()
        yield
        _progress_clients.clear()

    @pytest.mark.asyncio
    async def test_creates_new_client(self) -> None:
        """Verify creates a new ProgressClient on first call."""
        with mock_patch("aiperf.operator.client_cache.ProgressClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value = mock_client

            client = await _get_or_create_progress_client("job-1")

            assert client is mock_client
            mock_client.__aenter__.assert_called_once()
            assert "job-1" in _progress_clients

    @pytest.mark.asyncio
    async def test_returns_cached_client(self) -> None:
        """Verify returns same client on subsequent calls for same job_id."""
        with mock_patch("aiperf.operator.client_cache.ProgressClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value = mock_client

            client1 = await _get_or_create_progress_client("job-1")
            client2 = await _get_or_create_progress_client("job-1")

            assert client1 is client2
            # Only created once
            assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_different_jobs_get_different_clients(self) -> None:
        """Verify different job_ids get separate clients."""
        call_count = 0

        with mock_patch("aiperf.operator.client_cache.ProgressClient") as mock_cls:

            def make_client(*_args, **_kwargs):
                nonlocal call_count
                call_count += 1
                c = AsyncMock()
                c.__aenter__ = AsyncMock(return_value=c)
                return c

            mock_cls.side_effect = make_client

            client1 = await _get_or_create_progress_client("job-a")
            client2 = await _get_or_create_progress_client("job-b")

            assert client1 is not client2
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_close_removes_and_exits(self) -> None:
        """Verify close calls __aexit__ and removes from cache."""
        mock_client = AsyncMock()
        mock_client.__aexit__ = AsyncMock(return_value=None)
        _progress_clients["job-close"] = mock_client

        await _close_progress_client("job-close")

        assert "job-close" not in _progress_clients
        mock_client.__aexit__.assert_called_once_with(None, None, None)

    @pytest.mark.asyncio
    async def test_close_nonexistent_is_noop(self) -> None:
        """Verify closing a non-existent client does nothing."""
        await _close_progress_client("no-such-job")
        assert "no-such-job" not in _progress_clients


class TestRecoverTerminatedController:
    """Tests for terminated-controller salvage handling."""

    @pytest.fixture(autouse=True)
    def _clear_client_cache_state(self):
        """Clear completion and cancellation state between recovery tests."""
        from aiperf.operator.client_cache import _reset_for_testing

        _reset_for_testing()
        yield
        _reset_for_testing()

    @pytest.mark.asyncio
    async def test_recovers_results_from_sidecar(self) -> None:
        """Verify a terminated controller triggers completion salvage from sidecar files."""
        from aiperf.operator.handlers.monitor import (
            _maybe_recover_terminated_controller,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 1}})

        controller_pod = _make_pod(
            name="controller-0-0",
            container_statuses_raw=[
                {
                    "name": "control-plane",
                    "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
                },
                {
                    "name": "results-sidecar",
                    "state": {},
                },
            ],
        )

        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(
                return_value=_V1PodList(items=[controller_pod])
            )
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=mock_core,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                new_callable=AsyncMock,
                return_value=ControllerFetchResult(
                    metrics=None,
                    downloaded=["profile_export_aiperf.json"],
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle_completion,
        ):
            handled = await _maybe_recover_terminated_controller(
                AsyncMock(),
                {},
                "default",
                "test-jobset",
                "job-1",
                status={"workers": {"total": 1}},
                sb=sb,
                key="default/job-1",
                name="job-1",
            )

        assert handled is True
        mock_handle_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovers_from_on_disk_exports_when_salvage_fetch_returns_empty(
        self, temp_results_dir: Path
    ) -> None:
        """Terminated-controller salvage should trust final exports already on disk.

        The controller/sidecar race can leave ``fetch_results_with_retry``
        returning ``downloaded=[]`` even though the operator's results dir already
        contains the final compressed key exports. Salvage should still route
        through ``handle_completion`` instead of stamping Failed.
        """
        from aiperf.operator.handlers.monitor import (
            _maybe_recover_terminated_controller,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 1}})

        controller_pod = _make_pod(
            name="controller-0-0",
            container_statuses_raw=[
                {
                    "name": "control-plane",
                    "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
                },
                {
                    "name": "results-sidecar",
                    "state": {},
                },
            ],
        )

        run_dir = temp_results_dir / "default" / "job-1" / "1714064523"
        run_dir.mkdir(parents=True)
        summary = orjson.dumps(
            {
                "metrics": {
                    "request_throughput": {
                        "avg": 123.0,
                        "unit": "req/s",
                    }
                }
            }
        )
        (run_dir / "profile_export_aiperf.json.zst").write_bytes(
            zstandard.ZstdCompressor().compress(summary)
        )
        (run_dir / "profile_export_aiperf.csv.zst").write_bytes(b"metric,value\n")

        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(
                return_value=_V1PodList(items=[controller_pod])
            )
        )

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=mock_core,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                new_callable=AsyncMock,
                return_value=ControllerFetchResult(
                    metrics=None,
                    downloaded=[],
                    error="Failed to fetch results: ",
                ),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.handle_completion",
                new_callable=AsyncMock,
            ) as mock_handle_completion,
        ):
            handled = await _maybe_recover_terminated_controller(
                AsyncMock(),
                _FIXTURE_BODY,
                "default",
                "test-jobset",
                "job-1",
                status={"workers": {"total": 1}},
                sb=sb,
                key="default/job-1",
                name="job-1",
            )

        assert handled is True
        mock_handle_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_marks_failed_when_controller_terminated_without_results(
        self,
    ) -> None:
        """Verify unrecoverable controller termination marks the job failed."""
        from aiperf.operator.handlers.monitor import (
            _maybe_recover_terminated_controller,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 1}})

        controller_pod = _make_pod(
            name="controller-0-0",
            container_statuses_raw=[
                {
                    "name": "control-plane",
                    "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
                },
                {"name": "results-sidecar", "state": {}},
            ],
        )

        mock_delete = AsyncMock(return_value={})
        mock_custom = MagicMock(delete_namespaced_custom_object=mock_delete)
        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(
                return_value=_V1PodList(items=[controller_pod])
            )
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=mock_core,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                new_callable=AsyncMock,
                return_value=ControllerFetchResult(metrics=None, downloaded=[]),
            ),
            mock_patch("aiperf.operator.events.failed") as mock_failed_event,
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            handled = await _maybe_recover_terminated_controller(
                AsyncMock(),
                {},
                "default",
                "test-jobset",
                "job-1",
                status={"workers": {"total": 1}},
                sb=sb,
                key="default/job-1",
                name="job-1",
            )

        assert handled is True
        assert kopf_patch.status["phase"] == Phase.FAILED
        mock_failed_event.assert_called_once()
        mock_delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_salvage_when_cancelled_during_fetch(self) -> None:
        """Cancellation during salvage fetch must not stamp terminal status."""
        from aiperf.operator.client_cache import request_cancellation
        from aiperf.operator.handlers.monitor import (
            _maybe_recover_terminated_controller,
        )
        from aiperf.operator.status import StatusBuilder

        class CountingStatusBuilder(StatusBuilder):
            """StatusBuilder test double that records finalize calls."""

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.finalize_count = 0

            def finalize(self) -> None:
                self.finalize_count += 1
                super().finalize()

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = CountingStatusBuilder(kopf_patch, {"workers": {"total": 1}})

        controller_pod = _make_pod(
            name="controller-0-0",
            container_statuses_raw=[
                {
                    "name": "control-plane",
                    "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
                },
                {"name": "results-sidecar", "state": {}},
            ],
        )

        async def fake_fetch(*_args, **_kwargs):
            request_cancellation("default/job-1")
            return ControllerFetchResult(metrics=None, downloaded=[])

        mock_delete = AsyncMock(return_value={})
        mock_custom = MagicMock(delete_namespaced_custom_object=mock_delete)
        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(
                return_value=_V1PodList(items=[controller_pod])
            )
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=mock_core,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                side_effect=fake_fetch,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            mock_patch("aiperf.operator.events.failed") as mock_failed_event,
            mock_patch("aiperf.operator.events.completed") as mock_completed_event,
        ):
            handled = await _maybe_recover_terminated_controller(
                AsyncMock(),
                {},
                "default",
                "test-jobset",
                "job-1",
                status={"workers": {"total": 1}},
                sb=sb,
                key="default/job-1",
                name="job-1",
            )

        assert handled is True
        assert kopf_patch.status.get("phase") not in (Phase.FAILED, Phase.COMPLETED)
        mock_delete.assert_not_awaited()
        assert sb.finalize_count == 0
        mock_failed_event.assert_not_called()
        mock_completed_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_recovers_partial_checkpoint_when_final_export_missing(
        self, temp_results_dir: Path
    ) -> None:
        """Verify checkpoint-only salvage marks the job failed with partial results."""
        from aiperf.operator.handlers.monitor import (
            _maybe_recover_terminated_controller,
        )
        from aiperf.operator.status import ConditionType, StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 1}})

        controller_pod = _make_pod(
            name="controller-0-0",
            container_statuses_raw=[
                {
                    "name": "control-plane",
                    "state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}},
                },
                {"name": "results-sidecar", "state": {}},
            ],
        )

        checkpoint_dir = (
            temp_results_dir / "default" / "job-1" / "1714064523" / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "profile_export_aiperf_partial.json").write_text(
            '{"request_throughput":{"unit":"req/s","avg":123.0}}'
        )

        mock_delete = AsyncMock(return_value={})
        mock_custom = MagicMock(delete_namespaced_custom_object=mock_delete)
        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(
                return_value=_V1PodList(items=[controller_pod])
            )
        )

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CoreV1Api",
                return_value=mock_core,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.fetch_results_with_retry",
                new_callable=AsyncMock,
                return_value=ControllerFetchResult(
                    metrics=None,
                    downloaded=[],
                    checkpoints=["checkpoints/profile_export_aiperf_partial.json"],
                ),
            ),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.results_stored") as mock_results_stored,
            mock_patch("aiperf.operator.events.failed") as mock_failed_event,
            mock_patch(
                "aiperf.operator.handlers.monitor.try_claim_completion",
                new_callable=AsyncMock,
                return_value=True,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
        ):
            handled = await _maybe_recover_terminated_controller(
                AsyncMock(),
                _FIXTURE_BODY,
                "default",
                "test-jobset",
                "job-1",
                status={"workers": {"total": 1}},
                sb=sb,
                key="default/job-1",
                name="job-1",
            )

        assert handled is True
        assert kopf_patch.status["phase"] == Phase.FAILED
        assert kopf_patch.status["resultsPath"] == str(
            temp_results_dir / "default" / "job-1" / "1714064523"
        )
        assert (
            kopf_patch.status["conditions"][0]["type"]
            == ConditionType.RESULTS_AVAILABLE
        )
        # Round-7 fix: partial-checkpoint recovery must stamp runEpoch so
        # K8sChildJobExecutor._fetch_summary_from_operator can resolve the
        # /api/v1/results/<ns>/<job>/runs/<epoch>/ URL. Without this, sweep
        # children that hit checkpoint recovery silently drop out of the
        # parent aggregate even though the artifacts are on disk.
        assert kopf_patch.status.get("runEpoch") == 1714064523, (
            f"runEpoch must be stamped on partial recovery; got "
            f"{kopf_patch.status.get('runEpoch')!r}"
        )
        mock_results_stored.assert_called_once()
        mock_failed_event.assert_called_once()
        mock_delete.assert_awaited_once()


# =============================================================================
# Test monitor_progress - Job Timeout
# =============================================================================


class TestMonitorProgressTimeout:
    """Tests for job timeout detection in monitor_progress."""

    @pytest.mark.asyncio
    async def test_fails_job_on_timeout(self) -> None:
        """Verify monitor_progress fails a job that exceeds its timeout."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        past_time = "2020-01-01T00:00:00Z"

        mock_jobset = AsyncMock()
        mock_jobset.delete = AsyncMock()

        with (
            mock_patch("aiperf.operator.events.job_timeout") as mock_event,
            mock_patch(
                "aiperf.operator.handlers.monitor.close_progress_client",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
        ):
            await monitor_progress(
                body={
                    "metadata": {
                        "name": "timeout-job",
                        "creationTimestamp": _FIXTURE_CREATION_TS,
                    }
                },
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-timeout",
                    "startTime": past_time,
                },
                spec={"timeoutSeconds": 60},
                name="timeout-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.FAILED
        assert "timed out" in kopf_patch.status["error"]
        assert "completionTime" in kopf_patch.status
        mock_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_timeout_when_zero(self) -> None:
        """Verify timeout of 0 means no timeout check."""
        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            }
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                    "startTime": "2020-01-01T00:00:00Z",
                },
                spec={"timeoutSeconds": 0},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        # Should NOT have been set to failed
        assert kopf_patch.status.get("phase") != Phase.FAILED

    @pytest.mark.asyncio
    async def test_no_timeout_when_within_limit(self) -> None:
        """Verify no timeout when elapsed is within the limit."""
        from aiperf.operator.main import monitor_progress
        from aiperf.operator.status import format_timestamp

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_jobset = MagicMock()
        mock_jobset.raw = {
            "status": {
                "conditions": [],
                "replicatedJobsStatus": [],
            }
        }

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=MagicMock(
                    get_namespaced_custom_object=AsyncMock(
                        return_value=mock_jobset.raw
                    ),
                    delete_namespaced_custom_object=AsyncMock(return_value={}),
                ),
            ),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-123",
                    "startTime": format_timestamp(),
                },
                spec={"timeoutSeconds": 3600},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        assert kopf_patch.status.get("phase") != Phase.FAILED


# =============================================================================
# Test _handle_completion - CompletedBeforeMonitor backfill
# =============================================================================


class TestHandleCompletionBackfill:
    """Tests for _handle_completion CompletedBeforeMonitor condition backfill."""

    @pytest.mark.asyncio
    async def test_backfills_workers_ready_condition(
        self, temp_results_dir: Path
    ) -> None:
        """Verify backfills WorkersReady with CompletedBeforeMonitor reason."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 3}})

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed"),
            mock_patch("aiperf.operator.events.results_failed"),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-backfill",
                status={"workers": {"total": 3}},
                sb=sb,
                result=ControllerFetchResult(metrics=None, downloaded=[]),
            )

        # Find the WorkersReady condition
        conditions = kopf_patch.status.get("conditions", [])
        workers_ready = [c for c in conditions if c.get("type") == "WorkersReady"]
        assert len(workers_ready) == 1
        assert workers_ready[0]["reason"] == "CompletedBeforeMonitor"
        assert "3" in workers_ready[0]["message"]

    @pytest.mark.asyncio
    async def test_backfills_benchmark_running_condition(
        self, temp_results_dir: Path
    ) -> None:
        """Verify backfills BenchmarkRunning with CompletedBeforeMonitor reason."""
        from aiperf.operator.handlers.completion import (
            handle_completion as _handle_completion,
        )
        from aiperf.operator.status import StatusBuilder

        kopf_patch = MagicMock()
        kopf_patch.status = {}
        sb = StatusBuilder(kopf_patch, {"workers": {"total": 1}})

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", temp_results_dir),
            mock_patch("aiperf.operator.events.completed"),
            mock_patch("aiperf.operator.events.results_failed"),
            mock_patch(
                "aiperf.operator.handlers.completion.runs_index",
                new=MagicMock(
                    upsert_run_completed=AsyncMock(),
                    upsert_run_failed=AsyncMock(),
                    set_latest=AsyncMock(),
                ),
            ),
        ):
            await _handle_completion(
                body=_FIXTURE_BODY,
                namespace="default",
                jobset_name="test-jobset",
                job_id="job-backfill",
                status={"workers": {"total": 1}},
                sb=sb,
                result=ControllerFetchResult(metrics=None, downloaded=[]),
            )

        conditions = kopf_patch.status.get("conditions", [])
        benchmark_running = [
            c for c in conditions if c.get("type") == "BenchmarkRunning"
        ]
        assert len(benchmark_running) == 1
        assert benchmark_running[0]["reason"] == "CompletedBeforeMonitor"


# =============================================================================
# Test on_create - Preflight Integration
# =============================================================================


class TestOperatorPreflight:
    """Integration tests for on_create handler's preflight check interactions."""

    @pytest.fixture(autouse=True)
    def _mock_persistence(self):
        """Persistence runs before JobSet create (H1); stub for tests."""
        with (
            mock_patch(
                "aiperf.operator.handlers.create.save_job_spec_file",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.runs_index",
                new=MagicMock(upsert_run_created=AsyncMock()),
            ),
        ):
            yield

    @pytest.fixture
    def mock_all_events(self) -> dict[str, MagicMock]:
        """Mock all event functions and return them for assertion."""
        patches = {
            "event_spec_valid": mock_patch("aiperf.operator.events.spec_valid"),
            "event_spec_invalid": mock_patch("aiperf.operator.events.spec_invalid"),
            "event_endpoint_reachable": mock_patch(
                "aiperf.operator.events.endpoint_reachable"
            ),
            "event_endpoint_unreachable": mock_patch(
                "aiperf.operator.events.endpoint_unreachable"
            ),
            "event_resources_created": mock_patch(
                "aiperf.operator.events.resources_created"
            ),
            "event_created": mock_patch("aiperf.operator.events.created"),
            "event_failed": mock_patch("aiperf.operator.events.failed"),
            "event_preflight_passed": mock_patch(
                "aiperf.operator.events.preflight_passed"
            ),
            "event_preflight_failed": mock_patch(
                "aiperf.operator.events.preflight_failed"
            ),
            "event_preflight_warning": mock_patch(
                "aiperf.operator.events.preflight_warning"
            ),
        }
        mocks: dict[str, MagicMock] = {}
        stack = []
        for key, p in patches.items():
            m = p.start()
            mocks[key] = m
            stack.append(p)
        yield mocks
        for p in reversed(stack):
            p.stop()

    def _make_preflight_mock(self, results: Any) -> MagicMock:
        """Build a mock OperatorPreflightChecker that returns *results*."""
        mock_checker = MagicMock()
        mock_checker.run_all = AsyncMock(return_value=results)
        return mock_checker

    @pytest.mark.asyncio
    async def test_endpoint_dns_failure_marks_create_failed(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """DNS endpoint failures fail create-time validation permanently."""
        from aiperf.operator.main import on_create

        full_aiperfjob_spec["benchmark"]["endpoint"]["urls"] = [
            "http://missing.invalid:8000/v1"
        ]
        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        dns_error = (
            "DNS resolution failed for missing.invalid: "
            "[Errno -2] Name or service not known"
        )
        with (
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=EndpointHealthResult(reachable=False, error=dns_error),
            ),
            pytest.raises(kopf.PermanentError, match="Endpoint DNS resolution failed"),
        ):
            await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == "Failed"
        assert "DNS resolution failed" in kopf_patch.status["error"]
        conditions = kopf_patch.status.get("conditions", [])
        endpoint_condition = next(
            c for c in conditions if c["type"] == "EndpointReachable"
        )
        assert endpoint_condition["status"] == "False"
        assert endpoint_condition["reason"] == "EndpointDNSResolutionFailed"
        mock_all_events["event_failed"].assert_called_once()

    @pytest.mark.asyncio
    async def test_preflight_pass_sets_condition_and_creates_resources(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """When preflight passes, PreflightPassed is True and resources are created."""
        from aiperf.kubernetes.preflight import (
            CheckResult,
            CheckStatus,
            PreflightResults,
        )
        from aiperf.operator.main import on_create

        passing_results = PreflightResults()
        passing_results.add(CheckResult("K8s Version", CheckStatus.PASS, "ok"))
        passing_results.add(CheckResult("JobSet CRD", CheckStatus.PASS, "ok"))

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ) as mock_create_cm,
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ) as mock_create_custom,
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=self._make_preflight_mock(passing_results),
            ),
        ):
            result = await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        # PreflightPassed condition is True
        conditions = kopf_patch.status.get("conditions", [])
        preflight_cond = [c for c in conditions if c["type"] == "PreflightPassed"]
        assert len(preflight_cond) == 1
        assert preflight_cond[0]["status"] == "True"

        # event_preflight_passed was called
        mock_all_events["event_preflight_passed"].assert_called_once()

        # Resources were created
        mock_create_cm.assert_awaited_once()
        mock_create_custom.assert_awaited_once()
        assert "jobSetName" in result

    @pytest.mark.asyncio
    async def test_preflight_fail_sets_condition_and_blocks_resources(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """When preflight fails, PreflightPassed is False, phase is Failed, no resources."""
        from aiperf.kubernetes.preflight import (
            CheckResult,
            CheckStatus,
            PreflightResults,
        )
        from aiperf.operator.main import on_create

        failing_results = PreflightResults()
        failing_results.add(
            CheckResult("K8s Version", CheckStatus.FAIL, "Too old: v1.22")
        )

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=self._make_preflight_mock(failing_results),
            ),
            pytest.raises(kopf.PermanentError, match="Pre-flight checks failed"),
        ):
            await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        # PreflightPassed condition is False
        conditions = kopf_patch.status.get("conditions", [])
        preflight_cond = [c for c in conditions if c["type"] == "PreflightPassed"]
        assert len(preflight_cond) == 1
        assert preflight_cond[0]["status"] == "False"

        # Phase is Failed with the check detail in the error
        assert kopf_patch.status["phase"] == Phase.FAILED
        assert "K8s Version" in kopf_patch.status["error"]
        assert "Too old" in kopf_patch.status["error"]

        # event_preflight_failed was called
        mock_all_events["event_preflight_failed"].assert_called_once()

    @pytest.mark.asyncio
    async def test_preflight_warnings_emitted_as_events(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """When preflight passes with warnings, each warning emits an event."""
        from aiperf.kubernetes.preflight import (
            CheckResult,
            CheckStatus,
            PreflightResults,
        )
        from aiperf.operator.main import on_create

        warning_results = PreflightResults()
        warning_results.add(CheckResult("K8s Version", CheckStatus.PASS, "ok"))
        warning_results.add(CheckResult("DNS", CheckStatus.WARN, "CoreDNS not found"))
        warning_results.add(
            CheckResult("Network Policy", CheckStatus.WARN, "Restrictive policy found")
        )

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=self._make_preflight_mock(warning_results),
            ),
        ):
            await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        # event_preflight_warning called once per WARN check
        warn_mock = mock_all_events["event_preflight_warning"]
        assert warn_mock.call_count == 2
        warn_calls = [c.args for c in warn_mock.call_args_list]
        warn_names = {call[1] for call in warn_calls}
        assert warn_names == {"DNS", "Network Policy"}

    @pytest.mark.asyncio
    async def test_preflight_timeout_fails_job(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """When preflight times out, phase is Failed and no resources created."""
        from aiperf.kubernetes.preflight import (
            CheckResult,
            CheckStatus,
            PreflightResults,
        )
        from aiperf.operator.main import on_create

        # Simulate a timeout: the checker adds a Preflight Timeout FAIL entry
        timeout_results = PreflightResults()
        timeout_results.add(
            CheckResult(
                "Preflight Timeout",
                CheckStatus.FAIL,
                "Pre-flight checks timed out after 30s",
            )
        )

        body = {
            "metadata": {
                "name": "test-job",
                "namespace": "default",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                return_value=self._make_preflight_mock(timeout_results),
            ),
            pytest.raises(kopf.PermanentError, match="Pre-flight checks failed"),
        ):
            await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="test-job",
                namespace="default",
                uid="test-uid",
                patch=kopf_patch,
            )

        assert kopf_patch.status["phase"] == Phase.FAILED
        assert "timed out" in kopf_patch.status["error"]

    @pytest.mark.asyncio
    async def test_preflight_receives_correct_parameters(
        self,
        mock_all_events: dict[str, MagicMock],
        full_aiperfjob_spec: dict[str, Any],
    ) -> None:
        """Verify OperatorPreflightChecker is constructed with correct arguments."""
        from aiperf.kubernetes.preflight import PreflightResults
        from aiperf.operator.main import on_create

        passing_results = PreflightResults()

        body = {
            "metadata": {
                "name": "param-job",
                "namespace": "bench-ns",
                "creationTimestamp": _FIXTURE_CREATION_TS,
            }
        }
        kopf_patch = MagicMock()
        kopf_patch.status = {}

        mock_api = AsyncMock()
        AsyncMock()
        AsyncMock()

        mock_checker_cls = MagicMock()
        mock_checker_instance = MagicMock()
        mock_checker_instance.run_all = AsyncMock(return_value=passing_results)
        mock_checker_cls.return_value = mock_checker_instance

        with (
            mock_patch(
                "aiperf.operator.handlers.create.k8s_client",
                return_value=_fake_k8s_client(mock_api),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.check_endpoint_health",
                new_callable=AsyncMock,
                return_value=MagicMock(reachable=True, error=""),
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_config_map",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_custom_object",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.handlers.create.create_idempotent_role_binding",
                new_callable=AsyncMock,
            ),
            mock_patch(
                "aiperf.operator.preflight.OperatorPreflightChecker",
                mock_checker_cls,
            ),
        ):
            await on_create(
                body=body,
                spec=full_aiperfjob_spec,
                name="param-job",
                namespace="bench-ns",
                uid="uid-999",
                patch=kopf_patch,
            )

        # Verify the checker was constructed with expected arguments
        mock_checker_cls.assert_called_once()
        call_kwargs = mock_checker_cls.call_args.kwargs

        assert call_kwargs["api"] is mock_api
        assert call_kwargs["namespace"] == "bench-ns"
        assert call_kwargs["total_workers"] > 0
        assert call_kwargs["num_pods"] > 0

        # deployment is a KubernetesDeployment instance
        from aiperf.kubernetes.resources import KubernetesDeployment

        assert isinstance(call_kwargs["deployment"], KubernetesDeployment)
        assert call_kwargs["deployment"].namespace == "bench-ns"
        assert call_kwargs["deployment"].job_id == "param-job"

        # config and deploy_config are the correct types
        from aiperf.config import AIPerfConfig
        from aiperf.config.deployment import DeploymentConfig

        assert isinstance(call_kwargs["config"], AIPerfConfig)
        assert isinstance(call_kwargs["deploy_config"], DeploymentConfig)

        # run_all was called
        mock_checker_instance.run_all.assert_called_once()


class TestMonitorStaleReadLogging:
    """Verify that stale-read recovery failures surface as logged exceptions
    and that the apiserver hiccup does NOT get misread as benchmark failure."""

    @pytest.mark.asyncio
    async def test_stale_read_exception_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When the fresh CR re-read raises after JobSet NotFound, the handler
        logs the exception (with traceback) AND defers to the next monitor tick
        without stamping FAILED.

        Falling through to FAILED on a fresh-read exception is one half of the
        JobSet-not-found phase-stomp bug — apiserver hiccups must not
        overwrite a (possibly already-Completed) CR.
        """
        import logging

        from aiperf.operator.main import monitor_progress

        kopf_patch = MagicMock()
        kopf_patch.status = {}

        # The monitor makes 2 get_namespaced_custom_object calls:
        # 1. JobSet lookup → 404 (NotFound)
        # 2. Fresh AIPerfJob CR re-read → RuntimeError
        mock_get = AsyncMock(
            side_effect=[
                ApiException(status=404, reason="not found"),
                RuntimeError("api blip"),
            ]
        )
        mock_custom = MagicMock(get_namespaced_custom_object=mock_get)

        with (
            mock_patch(
                "aiperf.operator.handlers.monitor.k8s_client",
                return_value=_fake_k8s_client(MagicMock()),
            ),
            mock_patch(
                "aiperf.operator.handlers.monitor.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            caplog.at_level(logging.ERROR, logger="aiperf.operator.handlers.monitor"),
        ):
            await monitor_progress(
                body=_FIXTURE_BODY,
                status={
                    "phase": Phase.RUNNING,
                    "jobSetName": "test-jobset",
                    "jobId": "job-1",
                },
                spec={},
                name="test-job",
                namespace="default",
                patch=kopf_patch,
            )

        # Apiserver-blip recovery: phase is NOT stamped FAILED. The next
        # monitor tick will re-read fresh state and decide then.
        assert "phase" not in kopf_patch.status
        # The stale-read failure is logged with traceback (logger.exception)
        assert any(
            "Stale-read recovery failed" in rec.message and rec.exc_info is not None
            for rec in caplog.records
        ), (
            f"Expected stale-read exception log, got: {[r.message for r in caplog.records]}"
        )


class TestOpenRunsIndexSelfHeal:
    """`open_runs_index` must survive a corrupt on-disk runs index.

    Regression: the self-heal ran *after* `runs_index.open()`, which raises
    "file is not a database" on a corrupt DB — so the recovery was dead code
    and a corrupt index made the operator crash-loop at startup. The check
    must run before open().
    """

    @pytest.fixture(autouse=True)
    async def _isolate_runs_index(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Stub the fire-and-forget bootstrap scan and guarantee a clean close."""
        from aiperf.operator import runs_index

        monkeypatch.setattr(runs_index, "bootstrap", AsyncMock())
        yield
        await runs_index.close()

    async def test_open_runs_index_self_heals_corrupt_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A corrupt runs index is renamed to `.broken-*` and a fresh DB opens."""
        from aiperf.operator import main as operator_main
        from aiperf.operator import runs_index

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        db_path = tmp_path / ".aiperf_index.sqlite"
        corrupt_bytes = b"corrupt garbage " * 128
        db_path.write_bytes(corrupt_bytes)
        # A stale WAL sidecar that would re-corrupt the fresh DB if replayed.
        (tmp_path / ".aiperf_index.sqlite-wal").write_bytes(b"stale wal")

        await operator_main.open_runs_index()

        # A fresh, working index is now open at the original path.
        assert await runs_index.integrity_check(db_path) is True
        assert await runs_index.get_meta("schema_version") == "1"

        # The corrupt file was preserved for forensics under a `.broken-*` name.
        broken = list(tmp_path.glob(".aiperf_index.sqlite.broken-*"))
        assert len(broken) == 1
        assert broken[0].read_bytes() == corrupt_bytes

    async def test_open_runs_index_fresh_boot_opens_without_broken_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """First boot (no DB yet) opens a fresh index and leaves no `.broken-*`."""
        from aiperf.operator import main as operator_main
        from aiperf.operator import runs_index

        monkeypatch.setattr(OperatorEnvironment.RESULTS, "DIR", tmp_path)
        db_path = tmp_path / ".aiperf_index.sqlite"
        assert not db_path.exists()

        await operator_main.open_runs_index()

        assert await runs_index.get_meta("schema_version") == "1"
        assert list(tmp_path.glob(".aiperf_index.sqlite.broken-*")) == []
