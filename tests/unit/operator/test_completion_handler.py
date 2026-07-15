# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import kopf
import orjson
import pytest
import zstandard

from aiperf.kubernetes.constants import Annotations
from aiperf.operator.environment import _ResultsSettings
from aiperf.operator.handlers.completion import handle_completion
from aiperf.operator.models import ControllerFetchResult
from aiperf.operator.status import ConditionType, Phase, StatusBuilder

# metadata.creationTimestamp that maps to epoch 1714064523 — stable fixture
# so the epoch-keyed run dir under results_layout.run_dir is deterministic.
_FIXTURE_CREATION_TS = "2024-04-25T17:02:03Z"
_FIXTURE_BODY = {"metadata": {"creationTimestamp": _FIXTURE_CREATION_TS}}


def _results_settings(tmp_path: Path, **overrides: float) -> _ResultsSettings:
    """Build REAL results settings for patching OperatorEnvironment.RESULTS.

    A partial MagicMock stub auto-creates whatever field the handler reads, so
    field-name drift or numeric-comparison bugs silently no-op; a real
    ``_ResultsSettings`` instance raises AttributeError on unknown reads and
    keeps every unset field at its production default.
    """
    return _ResultsSettings(DIR=tmp_path, RETAIN_RUNS=5, **overrides)


@pytest.mark.asyncio
async def test_handle_completion_without_result_files_marks_failed() -> None:
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 90}})

    result = ControllerFetchResult(
        metrics={
            "aiperf_version": "0.6.0",
            "benchmark_id": "bench-1",
            "model": "mock",
            "endpoint_type": "chat",
            "streaming": True,
            "concurrency": 450000,
            "request_rate": None,
            "metrics": {},
        },
        downloaded=[],
        error="controller terminated before results were recoverable",
    )

    with (
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=_FIXTURE_BODY,
            namespace="test-ns",
            jobset_name="test-jobset",
            job_id="test-job",
            status={"workers": {"total": 90}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED
    # Lookup by type — finalize() appends derived Complete/Failed conditions
    # after handler-set ones, so positional indexing is brittle.
    by_type = {c["type"]: c for c in patch_obj.status["conditions"]}
    assert by_type["ResultsAvailable"]["status"] == "False"


@pytest.mark.asyncio
async def test_handle_completion_has_files_with_error_marks_failed(
    tmp_path: Path,
) -> None:
    """M3: ControllerFetchResult with has_files=True AND error set should NOT be Completed.

    A partial fetch can download some files (e.g. checkpoints) while still
    missing the key export artifacts. The error field is authoritative.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics=None,
        downloaded=["profile_export_aiperf.json", "checkpoints/aggregator-0.parquet"],
        error="key artifact write interrupted; retry needed",
    )

    results_failed_mock = MagicMock()
    completed_mock = MagicMock()
    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(tmp_path),
        ),
        patch(
            "aiperf.operator.handlers.completion.events.results_failed",
            results_failed_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.events.completed",
            completed_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.events.results_stored",
        ),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=_FIXTURE_BODY,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED
    # completed event must NOT fire on partial/errored result
    completed_mock.assert_not_called()
    # results_failed event must fire with the authoritative error message
    results_failed_mock.assert_called_once()
    _, kwargs = results_failed_mock.call_args
    args = results_failed_mock.call_args.args
    assert "key artifact write interrupted" in args[1]
    ra = next(
        c for c in patch_obj.status["conditions"] if c["type"] == "ResultsAvailable"
    )
    assert ra["status"] == "False"


@pytest.mark.asyncio
async def test_handle_completion_index_failure_sets_condition_and_event(
    tmp_path: Path,
) -> None:
    """M1: runs_index upsert failure should set INDEX_UPDATED=False and warn.

    Results are already on disk, so we must not retry the completion handler;
    instead surface the failure via a condition + Warning event.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics={"metrics": {"latency": 1.0}},
        downloaded=["profile_export_aiperf.json"],
        error="",
    )

    kopf_event_mock = MagicMock()
    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(tmp_path),
        ),
        patch("aiperf.operator.handlers.completion.events.results_stored"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(side_effect=RuntimeError("disk full")),
                upsert_run_failed=AsyncMock(side_effect=RuntimeError("disk full")),
                set_latest=AsyncMock(),
            ),
        ),
        patch(
            "aiperf.operator.events.kopf.warn",
            kopf_event_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.k8s_client",
        ),
        patch(
            "aiperf.operator.handlers.completion.client.CustomObjectsApi",
            return_value=MagicMock(
                delete_namespaced_custom_object=AsyncMock(return_value={})
            ),
        ),
    ):
        await handle_completion(
            body={
                "metadata": {
                    "name": "j1",
                    "namespace": "ns",
                    "creationTimestamp": _FIXTURE_CREATION_TS,
                }
            },
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-03-26T00:00:00Z"},
            sb=sb,
            result=result,
        )

    # Phase still reflects the actual result (Completed) — index failure
    # doesn't flip the job to FAILED, results are on disk.
    assert patch_obj.status["phase"] == Phase.COMPLETED
    # INDEX_UPDATED condition was set to False
    index_cond = next(
        (c for c in patch_obj.status["conditions"] if c["type"] == "IndexUpdated"),
        None,
    )
    assert index_cond is not None
    assert index_cond["status"] == "False"
    assert index_cond["reason"] == "IndexUpdateFailed"
    # Warning event was emitted via the events.index_update_failed wrapper
    # (post_event -> kopf.warn for Warning-type events).
    kopf_event_mock.assert_called_once()
    assert kopf_event_mock.call_args.kwargs["reason"] == "IndexUpdateFailed"


@pytest.mark.asyncio
async def test_condition_type_index_updated_exists() -> None:
    """Sanity: ConditionType.INDEX_UPDATED enum value exists and is spelled right."""
    assert ConditionType.INDEX_UPDATED == "IndexUpdated"


def _body_with_claim(claim_age_seconds: float) -> dict:
    """Build a CR body with a completion-claim annotation aged by N seconds.

    The claim timestamp value follows ``status.format_timestamp`` (ISO 8601
    with Z suffix) so ``_claim_age_seconds`` round-trips it correctly.
    """
    claimed_at = datetime.now(UTC) - timedelta(seconds=claim_age_seconds)
    ts = claimed_at.isoformat().replace("+00:00", "Z")
    return {
        "metadata": {
            "creationTimestamp": _FIXTURE_CREATION_TS,
            "annotations": {Annotations.COMPLETION_CLAIMED: ts},
        },
    }


@pytest.mark.asyncio
async def test_handle_completion_transient_fetch_failure_raises_temporary_error(
    tmp_path: Path,
) -> None:
    """Reproduces the CompletedBeforeMonitor -> ResultsFetchFailed race.

    Sub-second benchmarks: controller wrote the readiness marker AND the
    benchmark succeeded, but the operator's HTTP fetch missed the export
    window because the controller container is mid-teardown. The fetch
    returns ``error`` set with no key files. With a fresh completion claim,
    ``handle_completion`` must raise ``kopf.TemporaryError`` so the next
    monitor tick can retry via the orphan-claim recovery path — instead of
    flipping the CR straight to ``Phase.FAILED`` with ``ResultsFetchFailed``.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    # The race shape: error captured, key files NOT downloaded.
    result = ControllerFetchResult(
        metrics={"metrics": {"throughput": 100}},  # primary fetch returned metrics
        downloaded=[],  # but key files never landed
        error="ConnectionResetError: controller closed mid-stream",
    )

    body = _body_with_claim(claim_age_seconds=2.0)  # well within budget

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
        pytest.raises(kopf.TemporaryError) as exc_info,
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    # No terminal phase should have been written; conditions never finalize().
    assert "phase" not in patch_obj.status or patch_obj.status["phase"] not in (
        Phase.FAILED,
        Phase.COMPLETED,
    )
    assert "conditions" not in patch_obj.status
    assert "j1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_completion_partial_results_missing_key_exports_raises_temporary_error(
    tmp_path: Path,
) -> None:
    """Partial downloads without key exports must take the transient-retry path.

    DGX-scale fetches can return metrics plus non-key artifacts (for example
    inputs + partial checkpoints) while the controller is still finalizing the
    authoritative exports. That shape used to surface with ``error=''``, which
    bypassed ``maybe_raise_for_transient_fetch_failure`` and failed the CR
    immediately instead of retrying on the next monitor tick.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics={"metrics": {"throughput": 100}},
        downloaded=[
            "inputs.json",
            "gpu_telemetry_export.jsonl",
            "checkpoints/profile_export_aiperf_partial.json",
        ],
        error="",
    )
    body = _body_with_claim(claim_age_seconds=2.0)

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
        pytest.raises(kopf.TemporaryError) as exc_info,
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert "phase" not in patch_obj.status or patch_obj.status["phase"] not in (
        Phase.FAILED,
        Phase.COMPLETED,
    )
    assert "conditions" not in patch_obj.status
    assert "j1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_completion_transient_fetch_recovers_from_on_disk_key_exports(
    tmp_path: Path,
) -> None:
    """If the operator already has the final compressed key exports on disk,
    completion must recover from them instead of stamping Failed.

    DGX reruns showed a narrower race after the first fix: the controller HTTP
    fetch path could still stall and return ``downloaded=[]`` / empty metrics,
    but the operator's results dir already contained ``profile_export_aiperf``
    key files. That is authoritative evidence the run is recoverable.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    epoch = str(
        int(
            datetime.strptime(_FIXTURE_CREATION_TS, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=UTC)
            .timestamp()
        )
    )
    dest_dir = tmp_path / "ns" / "j1" / epoch
    dest_dir.mkdir(parents=True)
    payload = orjson.dumps(
        {
            "metrics": {
                "request_throughput": {
                    "avg": 123.4,
                    "unit": "requests/s",
                }
            }
        }
    )
    (dest_dir / "profile_export_aiperf.json.zst").write_bytes(
        zstandard.ZstdCompressor().compress(payload)
    )
    (dest_dir / "profile_export_aiperf.csv.zst").write_bytes(b"metric,value\n")

    result = ControllerFetchResult(
        metrics=None,
        downloaded=[],
        error="Failed to fetch results: ",
    )
    body = {"metadata": {"creationTimestamp": _FIXTURE_CREATION_TS}}

    results_failed_mock = MagicMock()
    completed_mock = MagicMock()
    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=0.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch(
            "aiperf.operator.handlers.completion.events.results_failed",
            results_failed_mock,
        ),
        patch(
            "aiperf.operator.handlers.completion.events.completed",
            completed_mock,
        ),
        patch("aiperf.operator.handlers.completion.events.results_stored"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
        patch(
            "aiperf.operator.handlers.completion.k8s_client",
        ),
        patch(
            "aiperf.operator.handlers.completion.client.CustomObjectsApi",
            return_value=MagicMock(
                delete_namespaced_custom_object=AsyncMock(return_value={})
            ),
        ),
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.COMPLETED
    assert patch_obj.status["resultsPath"] == str(dest_dir)
    results_failed_mock.assert_not_called()
    completed_mock.assert_called_once()


@pytest.mark.asyncio
async def test_handle_completion_transient_fetch_failure_after_budget_marks_failed(
    tmp_path: Path,
) -> None:
    """Past the retry budget, the operator gives up and marks Failed.

    Bounds the retry path so a permanently-broken controller doesn't keep
    the CR Pending forever — the budget is wall-clock from the completion
    claim timestamp.
    """
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics={"metrics": {"throughput": 100}},
        downloaded=[],
        error="ConnectionResetError: controller closed mid-stream",
    )

    # Claim aged past the budget — must NOT retry.
    body = _body_with_claim(claim_age_seconds=120.0)

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    # Budget exhausted -> legacy Failed path runs.
    assert patch_obj.status["phase"] == Phase.FAILED
    ra = next(
        c for c in patch_obj.status["conditions"] if c["type"] == "ResultsAvailable"
    )
    assert ra["reason"] == "ResultsFetchFailed"


@pytest.mark.asyncio
async def test_handle_completion_transient_fetch_no_claim_annotation_falls_through(
    tmp_path: Path,
) -> None:
    """Without a parseable claim timestamp, the retry has no anchor and we
    fall back to the legacy Failed path so the CR doesn't loop forever."""
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics={"metrics": {"throughput": 100}},
        downloaded=[],
        error="ConnectionResetError",
    )

    body = {"metadata": {"creationTimestamp": _FIXTURE_CREATION_TS}}  # no annotation

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED


@pytest.mark.asyncio
async def test_handle_completion_transient_retry_disabled_marks_failed(
    tmp_path: Path,
) -> None:
    """``TRANSIENT_FETCH_RETRY_BUDGET_SEC=0`` disables the retry entirely;
    the legacy Failed path is preserved for operators who want the old
    behaviour."""
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    result = ControllerFetchResult(
        metrics={"metrics": {"throughput": 100}},
        downloaded=[],
        error="ConnectionResetError",
    )
    body = _body_with_claim(claim_age_seconds=2.0)

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=0.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    assert patch_obj.status["phase"] == Phase.FAILED


@pytest.mark.asyncio
async def test_handle_completion_partial_fetch_with_files_does_not_retry(
    tmp_path: Path,
) -> None:
    """If at least one key file made it down, this is NOT the controller-
    shutdown race — fall through to the legacy partial-failure path so the
    operator can promote whatever artifacts exist (e.g. via file-derived
    metrics)."""
    patch_obj = MagicMock()
    patch_obj.status = {}
    sb = StatusBuilder(patch_obj, existing_status={"workers": {"total": 1}})

    # has_files=True (key file present) but error set (e.g. checkpoints
    # interrupted mid-write). The legacy "partial fetch" path handles this.
    result = ControllerFetchResult(
        metrics=None,
        downloaded=["profile_export_aiperf.json", "checkpoints/agg-0.parquet"],
        error="checkpoint write interrupted",
    )
    body = _body_with_claim(claim_age_seconds=2.0)

    with (
        patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(
                tmp_path,
                TRANSIENT_FETCH_RETRY_BUDGET_SEC=60.0,
                TRANSIENT_FETCH_RETRY_DELAY_SEC=5.0,
            ),
        ),
        patch("aiperf.operator.handlers.completion.events.results_failed"),
        patch("aiperf.operator.handlers.completion.events.completed"),
        patch("aiperf.operator.handlers.completion.events.results_stored"),
        patch(
            "aiperf.operator.handlers.completion.runs_index",
            new=MagicMock(
                upsert_run_completed=AsyncMock(),
                upsert_run_failed=AsyncMock(),
                set_latest=AsyncMock(),
            ),
        ),
    ):
        await handle_completion(
            body=body,
            namespace="ns",
            jobset_name="js",
            job_id="j1",
            status={"workers": {"total": 1}, "startTime": "2026-04-29T00:00:00Z"},
            sb=sb,
            result=result,
        )

    # has_files + has_error -> legacy Failed path (authoritative error).
    assert patch_obj.status["phase"] == Phase.FAILED


@pytest.mark.asyncio
async def test_run_retention_pass_offloads_prune_and_schedules_index_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rmtree prune must run OFF the event loop, and index drops must
    still be scheduled ON the loop for every deleted epoch.

    Regression lock for the to_thread offload: `_schedule_index_drop`
    resolves `asyncio.get_running_loop()`, which raises inside a worker
    thread — if scheduling stayed inside `enforce_retention` (as it did
    before the offload), the drops would silently no-op and the index would
    keep rows for pruned runs until the next bootstrap.
    """
    import asyncio
    import os
    import threading

    from aiperf.operator import runs_index
    from aiperf.operator.handlers import completion

    epochs = ["1714000000", "1714000060", "1714000120"]
    for i, epoch in enumerate(epochs):
        run = tmp_path / "ns" / "job-a" / epoch
        run.mkdir(parents=True)
        os.utime(run, (i + 1, i + 1))  # strictly increasing mtimes

    monkeypatch.setattr(
        completion.OperatorEnvironment,
        "RESULTS",
        _ResultsSettings(DIR=tmp_path, RETAIN_RUNS=1, RETAIN_DAYS=0),
    )

    # _schedule_index_drop lazily imports the REAL runs_index module; gate it
    # open and record delete_run calls.
    delete_run = AsyncMock()
    monkeypatch.setattr(runs_index, "is_open", lambda: True)
    monkeypatch.setattr(runs_index, "is_readonly", lambda: False)
    monkeypatch.setattr(runs_index, "delete_run", delete_run)

    # Track which thread performs the rmtree so we can assert the prune ran
    # off-loop (a slow PVC delete must not stall kopf).
    import shutil as _shutil

    loop_thread = threading.current_thread()
    prune_threads: list[threading.Thread] = []
    real_rmtree = _shutil.rmtree

    def _tracking_rmtree(path, *args, **kwargs):
        prune_threads.append(threading.current_thread())
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("shutil.rmtree", _tracking_rmtree)

    await completion._run_retention_pass("ns", "job-a", epochs[-1])

    # Let the fire-and-forget delete_run tasks run to completion.
    for _ in range(5):
        await asyncio.sleep(0)

    assert len(prune_threads) == 2, "keep=1 must prune the two oldest epochs"
    assert all(t is not loop_thread for t in prune_threads), (
        "retention rmtree must run in a worker thread, not on the kopf loop"
    )
    dropped = {call.args for call in delete_run.await_args_list}
    assert dropped == {("ns", "job-a", epochs[0]), ("ns", "job-a", epochs[1])}
    assert not (tmp_path / "ns" / "job-a" / epochs[0]).exists()
    assert not (tmp_path / "ns" / "job-a" / epochs[1]).exists()
    assert (tmp_path / "ns" / "job-a" / epochs[2]).exists()
