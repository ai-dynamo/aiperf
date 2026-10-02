# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial Kubernetes sweep tests.

Focuses on:
- AIPerfSweep child rollup bucket isolation for cancelled vs failed children.
- ``currentChildRef`` selection across running, pending, terminal, and malformed children.
- ``status.runs[]`` truncation safety nets that keep huge sweeps under the apiserver limit.
- Child metadata passthrough trust boundaries: user keys propagate, sweep keys stay authoritative.
- No-PVC aggregate harvest behavior: the operator mirrors sidecar paths at the results root.

Out of scope: full kopf handler registration and HTTP router rendering; see sibling files
``test_sweep_child_rollup.py``, ``test_aggregate_fetch.py``, and UI sweep-detail tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.config import BenchmarkConfig, BenchmarkRun, SweepVariation
from aiperf.operator.handlers.sweep import _aggregate_fetch as aggregate_fetch
from aiperf.operator.handlers.sweep import (
    _child_phase_buckets,
    _child_runs,
    child_rollup,
)
from aiperf.sweep_controller.k8s_executor import K8sChildJobExecutor

# ============================================================
# Helpers
# ============================================================


def _owned_child(
    *,
    name: str,
    phase: str | None,
    index: str,
    sweep_uid: str = "sweep-uid-7f2a",
    label: str = "concurrency-64",
) -> dict[str, Any]:
    """Build an AIPerfJob child shape as returned by the K8s custom object API."""
    status = {} if phase is None else {"phase": phase}
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "latency-sweep",
                    "uid": sweep_uid,
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "latency-sweep",
                "aiperf.nvidia.com/sweep-uid": sweep_uid,
                "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
                "aiperf.nvidia.com/variation-index": index,
                "aiperf.nvidia.com/variation-label": label,
            },
        },
        "status": status,
    }


def _triggering_child_body(
    *, name: str = "latency-sweep-v02-t1", phase: str = "Running"
) -> dict[str, Any]:
    """Build the child body kopf passes to the rollup field handler."""
    return {
        "metadata": {
            "name": name,
            "namespace": "aiperf-benchmarks",
            "uid": f"{name}-uid",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "latency-sweep",
                    "uid": "sweep-uid-7f2a",
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "latency-sweep",
                "aiperf.nvidia.com/sweep-uid": "sweep-uid-7f2a",
                "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
            },
        },
        "status": {"phase": phase},
    }


def _benchmark_config_for_child() -> BenchmarkConfig:
    """Build a real BenchmarkConfig so child metadata tests exercise Pydantic output."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 64,
                }
            ],
        }
    )


def _sweep_cr_with_child_metadata(
    child_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal AIPerfSweep CR accepted by K8sChildJobExecutor helpers."""
    spec: dict[str, Any] = {
        "image": "nvcr.io/nvidia/aiperf:adversarial",
        "podTemplate": {},
        "benchmark": {
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 1,
                }
            ],
        },
    }
    if child_metadata is not None:
        spec["childMetadata"] = child_metadata
    return {
        "metadata": {
            "name": "latency-sweep",
            "namespace": "aiperf-benchmarks",
            "uid": "sweep-uid-7f2a",
        },
        "spec": spec,
    }


def _benchmark_run_for_child() -> BenchmarkRun:
    """Build a real BenchmarkRun carrying variation metadata for child CR creation."""
    return BenchmarkRun(
        benchmark_id="bench-7f2a",
        cfg=_benchmark_config_for_child(),
        variation=SweepVariation(
            index=7,
            label="concurrency=64",
            values={"phases.profiling.concurrency": 64},
        ),
        trial=2,
        label="run_0008",
        artifact_dir=Path("/results/aiperf-benchmarks/latency-sweep"),
    )


def _fake_progress_client(downloaded: list[str] | BaseException) -> MagicMock:
    """Create an async ProgressClient double with sidecar list+download behavior."""
    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=None)
    if isinstance(downloaded, BaseException):
        fake.get_results_list = AsyncMock(side_effect=downloaded)
        fake.download_all_results = AsyncMock(side_effect=downloaded)
    else:
        fake.get_results_list = AsyncMock(
            return_value=[{"name": name, "size": 64} for name in downloaded]
        )
        fake.download_all_results = AsyncMock(return_value=downloaded)
    return fake


# ============================================================
# Child phase rollup buckets and current-child selection
# ============================================================


class TestChildRollupBuckets:
    """Adversarial phase bucketing for AIPerfSweep.status.runStates."""

    def test_tally_children_cancelled_terminal_does_not_increment_failed(self) -> None:
        children = [
            _owned_child(name="latency-sweep-v00-t1", phase="Cancelled", index="00"),
            _owned_child(name="latency-sweep-v01-t1", phase="Failed", index="01"),
            _owned_child(
                name="latency-sweep-v02-t1", phase="PartiallyFailed", index="02"
            ),
            _owned_child(name="latency-sweep-v03-t1", phase="Completed", index="03"),
        ]

        counts = _child_phase_buckets._tally_children(
            children,
            sweep_uid="sweep-uid-7f2a",
            sweep_name="latency-sweep",
            run_epoch="1778027124",
        )

        assert counts["cancelled"] == 1
        assert counts["failed"] == 2
        assert counts["completed"] == 1
        assert counts["in_flight"] == 0
        assert counts["total_terminal_phase"] == "Aggregating"

    def test_find_current_child_prefers_lowest_running_over_lower_pending(self) -> None:
        children = [
            _owned_child(name="latency-sweep-v00-t1", phase="Pending", index="00"),
            _owned_child(name="latency-sweep-v03-t1", phase="Running", index="03"),
            _owned_child(name="latency-sweep-v02-t1", phase="Processing", index="02"),
        ]

        current = _child_phase_buckets._find_current_child(children)

        assert current is not None
        assert current["metadata"]["name"] == "latency-sweep-v02-t1"

    def test_find_current_child_malformed_index_sorts_behind_valid_pending(
        self,
    ) -> None:
        children = [
            _owned_child(
                name="latency-sweep-vbad-t1", phase="Pending", index="not-int"
            ),
            _owned_child(name="latency-sweep-v04-t1", phase="Pending", index="04"),
        ]

        current = _child_phase_buckets._find_current_child(children)

        assert current is not None
        assert current["metadata"]["name"] == "latency-sweep-v04-t1"


class TestChildRollupHandler:
    """Handler-level status patch contracts that depend on multiple helpers."""

    @pytest.mark.asyncio
    async def test_on_child_phase_transition_cancelled_counts_toward_max_total_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        phase_calls: list[dict[str, str]] = []

        async def fake_phase_set(
            *,
            namespace: str,
            name: str,
            expect_phase: str,
            new_phase: str,
            expected_uid: str | None = None,
            api: object | None = None,
        ) -> None:
            phase_calls.append(
                {
                    "namespace": namespace,
                    "name": name,
                    "expect": expect_phase,
                    "new": new_phase,
                }
            )

        monkeypatch.setattr(child_rollup, "_patch_parent_status", AsyncMock())
        current_body = _triggering_child_body(
            name="latency-sweep-v03-t1", phase="Cancelled"
        )
        monkeypatch.setattr(
            child_rollup, "_read_current_child", AsyncMock(return_value=current_body)
        )
        monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", fake_phase_set)
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 4}),
        )
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "pending": 0,
                    "running": 0,
                    "completed": 2,
                    "failed": 1,
                    "cancelled": 1,
                    "in_flight": 0,
                    "total_terminal_phase": "Aggregating",
                    "owned_children": [],
                }
            ),
        )

        async def fake_k8s_client() -> object:
            return SimpleNamespace()

        class _ClientContext:
            async def __aenter__(self) -> object:
                return await fake_k8s_client()

            async def __aexit__(self, *_exc: object) -> None:
                return None

        import aiperf.kubernetes.client as kclient

        monkeypatch.setattr(kclient, "k8s_client", lambda: _ClientContext())

        await child_rollup.on_child_phase_transition(
            body=current_body,
            status={"phase": "Cancelled"},
            name="latency-sweep-v03-t1",
            namespace="aiperf-benchmarks",
        )

        assert phase_calls == [
            {
                "namespace": "aiperf-benchmarks",
                "name": "latency-sweep",
                "expect": "Running",
                "new": "Aggregating",
            }
        ]

    @pytest.mark.asyncio
    async def test_on_child_phase_transition_patches_current_child_ref_from_running_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_patch: dict[str, Any] = {}
        children = [
            _owned_child(name="latency-sweep-v00-t1", phase="Completed", index="00"),
            _owned_child(
                name="latency-sweep-v04-t1",
                phase="Profiling",
                index="04",
                label="gpu-count-8",
            ),
        ]

        async def fake_patch(*, body: dict[str, Any], **_kwargs: object) -> None:
            captured_patch.update(body)

        monkeypatch.setattr(child_rollup, "_patch_parent_status", fake_patch)
        current_body = _triggering_child_body(
            name="latency-sweep-v04-t1", phase="Profiling"
        )
        monkeypatch.setattr(
            child_rollup, "_read_current_child", AsyncMock(return_value=current_body)
        )
        monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", AsyncMock())
        monkeypatch.setattr(child_rollup, "_read_parent_status", AsyncMock())
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "pending": 0,
                    "running": 1,
                    "completed": 1,
                    "failed": 0,
                    "cancelled": 0,
                    "in_flight": 1,
                    "total_terminal_phase": None,
                    "owned_children": children,
                }
            ),
        )

        class _ClientContext:
            async def __aenter__(self) -> object:
                return SimpleNamespace()

            async def __aexit__(self, *_exc: object) -> None:
                return None

        import aiperf.kubernetes.client as kclient

        monkeypatch.setattr(kclient, "k8s_client", lambda: _ClientContext())

        await child_rollup.on_child_phase_transition(
            body=current_body,
            status={"phase": "Profiling"},
            name="latency-sweep-v04-t1",
            namespace="aiperf-benchmarks",
        )

        assert captured_patch["status"]["currentChildRef"] == {
            "name": "latency-sweep-v04-t1",
            "index": 4,
            "label": "gpu-count-8",
        }


# ============================================================
# runs[] truncation and malformed-state safety nets
# ============================================================


class TestSweepRunsTruncation:
    """Safety nets for huge or malformed AIPerfSweep.status.runs payloads."""

    @pytest.mark.asyncio
    async def test_append_run_entry_at_safety_threshold_stamps_truncated_not_append(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing_runs = [
            {"index": idx, "childName": f"latency-sweep-v{idx:04d}-t1"}
            for idx in range(1500)
        ]
        fake_custom = MagicMock()
        fake_custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "rv-1500"},
                "status": {"runs": existing_runs, "totalVariations": 2000},
            }
        )
        fake_custom.patch_namespaced_custom_object_status = AsyncMock()

        import kubernetes_asyncio.client as k8s_client

        monkeypatch.setattr(k8s_client, "CustomObjectsApi", lambda _api: fake_custom)

        await _child_runs.append_run_entry(
            "aiperf-benchmarks",
            "latency-sweep",
            {"index": 1500, "childName": "latency-sweep-v1500-t1"},
            api=MagicMock(),
        )

        patch_body = (
            fake_custom.patch_namespaced_custom_object_status.await_args.kwargs["body"]
        )
        assert patch_body == {
            "status": {
                "runsTruncated": {
                    "total": 2000,
                    "included": 1500,
                    "fetchURL": "http://aiperf-operator.aiperf-system:8081/api/v1/sweeps/aiperf-benchmarks/latency-sweep/children",
                }
            }
        }
        assert (
            fake_custom.patch_namespaced_custom_object_status.await_args.kwargs[
                "_content_type"
            ]
            == "application/merge-patch+json"
        )

    @pytest.mark.asyncio
    async def test_append_run_entry_non_list_runs_stamps_truncated_without_replace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_custom = MagicMock()
        fake_custom.get_namespaced_custom_object = AsyncMock(
            return_value={
                "metadata": {"resourceVersion": "rv-corrupt"},
                "status": {"runs": {"not": "a-list"}, "totalVariations": "9"},
            }
        )
        fake_custom.patch_namespaced_custom_object_status = AsyncMock()

        import kubernetes_asyncio.client as k8s_client

        monkeypatch.setattr(k8s_client, "CustomObjectsApi", lambda _api: fake_custom)

        await _child_runs.append_run_entry(
            "aiperf-benchmarks",
            "latency-sweep",
            {"index": 3, "childName": "latency-sweep-v03-t1"},
            api=MagicMock(),
        )

        patch_body = (
            fake_custom.patch_namespaced_custom_object_status.await_args.kwargs["body"]
        )
        assert patch_body["status"]["runsTruncated"] == {
            "total": 9,
            "included": 0,
            "fetchURL": "http://aiperf-operator.aiperf-system:8081/api/v1/sweeps/aiperf-benchmarks/latency-sweep/children",
        }
        assert "runs" not in patch_body["status"]


# ============================================================
# Child metadata passthrough trust boundary
# ============================================================


class TestChildMetadataPassthrough:
    """AIPerfSweep childMetadata merges user keys without ceding sweep-owned keys."""

    def test_build_child_metadata_user_annotation_cannot_override_variation_values(
        self,
    ) -> None:
        sweep = _sweep_cr_with_child_metadata(
            {
                "labels": {"team": "ai-platform"},
                "annotations": {
                    "runbook": "https://runbooks.example/aiperf-sweeps",
                    "aiperf.nvidia.com/variation-values": "attacker-controlled",
                },
            }
        )
        executor = K8sChildJobExecutor(
            api=None,
            sweep=sweep,
            with_trial_suffix=True,
            sweep_run_epoch="1778027124",
        )

        metadata = executor._build_child_metadata(
            _benchmark_run_for_child(), "latency-sweep-v07-t2"
        )

        assert metadata["labels"]["team"] == "ai-platform"
        assert (
            metadata["annotations"]["runbook"]
            == "https://runbooks.example/aiperf-sweeps"
        )
        variation_values = orjson.loads(
            metadata["annotations"]["aiperf.nvidia.com/variation-values"]
        )
        assert variation_values == {"phases.profiling.concurrency": 64}

    def test_build_child_spec_strips_snake_case_child_metadata(self) -> None:
        sweep = _sweep_cr_with_child_metadata()
        sweep["spec"]["child_metadata"] = {"labels": {"team": "ai-platform"}}
        executor = K8sChildJobExecutor(api=None, sweep=sweep, with_trial_suffix=True)

        spec = executor._build_child_spec(_benchmark_run_for_child())

        assert "child_metadata" not in spec
        assert "childMetadata" not in spec


# ============================================================
# Sweep aggregate no-PVC harvest
# ============================================================


class TestSweepAggregateFetchNoPvc:
    """The operator mirrors the sweep-controller emptyDir through the sidecar."""

    @pytest.mark.asyncio
    async def test_fetch_sweep_aggregate_downloads_to_results_root_not_epoch_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _fake_progress_client(
            [
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/aggregate.json",
                "aiperf-benchmarks/sweeps/latency-sweep/1778027124/children.json",
            ]
        )
        monkeypatch.setattr(
            aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_client
        )

        result = await aggregate_fetch.fetch_sweep_aggregate_to_disk(
            sweep_name="latency-sweep",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

        assert result.downloaded == 2
        assert result.listed == 2
        fake_client.download_all_results.assert_awaited_once()
        host_arg, dest_arg = fake_client.download_all_results.await_args.args
        assert (
            host_arg
            == "aiperf-latency-sweep-controller-0-0.aiperf-latency-sweep.aiperf-benchmarks.svc.cluster.local"
        )
        assert dest_arg == tmp_path
        assert not (
            tmp_path / "aiperf-benchmarks" / "sweeps" / "latency-sweep" / "latest.txt"
        ).exists()

    @pytest.mark.asyncio
    async def test_fetch_sweep_aggregate_timeout_returns_zero_without_pointer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = _fake_progress_client(TimeoutError("sidecar pre-stop race"))
        pointer_writes: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_client
        )
        monkeypatch.setattr(
            aggregate_fetch,
            "_write_sweep_latest_pointer",
            lambda *args: pointer_writes.append(args),
        )

        result = await aggregate_fetch.fetch_sweep_aggregate_to_disk(
            sweep_name="latency-sweep",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

        assert result.downloaded == 0
        assert pointer_writes == []


def test_cancel_before_child_start_is_bucketed_as_cancelled() -> None:
    """A child cancelled before it started is cancelled, not failed.

    _is_cancelled_result uses RunResult.was_cancelled, so cancelling a
    20-variation sweep after variation 3 must not report the remaining
    variations as failed or trip the failure policy.
    """
    from aiperf.orchestrator.models import RunResult
    from aiperf.sweep_controller.main import _is_cancelled_result

    result = RunResult(
        label="v3",
        success=False,
        error="sweep cancelled before child started",
        was_cancelled=True,
    )
    assert _is_cancelled_result(result) is True
