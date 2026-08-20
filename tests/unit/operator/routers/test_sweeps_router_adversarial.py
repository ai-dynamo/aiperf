# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator sweeps HTTP router.

Focuses on:
- namespace/name URL encoding at the ``/api/v1/sweeps`` route boundary
- missing live/archive sweeps vs Kubernetes API failures
- malformed live ``status.runStates`` and ``status.currentChildRef`` payloads
- ``runsTruncated.fetchURL`` and response-schema stability for dashboard clients
- live child label propagation into the children manifest response

Out of scope: sweep artifact download traversal, covered by the sibling results-files
and sweep-artifacts route tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from pathlib import Path
from typing import cast

import httpx
import orjson
import pytest
from aiperf.operator.sweep_union import SweepRecord
from fastapi import FastAPI
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.operator import sweep_union as union_mod
from aiperf.operator.results_layout import write_sweep_latest
from aiperf.operator.routers import _sweeps_live as live_mod
from aiperf.operator.routers import sweeps as mod
from aiperf.operator.routers.sweeps import create_sweeps_router

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714150923"


class _FakeCustomObjectsApi:
    """CustomObjectsApi fake that returns a fixed labelled child CR list."""

    def __init__(self, api: object, items: list[dict[str, object]]) -> None:
        del api
        self._items = items

    async def list_namespaced_custom_object(
        self,
        *,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        label_selector: str,
    ) -> dict[str, object]:
        assert group == "aiperf.nvidia.com"
        assert version == "v1alpha1"
        assert namespace == "bench-prod"
        assert plural == "aiperfjobs"
        assert label_selector == "aiperf.nvidia.com/sweep=llama-3-grid"
        return {"items": self._items}


def _app_for_sweeps_dir(results_dir: Path) -> FastAPI:
    """Build only the sweeps router so tests avoid operator lifespan side effects."""
    app = FastAPI()
    app.include_router(create_sweeps_router([cast(object, object())], results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like sweeps results directory."""
    transport = httpx.ASGITransport(app=_app_for_sweeps_dir(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.local"
    ) as c:
        yield c


def _record(
    *,
    namespace: str = "bench-prod",
    name: str = "llama-3-grid",
    source: str = "live",
    phase: str = "Running",
    total_variations: int = 3,
    completed_runs: int = 1,
    failed_runs: int = 0,
    cancelled_runs: int = 0,
    raw_status: dict[str, object] | None = None,
    raw_spec: dict[str, object] | None = None,
    aggregate_doc: dict[str, object] | None = None,
    current_child_ref: dict[str, object] | None = None,
    run_states: dict[str, int] | None = None,
) -> SweepRecord:
    """Return one sweep-union record with realistic dashboard identifiers."""
    return SweepRecord(
        namespace=namespace,
        name=name,
        source=source,
        phase=phase,
        total_variations=total_variations,
        completed_runs=completed_runs,
        failed_runs=failed_runs,
        cancelled_runs=cancelled_runs,
        age_seconds=42,
        model="meta-llama/Llama-3.1-8B-Instruct",
        aggregate_path=None,
        raw_status=raw_status or {"phase": phase},
        raw_spec=raw_spec or {},
        aggregate_doc=aggregate_doc,
        current_child_ref=current_child_ref,
        started_at="2026-05-18T12:00:00Z",
        completed_at=None,
        api_url="http://aiperf-operator.aiperf-system:8081",
        results_available=False,
        run_states=run_states or {"pending": 1, "running": 1, "completed": 1},
    )


def _aggregate_doc(**overrides: object) -> dict[str, object]:
    """Return an aggregate.json-shaped document for archived sweep routes."""
    doc: dict[str, object] = {
        "phase": "Succeeded",
        "totalVariations": 2,
        "completedRuns": 2,
        "failedRuns": 0,
        "maxTotalRuns": 2,
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "startedAt": "2026-05-18T12:00:00Z",
        "completedAt": "2026-05-18T12:05:00Z",
        "runStates": {"completed": 2, "failed": 0, "cancelled": 0},
        "specSummary": {
            "sweep_type": "grid",
            "dimensions": [
                {"name": "concurrency", "values": [32, 64]},
            ],
            "multi_run": {"trials": 1},
            "convergence": None,
        },
        "per_cell_aggregates": [
            {
                "variation_index": 1,
                "variation_label": "concurrency=64",
                "values": {"concurrency": 64},
                "trials_completed": 1,
                "trials_failed": 0,
                "metrics": {"request_throughput": {"avg": 912.5}},
                "children": [
                    {
                        "namespace": "bench-prod",
                        "name": "llama-3-grid-v01",
                        "trial_index": 0,
                        "phase": "Succeeded",
                    }
                ],
            },
            {
                "variation_index": 0,
                "variation_label": "concurrency=32",
                "values": {"concurrency": 32},
                "trials_completed": 1,
                "trials_failed": 0,
                "metrics": {"request_throughput": {"avg": 512.25}},
                "children": [],
            },
        ],
    }
    doc.update(overrides)
    return doc


def _seed_archived_sweep(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    name: str = "llama-3-grid",
    epoch: str = _EPOCH,
    aggregate: dict[str, object] | None = None,
) -> Path:
    """Create one archived sweep epoch and point latest.txt at it."""
    target = base_dir / namespace / "sweeps" / name / epoch
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("aggregate.json").write_bytes(
        orjson.dumps(aggregate or _aggregate_doc())
    )
    write_sweep_latest(base_dir, namespace, name, epoch)
    return target


async def _no_jobs(api: object, base_dir: Path, **kwargs: object) -> list[object]:
    del api, base_dir, kwargs
    return []


async def _no_pods(api: object, namespace: str, name: str, source: str) -> list[object]:
    del api, namespace, name, source
    return []


# ============================================================
# Config exposure boundary
# ============================================================


class TestSweepsRouterConfigExposure:
    """The public sweep config endpoint must not expose stored credentials."""

    @pytest.mark.asyncio
    async def test_get_sweep_config_redacts_endpoint_secrets_without_mutating_raw_spec(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        raw_spec: dict[str, object] = {
            "benchmark": {
                "endpoint": {
                    "urls": [
                        "https://user:password@api.example.invalid/v1"
                        "?api_key=query-secret&model=m"
                    ],
                    "apiKey": "sk-live-should-not-reach-browser",
                    "headers": {
                        "Authorization": "Bearer operator-service-account-token",
                        "X-AIPerf-Trace": "conv-2026-04-21-9c3a",
                    },
                }
            }
        }
        original_raw_spec = deepcopy(raw_spec)

        async def fake_find_any_sweep(
            api: object,
            base_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> SweepRecord:
            del api, base_dir, namespace, name, epoch
            return _record(raw_spec=raw_spec)

        monkeypatch.setattr(mod, "find_any_sweep", fake_find_any_sweep)

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid/config")

        assert response.status_code == 200
        endpoint = response.json()["spec"]["benchmark"]["endpoint"]
        assert endpoint["apiKey"] == "<redacted>"
        assert endpoint["urls"] == [
            "https://<redacted>@api.example.invalid/v1?api_key=<redacted>&model=m"
        ]
        assert endpoint["headers"] == {
            "Authorization": "<redacted>",
            "X-AIPerf-Trace": "conv-2026-04-21-9c3a",
        }
        assert "sk-live-should-not-reach-browser" not in response.text
        assert "operator-service-account-token" not in response.text
        assert "user:password" not in response.text
        assert "query-secret" not in response.text
        assert raw_spec == original_raw_spec


# ============================================================
# Path encoding / route trust boundary
# ============================================================


class TestSweepsRouterPathEncoding:
    """Namespace/name path parameters are decoded once and never treated as files."""

    @pytest.mark.asyncio
    async def test_get_sweep_url_encoded_name_resolves_matching_archive(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_find_aiperfsweep(
            api: object, namespace: str, name: str
        ) -> dict[str, object] | None:
            del api
            assert namespace == "bench-prod"
            assert name == "llama-3.1-grid"
            return None

        monkeypatch.setattr(union_mod, "find_aiperfsweep", fake_find_aiperfsweep)
        monkeypatch.setattr(mod, "list_all_jobs", _no_jobs)
        monkeypatch.setattr(mod, "fetch_sweep_pod_summaries", _no_pods)
        # Valid Kubernetes name; the encoded dot still exercises percent-decoding.
        _seed_archived_sweep(tmp_path, name="llama-3.1-grid")

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3%2E1-grid")

        assert response.status_code == 200
        body = response.json()
        assert body["sweep"]["namespace"] == "bench-prod"
        assert body["sweep"]["name"] == "llama-3.1-grid"
        assert body["sweep"]["source"] == "archived"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path",
        [
            param("/api/v1/sweeps/bench%2Fprod/llama-3-grid", id="encoded-slash-namespace"),
            param("/api/v1/sweeps/bench-prod/llama%2F3-grid", id="encoded-slash-name"),
        ],
    )  # fmt: skip
    async def test_get_sweep_encoded_slash_does_not_match_two_segment_route(
        self, client: httpx.AsyncClient, path: str
    ) -> None:
        response = await client.get(path)

        assert response.status_code == 404


# ============================================================
# Missing sweeps and Kubernetes API failures
# ============================================================


class TestSweepsRouterMissingAndApiErrors:
    """404 means absent sweep; apiserver failures must not be mislabeled as missing."""

    @pytest.mark.asyncio
    async def test_get_sweep_missing_live_and_archive_returns_404_with_sweep_key(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_find_aiperfsweep(
            api: object, namespace: str, name: str
        ) -> dict[str, object] | None:
            del api, namespace, name
            return None

        monkeypatch.setattr(union_mod, "find_aiperfsweep", fake_find_aiperfsweep)

        response = await client.get("/api/v1/sweeps/bench-prod/missing-llama-sweep")

        assert response.status_code == 404
        assert (
            response.json()["detail"]
            == "Sweep bench-prod/missing-llama-sweep not found"
        )

    @pytest.mark.asyncio
    async def test_get_sweep_kubernetes_api_500_returns_server_error_not_404(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_find_aiperfsweep(
            api: object, namespace: str, name: str
        ) -> dict[str, object] | None:
            del api, namespace, name
            raise ApiException(status=500, reason="apiserver overloaded")

        monkeypatch.setattr(union_mod, "find_aiperfsweep", fake_find_aiperfsweep)
        transport = httpx.ASGITransport(
            app=_app_for_sweeps_dir(tmp_path), raise_app_exceptions=False
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf.local"
        ) as c:
            response = await c.get("/api/v1/sweeps/bench-prod/llama-3-grid")

        assert response.status_code == 500
        assert "not found" not in response.text.lower()


# ============================================================
# Malformed live status payloads
# ============================================================


class TestSweepsRouterMalformedLiveStatus:
    """Malformed CR status fragments should not take down the dashboard list."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_states",
        [
            param(["completed", 1], id="list-instead-of-mapping"),
            param({"completed": "two"}, id="non-numeric-count"),
        ],
    )  # fmt: skip
    async def test_list_sweeps_malformed_run_states_returns_empty_rollup(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        run_states: object,
    ) -> None:
        async def fake_list_aiperfsweeps(
            api: object, *, namespace: str | None = None, all_namespaces: bool = True
        ) -> list[dict[str, object]]:
            del api, namespace
            assert all_namespaces is True
            return [
                {
                    "metadata": {
                        "namespace": "bench-prod",
                        "name": "llama-3-grid",
                        "creationTimestamp": "2026-05-18T12:00:00Z",
                    },
                    "spec": {
                        "template": {
                            "spec": {"models": ["meta-llama/Llama-3.1-8B-Instruct"]}
                        }
                    },
                    "status": {
                        "phase": "Running",
                        "totalVariations": 3,
                        "completedRuns": 1,
                        "failedRuns": 0,
                        "runStates": run_states,
                    },
                }
            ]

        monkeypatch.setattr(union_mod, "list_aiperfsweeps", fake_list_aiperfsweeps)

        response = await client.get("/api/v1/sweeps")

        assert response.status_code == 200
        assert response.json()["sweeps"][0]["run_states"] == {}

    @pytest.mark.asyncio
    async def test_get_sweep_scalar_current_child_ref_is_sanitized_to_null(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_find_any_sweep(
            api: object,
            base_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> SweepRecord:
            del api, base_dir, namespace, name, epoch
            return _record(
                raw_status={
                    "phase": "Running",
                    "currentChildRef": "llama-3-grid-v00",
                },
                current_child_ref=cast(dict[str, object], "llama-3-grid-v00"),
            )

        monkeypatch.setattr(mod, "find_any_sweep", fake_find_any_sweep)
        monkeypatch.setattr(mod, "list_all_jobs", _no_jobs)
        monkeypatch.setattr(mod, "fetch_sweep_pod_summaries", _no_pods)

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid")

        assert response.status_code == 200
        assert response.json()["sweep"]["current_child_ref"] is None

    def test_summary_redacts_compound_credential_in_current_child_label(self) -> None:
        summary = mod._summary(
            _record(
                current_child_ref={
                    "name": "llama-3-grid-v00",
                    "index": 0,
                    "label": "variables.my_auth_token=opaque-secret",
                }
            )
        )

        assert summary.current_child_ref == {
            "name": "llama-3-grid-v00",
            "index": 0,
            "label": "variables.my_auth_token=<redacted>",
        }


# ============================================================
# Response schema stability / list filters
# ============================================================


class TestSweepsRouterResponseSchema:
    """Dashboard clients rely on stable keys even as live/archive sources vary."""

    @pytest.mark.asyncio
    async def test_get_sweep_runs_truncated_fetch_url_shape_is_preserved(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_find_any_sweep(
            api: object,
            base_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> SweepRecord:
            del api, base_dir, namespace, name, epoch
            return _record(
                raw_status={
                    "phase": "Succeeded",
                    "runs": [],
                    "runsTruncated": {
                        "truncated": True,
                        "fetchURL": (
                            "http://aiperf-operator.aiperf-system:8081/api/v1/"
                            "sweeps/bench-prod/llama-3-grid/children"
                        ),
                    },
                }
            )

        monkeypatch.setattr(mod, "find_any_sweep", fake_find_any_sweep)
        monkeypatch.setattr(mod, "list_all_jobs", _no_jobs)
        monkeypatch.setattr(mod, "fetch_sweep_pod_summaries", _no_pods)

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid")

        assert response.status_code == 200
        truncated = response.json()["status"]["runsTruncated"]
        assert truncated == {
            "truncated": True,
            "fetchURL": (
                "http://aiperf-operator.aiperf-system:8081/api/v1/"
                "sweeps/bench-prod/llama-3-grid/children"
            ),
        }
        assert truncated["fetchURL"].endswith(
            "/api/v1/sweeps/bench-prod/llama-3-grid/children"
        )

    @pytest.mark.asyncio
    async def test_list_sweeps_query_filters_keep_server_snapshot_schema_stable(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_list_all_sweeps(
            api: object,
            base_dir: Path,
            *,
            all_namespaces: bool = True,
        ) -> list[SweepRecord]:
            del api, base_dir
            assert all_namespaces is True
            return [
                _record(namespace="bench-prod", name="llama-3-grid", phase="Running"),
                _record(namespace="bench-dev", name="mixtral-grid", phase="Succeeded"),
            ]

        monkeypatch.setattr(mod, "list_all_sweeps", fake_list_all_sweeps)

        response = await client.get("/api/v1/sweeps?ns=bench-prod&phase=running")

        assert response.status_code == 200
        sweeps = response.json()["sweeps"]
        assert {s["namespace"] for s in sweeps} == {"bench-prod", "bench-dev"}
        assert set(sweeps[0]) == {
            "namespace",
            "name",
            "source",
            "phase",
            "total_variations",
            "completed_runs",
            "failed_runs",
            "cancelled_runs",
            "age_seconds",
            "model",
            "started_at",
            "completed_at",
            "api_url",
            "results_available",
            "current_child_ref",
            "run_states",
        }

    @pytest.mark.asyncio
    async def test_cells_from_archived_aggregate_are_sorted_and_schema_stable(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_find_aiperfsweep(
            api: object, namespace: str, name: str
        ) -> dict[str, object] | None:
            del api, namespace, name
            return None

        monkeypatch.setattr(union_mod, "find_aiperfsweep", fake_find_aiperfsweep)
        _seed_archived_sweep(tmp_path)

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid/cells")

        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "archived"
        assert [cell["variation_index"] for cell in body["cells"]] == [0, 1]
        assert set(body["cells"][0]) == {
            "variation_index",
            "variation_label",
            "values",
            "trials_completed",
            "trials_failed",
            "metrics",
            "children",
        }


# ============================================================
# Child metadata / label propagation
# ============================================================


class TestSweepsRouterChildrenManifest:
    """Live child AIPerfJob labels become the manifest consumed by SweepDetail."""

    @pytest.mark.asyncio
    async def test_children_live_manifest_propagates_variation_labels_and_run_epoch(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        children = [
            {
                "metadata": {
                    "name": "llama-3-grid-v01-t2",
                    "labels": {
                        "aiperf.nvidia.com/sweep": "llama-3-grid",
                        "aiperf.nvidia.com/sweep-run-epoch": "1714150923",
                        "aiperf.nvidia.com/variation-index": "1",
                        "aiperf.nvidia.com/variation-label": "concurrency=64",
                    },
                },
                "status": {"runEpoch": "1714150999"},
            }
        ]

        async def fake_find_any_sweep(
            api: object,
            base_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> SweepRecord:
            del api, base_dir, namespace, name, epoch
            return _record(raw_status={"phase": "Running"})

        monkeypatch.setattr(mod, "find_any_sweep", fake_find_any_sweep)
        monkeypatch.setattr(
            live_mod.k8s,
            "CustomObjectsApi",
            lambda api: _FakeCustomObjectsApi(api, children),
        )

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid/children")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "sweepRunEpoch": "1714150923",
            "children": [
                {
                    "namespace": "bench-prod",
                    "name": "llama-3-grid-v01-t2",
                    "variationIndex": 1,
                    "variationLabel": "concurrency=64",
                    "variationValues": "",
                    "trialIndex": 2,
                    "childRunEpoch": "1714150999",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_children_live_manifest_filters_mismatched_sweep_label(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        children = [
            {
                "metadata": {
                    "name": "llama-3-grid-v00-t0",
                    "labels": {
                        "aiperf.nvidia.com/sweep": "llama-3-grid",
                        "aiperf.nvidia.com/sweep-run-epoch": "1714150923",
                        "aiperf.nvidia.com/variation-index": "0",
                        "aiperf.nvidia.com/variation-label": "concurrency=32",
                    },
                },
                "status": {"runEpoch": "1714150930"},
            },
            {
                "metadata": {
                    "name": "unrelated-grid-v09-t0",
                    "labels": {
                        "aiperf.nvidia.com/sweep": "other-grid",
                        "aiperf.nvidia.com/variation-index": "9",
                        "aiperf.nvidia.com/variation-label": "concurrency=999",
                    },
                },
                "status": {"runEpoch": "1714151000"},
            },
        ]

        async def fake_find_any_sweep(
            api: object,
            base_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> SweepRecord:
            del api, base_dir, namespace, name, epoch
            return _record(raw_status={"phase": "Running"})

        monkeypatch.setattr(mod, "find_any_sweep", fake_find_any_sweep)
        monkeypatch.setattr(
            live_mod.k8s,
            "CustomObjectsApi",
            lambda api: _FakeCustomObjectsApi(api, children),
        )

        response = await client.get("/api/v1/sweeps/bench-prod/llama-3-grid/children")

        assert response.status_code == 200
        names = [child["name"] for child in response.json()["children"]]
        assert names == ["llama-3-grid-v00-t0"]
