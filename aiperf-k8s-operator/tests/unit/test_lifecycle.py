# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle transition tests for the native Kubernetes operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiperf_k8s_operator.main as operator_main
import pytest
from aiperf_k8s_operator.contract import validate_envelope

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "contracts/native-k8s/v1/fixtures/valid-one-cell-envelope.json"
OBJECT_UID = "4f78fcbe-9aae-4cc9-ae19-204231b21575"


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


class _FakeApiClient:
    async def __aenter__(self) -> _FakeApiClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


@pytest.mark.asyncio
async def test_create_persists_pending_before_workload_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    envelope = validate_envelope(payload)
    events: list[str] = []
    status: dict[str, Any] | None = None

    class FakeObjects:
        async def patch_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            nonlocal status
            events.append(f"status:{kwargs['body']['status']['phase']}")
            status = kwargs["body"]["status"]

        async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
            return {
                "metadata": {
                    "name": envelope.job_id,
                    "namespace": envelope.namespace,
                    "uid": OBJECT_UID,
                },
                "spec": {"envelope": payload},
                **({"status": status} if status is not None else {}),
            }

    async def fake_references(*_: Any) -> operator_main.ReferenceMaterial:
        return operator_main.ReferenceMaterial({})

    async def fake_snapshot(*_: Any) -> None:
        events.append("config-snapshot")

    async def fake_identity(*_: Any) -> None:
        events.append("identity")

    async def fake_reconcile(*_: Any) -> dict[str, Any]:
        events.append("jobset")
        return {
            "phase": "Pending",
            "runId": envelope.run_id,
            "jobSet": envelope.job_id,
        }

    objects = FakeObjects()
    monkeypatch.setattr(operator_main.client, "ApiClient", _FakeApiClient)
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: object())
    monkeypatch.setattr(
        operator_main.client, "RbacAuthorizationV1Api", lambda _: object()
    )
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", lambda _: objects)
    monkeypatch.setattr(operator_main, "_reference_metadata", fake_references)
    monkeypatch.setattr(operator_main, "validate_references", lambda *_: None)
    monkeypatch.setattr(operator_main, "_ensure_config_snapshot", fake_snapshot)
    monkeypatch.setattr(operator_main, "ensure_workload_identity", fake_identity)
    monkeypatch.setattr(operator_main, "reconcile_job", fake_reconcile)

    deferred_patch: dict[str, Any] = {}
    await operator_main.create_job(
        {"envelope": payload},
        name=envelope.job_id,
        namespace=envelope.namespace,
        uid=OBJECT_UID,
        patch=deferred_patch,
    )

    assert events.index("status:Pending") < events.index("jobset")
    assert deferred_patch == {}


@pytest.mark.asyncio
async def test_result_publication_requires_publishing_results_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    phase = "Pending"

    class FakeObjects:
        status_patches: list[dict[str, Any]] = []

        async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
            return {
                "metadata": {
                    "name": "job-1",
                    "namespace": "bench",
                    "uid": OBJECT_UID,
                },
                "spec": {"envelope": payload},
                "status": {"phase": phase, "runId": "run-1", "jobSet": "job-1"},
            }

        async def patch_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            self.status_patches.append(kwargs["body"])

    objects = FakeObjects()
    provider = operator_main.KubernetesResultsLifecycle(objects)

    with pytest.raises(ValueError, match="PublishingResults"):
        await provider.mark_results_ready("bench", "job-1", "run-1")
    assert objects.status_patches == []

    phase = "PublishingResults"
    await provider.mark_results_ready("bench", "job-1", "run-1")
    assert objects.status_patches == [
        {
            "metadata": {"uid": OBJECT_UID},
            "status": {
                "phase": "Completed",
                "resultsReady": True,
                "runId": "run-1",
            },
        }
    ]


@pytest.mark.asyncio
async def test_failed_jobset_marks_only_its_current_nonterminal_owner_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    phase = "Pending"
    owner_uid = OBJECT_UID

    class FakeObjects:
        status_patches: list[dict[str, Any]] = []

        async def get_namespaced_custom_object(self, **_: Any) -> dict[str, Any]:
            return {
                "metadata": {
                    "name": "job-1",
                    "namespace": "bench",
                    "uid": OBJECT_UID,
                },
                "spec": {"envelope": payload},
                "status": {"phase": phase, "runId": "run-1", "jobSet": "job-1"},
            }

        async def patch_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            self.status_patches.append(kwargs["body"])

    objects = FakeObjects()
    monkeypatch.setattr(operator_main.client, "ApiClient", _FakeApiClient)
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", lambda _: objects)

    def failed_jobset() -> dict[str, Any]:
        return {
            "metadata": {
                "name": "job-1",
                "namespace": "bench",
                "labels": {
                    "aiperf.nvidia.com/run-id": "run-1",
                    "aiperf.nvidia.com/role": "jobset",
                },
                "annotations": {
                    "aiperf.nvidia.com/sha256": operator_main.envelope_sha256(
                        validate_envelope(payload)
                    )
                },
                "ownerReferences": [
                    {
                        "apiVersion": "aiperf.nvidia.com/v1alpha1",
                        "kind": "AIPerfJob",
                        "name": "job-1",
                        "uid": owner_uid,
                        "controller": True,
                    }
                ],
            },
            "status": {"terminalState": "Failed"},
        }

    await operator_main.observe_jobset(
        event={"type": "MODIFIED", "object": failed_jobset()}
    )
    assert objects.status_patches == [
        {
            "metadata": {"uid": OBJECT_UID},
            "status": {
                "phase": "Failed",
                "runId": "run-1",
                "jobSet": "job-1",
            },
        }
    ]

    objects.status_patches.clear()
    phase = "Completed"
    await operator_main.observe_jobset(
        event={"type": "MODIFIED", "object": failed_jobset()}
    )
    owner_uid = "stale-incarnation"
    phase = "Pending"
    await operator_main.observe_jobset(
        event={"type": "MODIFIED", "object": failed_jobset()}
    )
    assert objects.status_patches == []
