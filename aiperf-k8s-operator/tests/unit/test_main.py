# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import aiperf_k8s_operator.main as operator_main
import pytest
from aiperf_k8s_operator.settings import OperatorSettings

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "native-k8s" / "v1" / "fixtures"


@pytest.mark.asyncio
async def test_operator_and_results_api_share_one_supervised_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    api_started = asyncio.Event()
    api_stopped = asyncio.Event()
    observed: dict[str, object] = {}

    class FakeObjects:
        pass

    class FakeCore:
        pass

    class FakeApiClient:
        was_closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            self.was_closed = True

    class FakeServer:
        should_exit = False

        def __init__(self, config) -> None:
            observed["config"] = config

        async def serve(self) -> bool:
            api_started.set()
            while not self.should_exit:
                await asyncio.sleep(0)
            api_stopped.set()
            return True

    async def fake_operator(
        *, stop_flag, settings, clusterwide, standalone, memo
    ) -> None:
        await api_started.wait()
        observed["operator_started"] = True
        observed["posting_enabled"] = settings.posting.enabled
        observed["clusterwide"] = clusterwide
        observed["standalone"] = standalone
        observed["memo"] = memo
        stop_flag.set()

    api_client = FakeApiClient()
    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: api_client)
    monkeypatch.setattr(
        operator_main.client, "CustomObjectsApi", lambda _: FakeObjects()
    )
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: FakeCore())
    monkeypatch.setattr(
        operator_main.kubernetes_config, "load_incluster_config", lambda: None
    )
    monkeypatch.setattr(operator_main.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(operator_main.kopf, "operator", fake_operator)

    await operator_main.run_services(
        OperatorSettings(
            artifact_root=str(tmp_path), api_host="127.0.0.9", api_port=18080
        )
    )

    config = observed["config"]
    assert config.host == "127.0.0.9"
    assert config.port == 18080
    assert observed["operator_started"] is True
    assert observed["posting_enabled"] is False
    assert observed["clusterwide"] is True
    assert observed["standalone"] is True
    assert isinstance(observed["memo"], operator_main.ResultsIndex)
    assert api_stopped.is_set()
    assert api_client.was_closed
    paths = {route.path for route in config.app.routes if hasattr(route, "path")}
    assert "/healthz" in paths
    assert "/api/uploads/{namespace}/{job_id}/{run_id}/manifest" in paths


@pytest.mark.asyncio
async def test_create_sweep_validates_envelope() -> None:
    """Submitting an envelope with mismatched runId/namespace raises ValueError."""
    payload = json.loads((FIXTURES / "valid-sweep-envelope.json").read_text())
    # Fixture has runId="sweep-run-1", namespace="bench"; pass a wrong name to trigger error
    with pytest.raises(ValueError, match="identity"):
        await operator_main.create_sweep(
            spec={"sweepEnvelope": payload},
            name="wrong-name",
            namespace="bench",
            uid="test-uid",
        )


@pytest.mark.asyncio
async def test_create_sweep_provisions_rbac(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RBAC rule list for the sweep contains aiperfjobs and aiperfsweeps/status rules."""
    payload = json.loads((FIXTURES / "valid-sweep-envelope.json").read_text())
    roles_created: list[dict] = []

    class FakeApiClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        def sanitize_for_serialization(self, obj):
            return obj

    class FakeCore:
        async def create_namespaced_service_account(self, **_):
            pass

        async def read_namespaced_service_account(self, **_):
            return {}

        async def create_namespaced_config_map(self, **_):
            pass

        async def read_namespaced_config_map(self, **_):
            return {}

    class FakeRbac:
        async def create_namespaced_role(self, **kwargs):
            roles_created.append(kwargs["body"])

        async def read_namespaced_role(self, **_):
            return {}

        async def create_namespaced_role_binding(self, **_):
            pass

        async def read_namespaced_role_binding(self, **_):
            return {}

    class FakeObjects:
        async def get_namespaced_custom_object(self, **_):
            return {
                "metadata": {
                    "name": "sweep-run-1",
                    "namespace": "bench",
                    "uid": "test-uid",
                }
            }

        async def patch_namespaced_custom_object_status(self, **_):
            pass

        async def create_namespaced_custom_object(self, **_):
            pass

    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: FakeApiClient())
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: FakeCore())
    monkeypatch.setattr(
        operator_main.client, "RbacAuthorizationV1Api", lambda _: FakeRbac()
    )
    monkeypatch.setattr(
        operator_main.client, "CustomObjectsApi", lambda _: FakeObjects()
    )

    await operator_main.create_sweep(
        spec={"sweepEnvelope": payload},
        name="sweep-run-1",
        namespace="bench",
        uid="test-uid",
    )

    assert len(roles_created) == 1
    rules = roles_created[0]["rules"]
    aiperfjobs_rule = next(
        (r for r in rules if "aiperfjobs" in r.get("resources", [])), None
    )
    assert aiperfjobs_rule is not None, "Role must include an aiperfjobs rule"
    assert set(aiperfjobs_rule["verbs"]) >= {"create", "get", "list", "watch", "delete"}
    sweep_status_rule = next(
        (r for r in rules if "aiperfsweeps/status" in r.get("resources", [])), None
    )
    assert sweep_status_rule is not None, (
        "Role must include an aiperfsweeps/status rule"
    )
    assert "patch" in sweep_status_rule["verbs"]
    assert sweep_status_rule.get("resourceNames") == ["sweep-run-1"]


@pytest.mark.asyncio
async def test_create_sweep_provisions_jobset_with_sweep_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JobSet has exactly one replicatedJob, one replica, sweep-controller, automountServiceAccountToken True."""
    payload = json.loads((FIXTURES / "valid-sweep-envelope.json").read_text())
    jobsets_created: list[dict] = []

    class FakeApiClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        def sanitize_for_serialization(self, obj):
            return obj

    class FakeCore:
        async def create_namespaced_service_account(self, **_):
            pass

        async def read_namespaced_service_account(self, **_):
            return {}

        async def create_namespaced_config_map(self, **_):
            pass

        async def read_namespaced_config_map(self, **_):
            return {}

    class FakeRbac:
        async def create_namespaced_role(self, **_):
            pass

        async def read_namespaced_role(self, **_):
            return {}

        async def create_namespaced_role_binding(self, **_):
            pass

        async def read_namespaced_role_binding(self, **_):
            return {}

    class FakeObjects:
        async def get_namespaced_custom_object(self, **_):
            return {
                "metadata": {
                    "name": "sweep-run-1",
                    "namespace": "bench",
                    "uid": "test-uid",
                }
            }

        async def patch_namespaced_custom_object_status(self, **_):
            pass

        async def create_namespaced_custom_object(
            self, group, version, namespace, plural, body, **_
        ):
            if plural == "jobsets":
                jobsets_created.append(body)

    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: FakeApiClient())
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: FakeCore())
    monkeypatch.setattr(
        operator_main.client, "RbacAuthorizationV1Api", lambda _: FakeRbac()
    )
    monkeypatch.setattr(
        operator_main.client, "CustomObjectsApi", lambda _: FakeObjects()
    )

    await operator_main.create_sweep(
        spec={"sweepEnvelope": payload},
        name="sweep-run-1",
        namespace="bench",
        uid="test-uid",
    )

    assert len(jobsets_created) == 1, "Exactly one JobSet must be created"
    jobset = jobsets_created[0]
    replicated_jobs = jobset["spec"]["replicatedJobs"]
    assert len(replicated_jobs) == 1, "JobSet must have exactly one replicatedJob"
    job = replicated_jobs[0]
    assert job["replicas"] == 1
    pod_spec = job["template"]["spec"]["template"]["spec"]
    # sweep-controller needs True so it can call the kube API to create child AIPerfJobs
    assert pod_spec.get("automountServiceAccountToken") is True
    containers = pod_spec["containers"]
    assert len(containers) == 1
    container = containers[0]
    assert container["name"] == "sweep-controller"
    assert container["image"] == payload["imageReference"]
    assert container["command"] == payload["sweepController"]["command"]
    assert container["args"] == payload["sweepController"]["argv"]


@pytest.mark.asyncio
async def test_sweep_status_roll_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Child AIPerfJob events owned by a sweep update status.childRuns, completedRuns, failedRuns."""
    sweep_name = "sweep-run-1"
    sweep_uid = "test-sweep-uid"
    job_name = "child-job-1"
    namespace = "bench"

    status_patches: list[dict] = []

    class FakeApiClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

    class FakeObjects:
        async def get_namespaced_custom_object(self, **_):
            return {
                "metadata": {"uid": sweep_uid},
                "status": {"childRuns": [], "completedRuns": 0, "failedRuns": 0},
            }

        async def patch_namespaced_custom_object_status(self, **kwargs):
            status_patches.append(kwargs)

    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: FakeApiClient())
    monkeypatch.setattr(
        operator_main.client, "CustomObjectsApi", lambda _: FakeObjects()
    )

    event = {
        "type": "MODIFIED",
        "object": {
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "ownerReferences": [
                    {
                        "apiVersion": "aiperf.nvidia.com/v1alpha1",
                        "kind": "AIPerfSweep",
                        "name": sweep_name,
                        "uid": sweep_uid,
                        "controller": True,
                    }
                ],
            },
            "status": {"phase": "Completed"},
        },
    }

    await operator_main.observe_child_run(event=event)

    assert len(status_patches) == 1, "Exactly one status patch must be issued"
    patch_body = status_patches[0]["body"]
    child_runs = patch_body["status"]["childRuns"]
    assert len(child_runs) == 1
    assert child_runs[0]["name"] == job_name
    assert child_runs[0]["phase"] == "Completed"
    assert patch_body["status"]["completedRuns"] == 1
    assert patch_body["status"]["failedRuns"] == 0


@pytest.mark.asyncio
async def test_kubernetes_results_lifecycle_requires_the_exact_cr_identity() -> None:
    payload = json.loads((FIXTURES / "valid-one-cell-envelope.json").read_text())
    object_uid = "4f78fcbe-9aae-4cc9-ae19-204231b21575"

    class FakeObjects:
        status_patch = None

        async def get_namespaced_custom_object(self, **_):
            return {
                "metadata": {
                    "name": "job-1",
                    "namespace": "bench",
                    "uid": object_uid,
                },
                "spec": {"envelope": payload},
                "status": {
                    "phase": "PublishingResults",
                    "runId": "run-1",
                    "jobSet": "job-1",
                },
            }

        async def patch_namespaced_custom_object_status(self, **kwargs):
            self.status_patch = kwargs

    objects = FakeObjects()
    lifecycle = operator_main.KubernetesResultsLifecycle(objects)
    with pytest.raises(ValueError, match="identity"):
        await lifecycle.mark_results_ready("bench", "job-1", "other-run")
    await lifecycle.mark_results_ready("bench", "job-1", "run-1")
    assert objects.status_patch["body"] == {
        "metadata": {"uid": object_uid},
        "status": {"phase": "Completed", "resultsReady": True, "runId": "run-1"},
    }


@pytest.mark.asyncio
async def test_delete_handler_removes_every_per_incarnation_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads((FIXTURES / "valid-one-cell-envelope.json").read_text())
    deleted: list[tuple[str, str]] = []
    uid = "4f78fcbe-9aae-4cc9-ae19-204231b21575"

    class FakeApiClient:
        was_closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            self.was_closed = True

    class FakeCore:
        async def delete_namespaced_service_account(self, name, namespace, **_):
            deleted.append(("ServiceAccount", name))

        async def delete_namespaced_secret(self, name, namespace, **_):
            deleted.append(("Secret", name))

        async def delete_namespaced_config_map(self, name, namespace, **_):
            deleted.append(("ConfigMap", name))

    class FakeRbac:
        async def delete_namespaced_role(self, name, namespace, **_):
            deleted.append(("Role", name))

        async def delete_namespaced_role_binding(self, name, namespace, **_):
            deleted.append(("RoleBinding", name))

    class FakeObjects:
        async def delete_namespaced_custom_object(self, **kwargs):
            deleted.append(("JobSet", kwargs["name"]))

    api_client = FakeApiClient()

    class FakeIndex:
        def release_identity(self, identity) -> bool:
            raise AssertionError(f"published results must not be released: {identity}")

    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: api_client)
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: FakeCore())
    monkeypatch.setattr(
        operator_main.client, "RbacAuthorizationV1Api", lambda _: FakeRbac()
    )
    monkeypatch.setattr(
        operator_main.client, "CustomObjectsApi", lambda _: FakeObjects()
    )

    await operator_main.delete_job(
        {"envelope": payload},
        name="job-1",
        namespace="bench",
        uid=uid,
        memo=FakeIndex(),
    )

    assert deleted == [
        ("JobSet", "job-1"),
        ("ServiceAccount", "aiperf-workload-6570bdff37b73e9e"),
        ("Role", "aiperf-workload-6570bdff37b73e9e"),
        ("RoleBinding", "aiperf-workload-6570bdff37b73e9e"),
        ("ConfigMap", "aiperf-config-76698a922acf3f34"),
        ("Secret", "bootstrap-controller"),
        ("Secret", "bootstrap-cell-0"),
    ]
    assert api_client.was_closed
