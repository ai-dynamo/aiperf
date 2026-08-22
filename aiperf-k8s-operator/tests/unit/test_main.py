# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import aiperf_k8s_operator.main as operator_main
import aiperf_k8s_operator.reconciliation as reconciliation
import pytest
from aiperf_k8s_operator.contract import validate_envelope
from aiperf_k8s_operator.settings import OperatorSettings
from aiperf_k8s_operator.upload_auth import (
    derive_upload_public_key,
    results_read_token_sha256,
)
from kubernetes_asyncio.client.exceptions import ApiException

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
async def test_kubernetes_upload_verifier_requires_the_exact_cr_identity() -> None:
    payload = json.loads((FIXTURES / "valid-one-cell-envelope.json").read_text())
    bootstrap_by_name: dict[str, bytes] = {}
    for role in payload["roles"]:
        reference = role.get("bootstrap")
        if reference is not None:
            bootstrap = f"{role['name']}-private".encode()
            reference["sha256"] = hashlib.sha256(bootstrap).hexdigest()
            bootstrap_by_name[reference["secretName"]] = bootstrap
    for reference in payload["cellBootstraps"]:
        bootstrap = b"cell-private"
        reference["sha256"] = hashlib.sha256(bootstrap).hexdigest()
        bootstrap_by_name[reference["secretName"]] = bootstrap
    envelope = validate_envelope(payload)
    object_uid = "4f78fcbe-9aae-4cc9-ae19-204231b21575"
    raw_read_token = bytes(range(32))
    read_secret = reconciliation.build_results_read_secret(
        envelope, object_uid, raw_read_token
    )
    sidecar = next(role for role in envelope.roles if role.name == "results-sidecar")
    upload_public_key = derive_upload_public_key(
        bootstrap_by_name[sidecar.bootstrap.secret_name],
        envelope.namespace,
        envelope.job_id,
        envelope.run_id,
        object_uid,
    )
    authority = reconciliation.build_results_authority(
        envelope,
        object_uid,
        upload_public_key,
        results_read_token_sha256(raw_read_token),
    )

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

    class FakeCore:
        config_map_error: ApiException | None = None

        async def read_namespaced_secret(self, name: str, namespace: str):
            if name == read_secret["metadata"]["name"]:
                return read_secret
            bootstrap = bootstrap_by_name[name]
            reference = next(
                reference
                for reference in [
                    *(role.get("bootstrap") for role in payload["roles"]),
                    *payload["cellBootstraps"],
                ]
                if reference is not None and reference["secretName"] == name
            )
            return SimpleNamespace(
                immutable=True,
                metadata=SimpleNamespace(
                    name=name,
                    namespace=namespace,
                    labels={
                        "aiperf.nvidia.com/run-id": envelope.run_id,
                        "aiperf.nvidia.com/role": reference.get("role", "cell"),
                    },
                    annotations={"aiperf.nvidia.com/sha256": reference["sha256"]},
                    owner_references=[
                        SimpleNamespace(
                            api_version="aiperf.nvidia.com/v1alpha1",
                            kind="AIPerfJob",
                            name=envelope.job_id,
                            uid=object_uid,
                            controller=True,
                            block_owner_deletion=None,
                        )
                    ],
                ),
                data={"bootstrap": base64.b64encode(bootstrap).decode()},
            )

        async def read_namespaced_config_map(self, **_):
            if self.config_map_error is not None:
                raise self.config_map_error
            return authority

    objects = FakeObjects()
    core = FakeCore()
    provider = operator_main.KubernetesUploadVerifiers(objects, core, object())
    resolved = await provider.authorities("bench", "job-1", "run-1")
    assert resolved == operator_main.RunAuthorities(
        object_uid, upload_public_key, results_read_token_sha256(raw_read_token)
    )
    assert await provider.authorities("bench", "job-1", "other-run") is None
    authority["data"]["unexpected"] = "must-not-be-accepted"
    assert await provider.authorities("bench", "job-1", "run-1") is None
    del authority["data"]["unexpected"]
    read_secret["data"]["unexpected"] = base64.b64encode(
        b"authority-confusion"
    ).decode()
    assert await provider.authorities("bench", "job-1", "run-1") is None
    del read_secret["data"]["unexpected"]
    await provider.mark_results_ready("bench", "job-1", "run-1", object_uid)
    assert objects.status_patch["body"] == {
        "metadata": {"uid": object_uid},
        "status": {"phase": "Completed", "resultsReady": True, "runId": "run-1"},
    }
    core.config_map_error = ApiException(status=500, reason="apiserver unavailable")
    with pytest.raises(ApiException, match="apiserver unavailable"):
        await provider.authorities("bench", "job-1", "run-1")


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

        async def read_namespaced_config_map(self, name, namespace):
            raise ApiException(status=404, reason="NotFound")

    class FakeRbac:
        async def delete_namespaced_role(self, name, namespace, **_):
            deleted.append(("Role", name))

        async def delete_namespaced_role_binding(self, name, namespace, **_):
            deleted.append(("RoleBinding", name))

    class FakeObjects:
        async def delete_namespaced_custom_object(self, **kwargs):
            deleted.append(("JobSet", kwargs["name"]))

    api_client = FakeApiClient()
    released = []

    class FakeIndex:
        def release_identity(self, identity) -> bool:
            released.append(identity)
            return True

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
        ("Secret", "aiperf-results-read-76698a922acf3f34"),
        ("ConfigMap", "aiperf-results-authority-76698a922acf3f34"),
        ("ConfigMap", "aiperf-config-76698a922acf3f34"),
        ("Secret", "bootstrap-controller"),
        ("Secret", "bootstrap-sidecar"),
        ("Secret", "bootstrap-cell-0"),
    ]
    assert released == [operator_main.ResultIdentity("bench", "job-1", "run-1", uid)]
    assert api_client.was_closed
