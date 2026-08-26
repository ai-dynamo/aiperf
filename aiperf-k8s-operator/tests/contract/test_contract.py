# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import aiperf_k8s_operator.main as operator_main
import aiperf_k8s_operator.reconciliation as reconciliation
import pytest
from aiperf_k8s_operator.contract import validate_bootstrap_metadata, validate_envelope
from aiperf_k8s_operator.main import reconcile_job
from aiperf_k8s_operator.reconciliation import build_jobset, validate_references

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "native-k8s" / "v1" / "fixtures"
PACKAGE = ROOT / "aiperf-k8s-operator" / "src" / "aiperf_k8s_operator"
RESULTS_UPLOAD_URL = "http://operator.system.svc:8080"
OBJECT_UID = "4f78fcbe-9aae-4cc9-ae19-204231b21575"


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def fixture_with_bootstrap_data() -> tuple[dict[str, object], dict[str, bytes]]:
    payload = fixture("valid-one-cell-envelope.json")
    bootstrap_data: dict[str, bytes] = {}
    for role in payload["roles"]:
        reference = role.get("bootstrap")
        if reference is None:
            continue
        data = f"private-{role['name']}-bootstrap".encode()
        reference["sha256"] = hashlib.sha256(data).hexdigest()
        bootstrap_data[reference["secretName"]] = data
    for reference in payload["cellBootstraps"]:
        data = f"private-cell-{reference['cellId']}-bootstrap".encode()
        reference["sha256"] = hashlib.sha256(data).hexdigest()
        bootstrap_data[reference["secretName"]] = data
    return payload, bootstrap_data


def reference_metadata(envelope: Any) -> dict[str, dict[str, Any]]:
    bootstraps = [
        *(
            role.bootstrap
            for role in envelope.roles
            if role.name != "cell" and role.bootstrap is not None
        ),
        *envelope.cell_bootstraps,
    ]
    return {
        bootstrap.secret_name: {
            "immutable": True,
            "metadata": {
                "name": bootstrap.secret_name,
                "labels": {
                    "aiperf.nvidia.com/run-id": envelope.run_id,
                    "aiperf.nvidia.com/role": bootstrap.role,
                },
                "annotations": {"aiperf.nvidia.com/sha256": bootstrap.sha256},
                "ownerReferences": [
                    reconciliation.owner_reference(envelope, OBJECT_UID)
                ],
            },
        }
        for bootstrap in bootstraps
    }


def test_operator_sources_never_import_legacy_aiperf_package() -> None:
    for source in PACKAGE.glob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        imports = [
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        ]
        imports.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert not [
            name for name in imports if name == "aiperf" or name.startswith("aiperf.")
        ], source


def test_envelope_requires_a_pullable_immutable_image_reference() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    payload["imageReference"] = payload["imageDigest"]

    with pytest.raises(ValueError):
        validate_envelope(payload)


def test_envelope_binds_image_reference_to_image_digest() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    payload["imageReference"] = "registry.example.com/aiperf/runner@sha256:" + "1" * 64

    with pytest.raises(
        ValueError, match="imageReference digest must equal imageDigest"
    ):
        validate_envelope(payload)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("namespace", "NOT_A_NAMESPACE"),
        ("namespace", "n" * 64),
        ("jobId", "../other"),
        ("jobId", "job.with.dot"),
        ("runId", "run/other"),
        ("runId", "r" * 64),
        ("artifactRoot", "/results/../secrets"),
        ("artifactRoot", "/etc"),
        ("artifactRoot", "/results//nested"),
    ],
)
def test_envelope_rejects_noncanonical_kubernetes_identity(
    field: str, invalid: str
) -> None:
    payload = fixture("valid-one-cell-envelope.json")
    payload[field] = invalid

    with pytest.raises(ValueError):
        validate_envelope(payload)


def test_envelope_requires_a_config_content_digest() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    payload["configRef"].pop("sha256", None)

    with pytest.raises(ValueError):
        validate_envelope(payload)


def test_jobset_uses_the_exact_digest_qualified_image_reference() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    image_reference = "registry.example.com/aiperf/runner@" + str(
        payload["imageDigest"]
    )
    payload["imageReference"] = image_reference
    envelope = validate_envelope(payload)

    jobset = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    jobs = jobset["spec"]["replicatedJobs"]
    assert all(
        container["image"] == image_reference
        for job in jobs
        for container in job["template"]["spec"]["template"]["spec"]["containers"]
    )


def test_envelope_projects_exact_two_jobsets() -> None:
    envelope = validate_envelope(fixture("valid-multi-cell-envelope.json"))
    jobset = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    assert jobset["metadata"] == {
        "name": envelope.job_id,
        "namespace": envelope.namespace,
        "ownerReferences": [reconciliation.owner_reference(envelope, OBJECT_UID)],
        "labels": {
            "aiperf.nvidia.com/run-id": envelope.run_id,
            "aiperf.nvidia.com/role": "jobset",
        },
        "annotations": {
            "aiperf.nvidia.com/sha256": "23b3b5b200304cfd8fcbeba285ac8745c0d3792f3b3bd0e73a8e44f9c3d3de1b"
        },
    }
    jobs = jobset["spec"]["replicatedJobs"]
    assert [job["name"] for job in jobs] == [
        "controller",
        "cell-0",
        "cell-1",
        "cell-2",
        "cell-3",
    ]
    controller_pod = jobs[0]["template"]["spec"]["template"]["spec"]
    cell_pods = [job["template"]["spec"]["template"]["spec"] for job in jobs[1:]]
    assert controller_pod["containers"][1]["name"] == "results-sidecar"
    assert all(job["replicas"] == 1 for job in jobs[1:])
    assert all(
        container["image"] == envelope.image_reference
        for pod in [controller_pod, *cell_pods]
        for container in pod["containers"]
    )

    roles = {role.name: role for role in envelope.roles}
    for pod in [controller_pod, *cell_pods]:
        assert pod["securityContext"] == {"runAsUser": 0}
    controller = controller_pod["containers"][0]
    assert controller["volumeMounts"][0] == {
        "name": "bootstrap-controller",
        "mountPath": roles["controller"].bootstrap.mount_path,
        "subPath": "bootstrap",
        "readOnly": True,
    }
    sidecar = controller_pod["containers"][1]
    assert all(
        not mount["name"].startswith("bootstrap-") for mount in sidecar["volumeMounts"]
    )

    controller_environment = {
        entry["name"]: entry["value"]
        for entry in controller_pod["containers"][0]["env"]
    }
    assert controller_environment["AIPERF_CELL_LAUNCHER"] == "k8s"
    assert controller_environment["AIPERF_CELL_COUNT"] == str(envelope.cells)
    assert (
        controller_environment["AIPERF_CONTROLLER_BOOTSTRAP_FILE"]
        == roles["controller"].bootstrap.mount_path
    )

    cell_bootstraps = {
        reference.cell_id: reference for reference in envelope.cell_bootstraps
    }
    for cell_id, pod in enumerate(cell_pods):
        bootstrap = cell_bootstraps[cell_id]
        container = pod["containers"][0]
        assert container["volumeMounts"][0] == {
            "name": f"bootstrap-cell-{cell_id}",
            "mountPath": bootstrap.mount_path,
            "subPath": "bootstrap",
            "readOnly": True,
        }
        environment = {entry["name"]: entry for entry in container["env"]}
        assert (
            environment["AIPERF_ROLE_BOOTSTRAP_FILE"]["value"] == bootstrap.mount_path
        )
        assert environment["AIPERF_CELL_LAUNCHER"]["value"] == "k8s"
        assert environment["AIPERF_CELL_COUNT"]["value"] == str(envelope.cells)
        assert (
            environment["AIPERF_CELL_CONTROLLER_ADDR"]["value"]
            == f"tcp://{envelope.controller_address}"
        )
        assert environment["AIPERF_CELL_ID"]["value"] == str(cell_id)
        assert {
            volume["name"]: volume["secret"]
            for volume in pod["volumes"]
            if "secret" in volume
        }[f"bootstrap-cell-{cell_id}"] == {
            "secretName": bootstrap.secret_name,
            "defaultMode": 0o600,
        }


def test_jobset_projects_reporting_identity_and_one_shared_results_volume() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    jobset = reconciliation.build_jobset(
        envelope,
        results_upload_base_url="http://operator.system.svc:8080",
        object_uid=OBJECT_UID,
    )
    jobs = jobset["spec"]["replicatedJobs"]
    controller_pod = jobs[0]["template"]["spec"]["template"]["spec"]
    cell_pod = jobs[1]["template"]["spec"]["template"]["spec"]
    controller, sidecar = controller_pod["containers"]
    expected_config = reconciliation.config_snapshot_name(envelope, OBJECT_UID)
    assert all(
        next(volume for volume in pod["volumes"] if volume["name"] == "config")
        == {"name": "config", "configMap": {"name": expected_config}}
        for pod in (controller_pod, cell_pod)
    )
    assert expected_config != envelope.config_ref.name

    assert controller_pod["serviceAccountName"] == reconciliation.workload_name(
        envelope
    )
    assert controller_pod["automountServiceAccountToken"] is False
    assert cell_pod["automountServiceAccountToken"] is False
    assert controller_pod["volumes"][-2:] == [
        {"name": "results", "emptyDir": {}},
        {
            "name": "controller-kube-api",
            "projected": {
                "defaultMode": 0o600,
                "sources": [
                    {
                        "serviceAccountToken": {
                            "path": "token",
                            "expirationSeconds": 3600,
                        }
                    },
                    {
                        "configMap": {
                            "name": "kube-root-ca.crt",
                            "items": [{"key": "ca.crt", "path": "ca.crt"}],
                        }
                    },
                ],
            },
        },
    ]
    for container in (controller, sidecar):
        assert {mount["name"]: mount for mount in container["volumeMounts"]}[
            "results"
        ] == {"name": "results", "mountPath": "/results"}

    controller_environment = {item["name"]: item["value"] for item in controller["env"]}
    assert controller_environment["AIPERF_JOB_ID"] == "job-1"
    assert controller_environment["AIPERF_NAMESPACE"] == "bench"
    assert controller_environment["AIPERF_RUN_ID"] == "run-1"
    assert controller_environment["AIPERF_JOB_UID"] == OBJECT_UID
    assert {mount["name"] for mount in controller["volumeMounts"]} >= {
        "controller-kube-api"
    }
    assert "controller-kube-api" not in {
        mount["name"] for mount in sidecar["volumeMounts"]
    }

    sidecar_environment = {item["name"]: item["value"] for item in sidecar["env"]}
    assert sidecar_environment == {
        "AIPERF_CELL_LAUNCHER": "k8s",
        "AIPERF_JOB_ID": "job-1",
        "AIPERF_NAMESPACE": "bench",
        "AIPERF_RUN_ID": "run-1",
        "AIPERF_RESULTS_DIR": "/results",
        "AIPERF_RESULTS_UPLOAD_URL": "http://operator.system.svc:8080",
    }
    assert jobset["metadata"]["ownerReferences"] == [
        {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "name": "job-1",
            "uid": OBJECT_UID,
            "controller": True,
        }
    ]


def test_workload_identity_is_per_run_and_can_patch_only_status() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    service_account, role, binding = reconciliation.build_workload_identity(
        envelope, OBJECT_UID
    )
    expected_name = reconciliation.workload_name(envelope)

    assert expected_name.startswith("aiperf-workload-")
    assert len(expected_name) <= 63
    assert service_account["metadata"]["name"] == expected_name
    assert service_account["metadata"]["namespace"] == "bench"
    assert role["rules"] == [
        {
            "apiGroups": ["aiperf.nvidia.com"],
            "resources": ["aiperfjobs/status"],
            "resourceNames": ["job-1"],
            "verbs": ["patch"],
        },
    ]
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": expected_name,
    }
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": expected_name, "namespace": "bench"}
    ]
    for resource in (service_account, role, binding):
        assert resource["metadata"]["ownerReferences"] == [
            reconciliation.owner_reference(envelope, OBJECT_UID)
        ]


def test_config_snapshot_binds_verified_content_to_the_cr_incarnation() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    content = {
        "data": {"benchmark.yaml": "profile: true\n"},
        "binaryData": {"token.bin": "AAE="},
    }
    canonical = b'{"binaryData":{"token.bin":"AAE="},"data":{"benchmark.yaml":"profile: true\\n"}}'
    payload["configRef"]["sha256"] = hashlib.sha256(canonical).hexdigest()
    envelope = validate_envelope(payload)
    source = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": envelope.config_ref.name,
            "namespace": envelope.namespace,
            "labels": {"attacker": "must-not-be-copied"},
        },
        **content,
    }

    snapshot = reconciliation.build_config_snapshot(envelope, OBJECT_UID, source)

    assert snapshot == {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": reconciliation.config_snapshot_name(envelope, OBJECT_UID),
            "namespace": "bench",
            "ownerReferences": [reconciliation.owner_reference(envelope, OBJECT_UID)],
            "labels": {
                "aiperf.nvidia.com/namespace": "bench",
                "aiperf.nvidia.com/job-id": "job-1",
                "aiperf.nvidia.com/run-id": "run-1",
                "aiperf.nvidia.com/object-uid": OBJECT_UID,
                "aiperf.nvidia.com/role": "config-snapshot",
            },
            "annotations": {
                "aiperf.nvidia.com/envelope-sha256": reconciliation.envelope_sha256(
                    envelope
                ),
                "aiperf.nvidia.com/content-sha256": envelope.config_ref.sha256,
            },
        },
        "immutable": True,
        **content,
    }

    source["data"]["benchmark.yaml"] = "privileged: true\n"
    with pytest.raises(ValueError, match="ConfigMap content digest"):
        reconciliation.build_config_snapshot(envelope, OBJECT_UID, source)


class WorkloadCoreApi:
    def __init__(self) -> None:
        self.service_accounts: dict[tuple[str, str], dict[str, Any]] = {}

    async def create_namespaced_service_account(
        self, namespace: str, body: dict[str, Any]
    ) -> None:
        from kubernetes_asyncio.client.exceptions import ApiException

        key = (namespace, body["metadata"]["name"])
        if key in self.service_accounts:
            raise ApiException(status=409, reason="AlreadyExists")
        self.service_accounts[key] = copy.deepcopy(body)

    async def read_namespaced_service_account(
        self, name: str, namespace: str
    ) -> dict[str, Any]:
        return copy.deepcopy(self.service_accounts[(namespace, name)])


class WorkloadRbacApi:
    def __init__(self) -> None:
        self.roles: dict[tuple[str, str], dict[str, Any]] = {}
        self.bindings: dict[tuple[str, str], dict[str, Any]] = {}

    async def create_namespaced_role(
        self, namespace: str, body: dict[str, Any]
    ) -> None:
        await self._create(self.roles, namespace, body)

    async def read_namespaced_role(self, name: str, namespace: str) -> dict[str, Any]:
        return copy.deepcopy(self.roles[(namespace, name)])

    async def create_namespaced_role_binding(
        self, namespace: str, body: dict[str, Any]
    ) -> None:
        await self._create(self.bindings, namespace, body)

    async def read_namespaced_role_binding(
        self, name: str, namespace: str
    ) -> dict[str, Any]:
        return copy.deepcopy(self.bindings[(namespace, name)])

    @staticmethod
    async def _create(
        resources: dict[tuple[str, str], dict[str, Any]],
        namespace: str,
        body: dict[str, Any],
    ) -> None:
        from kubernetes_asyncio.client.exceptions import ApiException

        key = (namespace, body["metadata"]["name"])
        if key in resources:
            raise ApiException(status=409, reason="AlreadyExists")
        resources[key] = copy.deepcopy(body)


@pytest.mark.asyncio
async def test_workload_identity_reconcile_is_idempotent_for_exact_resources() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    core = WorkloadCoreApi()
    rbac = WorkloadRbacApi()

    await operator_main.ensure_workload_identity(envelope, OBJECT_UID, core, rbac)
    first = (copy.deepcopy(core.service_accounts), copy.deepcopy(rbac.roles))
    await operator_main.ensure_workload_identity(envelope, OBJECT_UID, core, rbac)

    assert (core.service_accounts, rbac.roles) == first
    assert len(rbac.bindings) == 1


@pytest.mark.asyncio
async def test_workload_identity_reconcile_rejects_conflicting_role() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    core = WorkloadCoreApi()
    rbac = WorkloadRbacApi()
    await operator_main.ensure_workload_identity(envelope, OBJECT_UID, core, rbac)
    role = next(iter(rbac.roles.values()))
    role["rules"][0]["resourceNames"] = ["other-job"]

    with pytest.raises(ValueError, match="existing workload Role does not match"):
        await operator_main.ensure_workload_identity(envelope, OBJECT_UID, core, rbac)


def test_metadata_validation_never_reads_secret_data() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = reference_metadata(envelope)
    metadata[envelope.roles[0].bootstrap.secret_name]["data"] = {
        "must-not-be-read": "not-a-real-secret"
    }
    validate_references(envelope, metadata)
    with pytest.raises(ValueError, match="role label"):
        validate_bootstrap_metadata(
            envelope.roles[0].bootstrap,
            {
                "immutable": True,
                "metadata": {"name": envelope.roles[0].bootstrap.secret_name},
            },
        )


def test_reference_validation_rejects_wrong_run_id() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = reference_metadata(envelope)
    metadata[envelope.roles[0].bootstrap.secret_name]["metadata"]["labels"][
        "aiperf.nvidia.com/run-id"
    ] = "other-run"

    with pytest.raises(ValueError, match="run-id label"):
        validate_references(envelope, metadata)


def test_reference_validation_requires_exact_cr_owner_binding() -> None:
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = reference_metadata(envelope)
    name = envelope.roles[0].bootstrap.secret_name
    metadata[name]["metadata"]["ownerReferences"][0]["uid"] = "other-incarnation"

    with pytest.raises(ValueError, match="owner reference"):
        validate_references(envelope, metadata, OBJECT_UID)


def test_envelope_rejects_duplicate_cell_bootstrap_secrets() -> None:
    payload = fixture("valid-multi-cell-envelope.json")
    payload["cellBootstraps"][1]["secretName"] = payload["cellBootstraps"][0][
        "secretName"
    ]

    with pytest.raises(ValueError, match="bootstrap Secret names must be unique"):
        validate_envelope(payload)


def test_envelope_rejects_huge_cell_count_without_expanding_bootstraps() -> None:
    payload = fixture("valid-one-cell-envelope.json")
    payload["cells"] = 1_000_000_000

    with pytest.raises(
        ValueError, match="cellBootstraps must contain each cell id exactly once"
    ):
        validate_envelope(payload)


def test_envelope_requires_an_unambiguous_controller_coordinate() -> None:
    malformed = fixture("valid-one-cell-envelope.json")
    malformed["controllerAddress"] = "controller:443:8443"
    with pytest.raises(
        ValueError,
        match="controllerAddress must be tcp://HOST:PORT or tcp://\\[IPv6\\]:PORT",
    ):
        validate_envelope(malformed)

    ipv6 = fixture("valid-one-cell-envelope.json")
    ipv6["controllerAddress"] = "tcp://[2001:db8::1]:443"
    envelope = validate_envelope(ipv6)
    cell = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)["spec"][
        "replicatedJobs"
    ][1]
    environment = {
        entry["name"]: entry["value"]
        for entry in cell["template"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert environment["AIPERF_CELL_CONTROLLER_ADDR"] == "tcp://[2001:db8::1]:443"


def test_controller_container_receives_the_envelope_controller_port() -> None:
    # valid-one-cell-envelope.json has controllerAddress "controller:443"
    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    jobset = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    jobs = jobset["spec"]["replicatedJobs"]
    controller_pod = jobs[0]["template"]["spec"]["template"]["spec"]
    controller_env = {
        entry["name"]: entry["value"]
        for entry in controller_pod["containers"][0]["env"]
    }
    assert controller_env["AIPERF_CONTROLLER_PORT"] == "443"

    # No cell container carries AIPERF_CONTROLLER_PORT
    for job in jobs[1:]:
        cell_env = {
            entry["name"]: entry.get("value")
            for entry in job["template"]["spec"]["template"]["spec"]["containers"][0][
                "env"
            ]
        }
        assert "AIPERF_CONTROLLER_PORT" not in cell_env


@pytest.mark.asyncio
async def test_reconcile_creates_projected_jobset() -> None:
    class FakeJobSets:
        kwargs: dict[str, object]

        async def create_namespaced_custom_object(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    jobsets = FakeJobSets()
    status = await reconcile_job(
        envelope,
        jobsets,
        reference_metadata(envelope),
        RESULTS_UPLOAD_URL,
        OBJECT_UID,
    )

    assert status == {
        "phase": "Pending",
        "runId": envelope.run_id,
        "jobSet": envelope.job_id,
    }
    assert jobsets.kwargs["group"] == "jobset.x-k8s.io"
    assert jobsets.kwargs["namespace"] == envelope.namespace
    assert jobsets.kwargs["body"] == build_jobset(
        envelope, RESULTS_UPLOAD_URL, OBJECT_UID
    )


@pytest.mark.asyncio
async def test_reconcile_rejects_wrong_reference_before_creating_jobset() -> None:
    class FakeJobSets:
        was_created = False

        async def create_namespaced_custom_object(self, **_: object) -> None:
            self.was_created = True

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = reference_metadata(envelope)
    metadata[envelope.roles[0].bootstrap.secret_name]["metadata"]["labels"][
        "aiperf.nvidia.com/role"
    ] = "cell"
    jobsets = FakeJobSets()

    with pytest.raises(ValueError, match="role label"):
        await reconcile_job(envelope, jobsets, metadata, RESULTS_UPLOAD_URL, OBJECT_UID)
    assert not jobsets.was_created


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_for_matching_existing_jobset() -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    status = await reconcile_job(
        envelope,
        ExistingJobSet(),
        reference_metadata(envelope),
        RESULTS_UPLOAD_URL,
        OBJECT_UID,
    )
    assert status == {
        "phase": "Pending",
        "runId": envelope.run_id,
        "jobSet": envelope.job_id,
    }


@pytest.mark.asyncio
async def test_reconcile_accepts_the_pinned_jobset_group_default() -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    admitted = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    for job in admitted["spec"]["replicatedJobs"]:
        job["groupName"] = "default"

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return admitted

    status = await reconcile_job(
        envelope,
        ExistingJobSet(),
        reference_metadata(envelope),
        RESULTS_UPLOAD_URL,
        OBJECT_UID,
    )

    assert status["phase"] == "Pending"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "suspend"), True),
        (
            (
                "spec",
                "replicatedJobs",
                0,
                "template",
                "spec",
                "template",
                "spec",
                "hostNetwork",
            ),
            True,
        ),
        (
            (
                "spec",
                "replicatedJobs",
                0,
                "template",
                "spec",
                "template",
                "spec",
                "containers",
                0,
                "securityContext",
            ),
            {"privileged": True},
        ),
    ],
)
async def test_reconcile_rejects_existing_jobset_with_extra_execution_authority(
    path: tuple[str | int, ...], value: object
) -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    existing = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    target: Any = existing
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return existing

    with pytest.raises(ValueError, match="JobSet does not match submitted envelope"):
        await reconcile_job(
            envelope,
            ExistingJobSet(),
            reference_metadata(envelope),
            RESULTS_UPLOAD_URL,
            OBJECT_UID,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("run", "other-run"),
        ("role", "controller"),
        (
            "digest",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        ),
    ],
)
async def test_reconcile_rejects_conflicting_jobset_identity(
    field: str, wrong_value: str
) -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    existing = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    if field == "run":
        existing["metadata"]["labels"]["aiperf.nvidia.com/run-id"] = wrong_value
    elif field == "role":
        existing["metadata"]["labels"]["aiperf.nvidia.com/role"] = wrong_value
    else:
        existing["metadata"].setdefault("annotations", {})[
            "aiperf.nvidia.com/sha256"
        ] = wrong_value

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return existing

    with pytest.raises(ValueError, match="JobSet does not match submitted envelope"):
        await reconcile_job(
            envelope,
            ExistingJobSet(),
            reference_metadata(envelope),
            RESULTS_UPLOAD_URL,
            OBJECT_UID,
        )


@pytest.mark.asyncio
async def test_reconcile_rejects_existing_jobset_with_modified_pod_spec() -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    existing = build_jobset(envelope, RESULTS_UPLOAD_URL, OBJECT_UID)
    controller = existing["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]["containers"][0]
    controller["command"] = ["malicious-controller"]

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return existing

    with pytest.raises(ValueError, match="JobSet does not match submitted envelope"):
        await reconcile_job(
            envelope,
            ExistingJobSet(),
            reference_metadata(envelope),
            RESULTS_UPLOAD_URL,
            OBJECT_UID,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_name", "event_namespace"),
    [("other-job", "bench"), ("job-1", "other-namespace")],
)
async def test_create_handler_rejects_cross_object_identity_before_api_access(
    monkeypatch: pytest.MonkeyPatch, event_name: str, event_namespace: str
) -> None:
    def forbidden_api() -> None:
        raise AssertionError("identity mismatch reached Kubernetes API access")

    monkeypatch.setattr(operator_main.client, "CoreV1Api", forbidden_api)
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", forbidden_api)

    with pytest.raises(ValueError, match="AIPerfJob metadata does not match envelope"):
        await operator_main.create_job(
            {"envelope": fixture("valid-one-cell-envelope.json")},
            name=event_name,
            namespace=event_namespace,
            uid=OBJECT_UID,
            patch={},
        )


@pytest.mark.asyncio
async def test_create_handler_revalidates_references_without_results_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubernetes_asyncio import client as kubernetes_client

    payload, bootstrap_data = fixture_with_bootstrap_data()
    config_data = {"benchmark.yaml": "profile: true\n"}
    canonical_config = b'{"binaryData":{},"data":{"benchmark.yaml":"profile: true\\n"}}'
    payload["configRef"]["sha256"] = hashlib.sha256(canonical_config).hexdigest()
    envelope = validate_envelope(payload)
    identities = reference_metadata(envelope)
    events: list[str] = []

    class FakeCore(WorkloadCoreApi):
        async def read_namespaced_secret(
            self, name: str, namespace: str
        ) -> kubernetes_client.V1Secret:
            events.append(f"bootstrap:{name}")
            identity = identities[name]
            return kubernetes_client.V1Secret(
                immutable=True,
                metadata=kubernetes_client.V1ObjectMeta(
                    name=name,
                    namespace=namespace,
                    labels=identity["metadata"]["labels"],
                    annotations=identity["metadata"]["annotations"],
                    owner_references=[
                        kubernetes_client.V1OwnerReference(
                            api_version="aiperf.nvidia.com/v1alpha1",
                            kind="AIPerfJob",
                            name=envelope.job_id,
                            uid=OBJECT_UID,
                            controller=True,
                        )
                    ],
                ),
                data={"bootstrap": base64.b64encode(bootstrap_data[name]).decode()},
            )

        async def create_namespaced_config_map(
            self, namespace: str, body: dict[str, Any]
        ) -> None:
            assert body["metadata"]["name"] == reconciliation.config_snapshot_name(
                envelope, OBJECT_UID
            )
            events.append("config-snapshot")
            self.config_snapshot = copy.deepcopy(body)

        async def read_namespaced_config_map(
            self, name: str, namespace: str
        ) -> dict[str, Any]:
            if name == envelope.config_ref.name:
                events.append("config-source")
                return {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {"name": name, "namespace": namespace},
                    "data": config_data,
                }
            raise AssertionError("only the source ConfigMap may be read")

    class FakeObjects:
        status: dict[str, Any] | None = None

        async def patch_namespaced_custom_object(self, **_: Any) -> None:
            raise AssertionError("reconciliation must not patch mutable CR metadata")

        async def patch_namespaced_custom_object_status(self, **kwargs: Any) -> None:
            events.append(f"status:{kwargs['body']['status']['phase']}")
            self.status = copy.deepcopy(kwargs["body"]["status"])

        async def create_namespaced_custom_object(self, **_: object) -> None:
            events.append("jobset")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            events.append("cr-status" if self.status is None else "cr-revalidation")
            return {
                "metadata": {
                    "name": envelope.job_id,
                    "namespace": envelope.namespace,
                    "uid": OBJECT_UID,
                },
                "spec": {"envelope": payload},
                **({"status": self.status} if self.status is not None else {}),
            }

    class FakeApiClient:
        was_closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            self.was_closed = True

    core = FakeCore()
    rbac = WorkloadRbacApi()
    objects = FakeObjects()
    upload_url = "http://operator.system.svc:8080"
    monkeypatch.setenv("AIPERF_K8S_OPERATOR_RESULTS_UPLOAD_BASE_URL", upload_url)
    api_client = FakeApiClient()
    monkeypatch.setattr(operator_main.client, "ApiClient", lambda: api_client)
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda _: core)
    monkeypatch.setattr(operator_main.client, "RbacAuthorizationV1Api", lambda _: rbac)
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", lambda _: objects)

    status_patch: dict[str, Any] = {}
    result = await operator_main.create_job(
        {"envelope": payload},
        name=envelope.job_id,
        namespace=envelope.namespace,
        uid=OBJECT_UID,
        patch=status_patch,
    )

    assert events.index("cr-status") < events.index("status:Pending")
    assert events.index("status:Pending") < events.index("jobset")
    assert events.index("config-source") < events.index("config-snapshot")
    assert events.index("config-snapshot") < events.index("jobset")
    assert events.index("jobset") < events.index("cr-revalidation")
    assert events[-1].startswith("bootstrap:")
    assert {
        event.removeprefix("bootstrap:")
        for event in events
        if event.startswith("bootstrap:")
    } == set(identities)
    assert all(events.count(f"bootstrap:{name}") == 2 for name in identities)
    assert "read-capability" not in events
    assert "authority" not in events
    assert api_client.was_closed
    assert result is None
    assert status_patch == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missing_field", "message"),
    [("labels", "role label"), ("annotations", "digest annotation")],
)
async def test_create_handler_rejects_absent_secret_metadata_maps(
    monkeypatch: pytest.MonkeyPatch, missing_field: str, message: str
) -> None:
    from kubernetes_asyncio import client as kubernetes_client

    payload, bootstrap_data = fixture_with_bootstrap_data()
    envelope = validate_envelope(payload)
    identities = reference_metadata(envelope)

    class FakeSecrets:
        def __init__(self, *_: object) -> None:
            pass

        async def read_namespaced_secret(
            self, name: str, namespace: str
        ) -> kubernetes_client.V1Secret:
            metadata = identities[name]["metadata"]
            values = {
                "labels": metadata["labels"],
                "annotations": metadata["annotations"],
            }
            values[missing_field] = None
            return kubernetes_client.V1Secret(
                immutable=True,
                metadata=kubernetes_client.V1ObjectMeta(
                    name=name,
                    namespace=namespace,
                    **values,
                    owner_references=[
                        kubernetes_client.V1OwnerReference(
                            api_version="aiperf.nvidia.com/v1alpha1",
                            kind="AIPerfJob",
                            name=envelope.job_id,
                            uid=OBJECT_UID,
                            controller=True,
                        )
                    ],
                ),
                data={"bootstrap": base64.b64encode(bootstrap_data[name]).decode()},
            )

    class FakeJobSets:
        def __init__(self, *_: object) -> None:
            pass

        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise AssertionError("invalid Secret metadata reached JobSet creation")

    class FakeApiClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

    monkeypatch.setattr(operator_main.client, "ApiClient", FakeApiClient)
    monkeypatch.setattr(operator_main.client, "CoreV1Api", FakeSecrets)
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", FakeJobSets)
    monkeypatch.setattr(
        operator_main.client, "RbacAuthorizationV1Api", lambda _: WorkloadRbacApi()
    )

    with pytest.raises(ValueError, match=message):
        await operator_main.create_job(
            {"envelope": payload},
            name=envelope.job_id,
            namespace=envelope.namespace,
            uid=OBJECT_UID,
            patch={},
        )
