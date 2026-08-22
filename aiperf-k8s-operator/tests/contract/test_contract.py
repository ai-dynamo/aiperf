# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path
from typing import Any

import aiperf_k8s_operator.main as operator_main
import pytest
from aiperf_k8s_operator.contract import validate_bootstrap_metadata, validate_envelope
from aiperf_k8s_operator.main import reconcile_job
from aiperf_k8s_operator.reconciliation import build_jobset, validate_references

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "contracts" / "native-k8s" / "v1" / "fixtures"
PACKAGE = ROOT / "aiperf-k8s-operator" / "src" / "aiperf_k8s_operator"


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


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


def test_envelope_projects_exact_two_jobsets() -> None:
    envelope = validate_envelope(fixture("valid-multi-cell-envelope.json"))
    jobset = build_jobset(envelope)
    assert jobset["metadata"] == {
        "name": envelope.job_id,
        "namespace": envelope.namespace,
        "labels": {
            "aiperf.nvidia.com/run-id": envelope.run_id,
            "aiperf.nvidia.com/role": "jobset",
        },
        "annotations": {
            "aiperf.nvidia.com/sha256": "4686fd14d91975667fd3fc3164d113d70fad54452d0b7e4b146aaba6adc5d77c"
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
        container["image"] == envelope.image_digest
        for pod in [controller_pod, *cell_pods]
        for container in pod["containers"]
    )

    roles = {role.name: role for role in envelope.roles}
    for pod in [controller_pod, *cell_pods]:
        assert pod["securityContext"] == {"runAsUser": 0}
    for container in controller_pod["containers"]:
        role = roles[container["name"]]
        assert container["volumeMounts"][0] == {
            "name": f"bootstrap-{role.name}",
            "mountPath": role.bootstrap.mount_path,
            "subPath": "bootstrap",
            "readOnly": True,
        }

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
    cell = build_jobset(envelope)["spec"]["replicatedJobs"][1]
    environment = {
        entry["name"]: entry["value"]
        for entry in cell["template"]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    assert environment["AIPERF_CELL_CONTROLLER_ADDR"] == "tcp://[2001:db8::1]:443"


@pytest.mark.asyncio
async def test_reconcile_creates_projected_jobset() -> None:
    class FakeJobSets:
        kwargs: dict[str, object]

        async def create_namespaced_custom_object(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    jobsets = FakeJobSets()
    status = await reconcile_job(envelope, jobsets, reference_metadata(envelope))

    assert status == {
        "phase": "Pending",
        "runId": envelope.run_id,
        "jobSet": envelope.job_id,
    }
    assert jobsets.kwargs["group"] == "jobset.x-k8s.io"
    assert jobsets.kwargs["namespace"] == envelope.namespace
    assert jobsets.kwargs["body"] == build_jobset(envelope)


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
        await reconcile_job(envelope, jobsets, metadata)
    assert not jobsets.was_created


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_for_matching_existing_jobset() -> None:
    from kubernetes_asyncio.client.exceptions import ApiException

    class ExistingJobSet:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            raise ApiException(status=409, reason="AlreadyExists")

        async def get_namespaced_custom_object(self, **_: object) -> dict[str, Any]:
            return build_jobset(envelope)

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    status = await reconcile_job(
        envelope, ExistingJobSet(), reference_metadata(envelope)
    )
    assert status == {
        "phase": "Pending",
        "runId": envelope.run_id,
        "jobSet": envelope.job_id,
    }


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
    existing = build_jobset(envelope)
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

    with pytest.raises(ValueError, match="JobSet identity does not match"):
        await reconcile_job(envelope, ExistingJobSet(), reference_metadata(envelope))


@pytest.mark.asyncio
async def test_create_handler_loads_each_secret_reference_before_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kubernetes_asyncio import client as kubernetes_client

    envelope = validate_envelope(fixture("valid-one-cell-envelope.json"))
    metadata = reference_metadata(envelope)

    class FakeSecrets:
        names: list[str] = []

        async def read_namespaced_secret(
            self, name: str, namespace: str
        ) -> kubernetes_client.V1Secret:
            self.names.append(name)
            identity = metadata[name]
            return kubernetes_client.V1Secret(
                immutable=True,
                metadata=kubernetes_client.V1ObjectMeta(
                    name=name,
                    namespace=namespace,
                    labels=identity["metadata"]["labels"],
                    annotations=identity["metadata"]["annotations"],
                ),
            )

    class FakeJobSets:
        async def create_namespaced_custom_object(self, **_: object) -> None:
            return None

    secrets = FakeSecrets()
    monkeypatch.setattr(operator_main.client, "CoreV1Api", lambda: secrets)
    monkeypatch.setattr(operator_main.client, "CustomObjectsApi", FakeJobSets)

    result = await operator_main.create_job(
        {"envelope": fixture("valid-one-cell-envelope.json")}
    )

    assert result["status"]["runId"] == envelope.run_id
    assert set(secrets.names) == set(metadata)
