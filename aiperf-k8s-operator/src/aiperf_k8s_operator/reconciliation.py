# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile immutable native envelopes into the three-role JobSet topology."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from .contract import (
    BootstrapReference,
    CellBootstrapReference,
    ControllerEnvelope,
    RoleEnvelope,
    validate_bootstrap_metadata,
)


def _controller_coordinate(address: str) -> str:
    return address if address.startswith("tcp://") else f"tcp://{address}"


def _container(
    role: RoleEnvelope,
    envelope: ControllerEnvelope,
    bootstrap: BootstrapReference | CellBootstrapReference,
    results_upload_base_url: str,
    object_uid: str,
    cell_id: int | None = None,
) -> dict[str, Any]:
    """Project one immutable role without interpreting its command, argv, or image."""
    environment: dict[str, Any] = dict(role.environment)
    environment["AIPERF_CELL_LAUNCHER"] = "k8s"
    environment["AIPERF_ROLE_BOOTSTRAP_FILE"] = bootstrap.mount_path
    if role.name == "controller":
        environment["AIPERF_JOB_ID"] = envelope.job_id
        environment["AIPERF_JOB_UID"] = object_uid
        environment["AIPERF_NAMESPACE"] = envelope.namespace
        environment["AIPERF_RUN_ID"] = envelope.run_id
        environment["AIPERF_CELL_COUNT"] = str(envelope.cells)
        environment["AIPERF_CONTROLLER_BOOTSTRAP_FILE"] = bootstrap.mount_path
    elif role.name == "cell":
        if cell_id is None:
            raise ValueError("cell projection requires a cell id")
        environment["AIPERF_CELL_COUNT"] = str(envelope.cells)
        environment["AIPERF_CELL_CONTROLLER_ADDR"] = _controller_coordinate(
            envelope.controller_address
        )
        environment["AIPERF_CELL_ID"] = str(cell_id)
    else:
        environment["AIPERF_JOB_ID"] = envelope.job_id
        environment["AIPERF_JOB_UID"] = object_uid
        environment["AIPERF_NAMESPACE"] = envelope.namespace
        environment["AIPERF_RESULTS_DIR"] = envelope.artifact_root
        environment["AIPERF_RESULTS_UPLOAD_URL"] = results_upload_base_url
        environment["AIPERF_RUN_ID"] = envelope.run_id
    volume_mounts = [
        {
            "name": f"bootstrap-{bootstrap_name(bootstrap)}",
            "mountPath": bootstrap.mount_path,
            "subPath": "bootstrap",
            "readOnly": True,
        },
        {"name": "config", "mountPath": "/etc/aiperf/config", "readOnly": True},
    ]
    if role.name in {"controller", "results-sidecar"}:
        volume_mounts.append({"name": "results", "mountPath": envelope.artifact_root})
        volume_mounts.append(
            {
                "name": "authority-gate",
                "mountPath": "/var/run/aiperf/authority",
                "readOnly": True,
            }
        )
    if role.name == "controller":
        volume_mounts.append(
            {
                "name": "controller-kube-api",
                "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
                "readOnly": True,
            }
        )
    return {
        "name": role.name,
        "image": envelope.image_reference,
        "command": role.command,
        "args": role.argv,
        "env": [
            {"name": key, **({"value": value} if isinstance(value, str) else value)}
            for key, value in sorted(environment.items())
        ],
        "volumeMounts": volume_mounts,
    }


def _role_by_name(envelope: ControllerEnvelope, name: str) -> RoleEnvelope:
    return next(role for role in envelope.roles if role.name == name)


def _role_bootstrap(role: RoleEnvelope) -> BootstrapReference:
    if role.bootstrap is None:
        raise ValueError(f"{role.name} role has no bootstrap reference")
    return role.bootstrap


def _volumes(
    bootstraps: list[BootstrapReference | CellBootstrapReference],
    envelope: ControllerEnvelope,
    object_uid: str,
    *,
    has_results: bool = False,
) -> list[dict[str, Any]]:
    volumes = [
        *[
            {
                "name": f"bootstrap-{bootstrap_name(bootstrap)}",
                "secret": {
                    "secretName": bootstrap.secret_name,
                    "defaultMode": 0o600,
                },
            }
            for bootstrap in bootstraps
        ],
        {
            "name": "config",
            "configMap": {"name": config_snapshot_name(envelope, object_uid)},
        },
    ]
    if has_results:
        volumes.extend(
            [
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
                {
                    "name": "authority-gate",
                    "configMap": {
                        "name": authority_name(envelope, object_uid),
                        "optional": False,
                    },
                },
            ]
        )
    return volumes


def bootstrap_name(bootstrap: BootstrapReference | CellBootstrapReference) -> str:
    if isinstance(bootstrap, CellBootstrapReference):
        return f"cell-{bootstrap.cell_id}"
    return bootstrap.role


def workload_name(envelope: ControllerEnvelope) -> str:
    """Return a DNS-safe per-run workload identity name."""
    identity = f"{envelope.namespace}\0{envelope.job_id}\0{envelope.run_id}".encode()
    return f"aiperf-workload-{hashlib.sha256(identity).hexdigest()[:16]}"


def _incarnation_suffix(envelope: ControllerEnvelope, object_uid: str) -> str:
    identity = (
        f"{envelope.namespace}\0{envelope.job_id}\0{envelope.run_id}\0{object_uid}"
    ).encode()
    return hashlib.sha256(identity).hexdigest()[:16]


def authority_name(envelope: ControllerEnvelope, object_uid: str) -> str:
    """Return the deterministic immutable authority ConfigMap name."""
    return f"aiperf-results-authority-{_incarnation_suffix(envelope, object_uid)}"


def config_snapshot_name(envelope: ControllerEnvelope, object_uid: str) -> str:
    """Return the deterministic immutable configuration snapshot name."""
    return f"aiperf-config-{_incarnation_suffix(envelope, object_uid)}"


def results_read_secret_name(envelope: ControllerEnvelope, object_uid: str) -> str:
    """Return the deterministic dedicated results-read Secret name."""
    return f"aiperf-results-read-{_incarnation_suffix(envelope, object_uid)}"


def owner_reference(envelope: ControllerEnvelope, object_uid: str) -> dict[str, Any]:
    """Return the exact AIPerfJob owner reference for this object incarnation."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "name": envelope.job_id,
        "uid": object_uid,
        "controller": True,
    }


def _authority_labels(
    envelope: ControllerEnvelope, object_uid: str, role: str
) -> dict[str, str]:
    labels = {
        "aiperf.nvidia.com/namespace": envelope.namespace,
        "aiperf.nvidia.com/job-id": envelope.job_id,
        "aiperf.nvidia.com/run-id": envelope.run_id,
        "aiperf.nvidia.com/object-uid": object_uid,
        "aiperf.nvidia.com/role": role,
    }
    if any(not _is_valid_label_value(value) for value in labels.values()):
        raise ValueError("AIPerfJob authority identity is not a Kubernetes label value")
    return labels


def _is_valid_label_value(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 63
        and value[0].isalnum()
        and value[-1].isalnum()
        and value.isascii()
        and all(character.isalnum() or character in "-_." for character in value)
    )


def _authority_annotations(envelope: ControllerEnvelope) -> dict[str, str]:
    return {"aiperf.nvidia.com/envelope-sha256": envelope_sha256(envelope)}


def build_results_read_secret(
    envelope: ControllerEnvelope, object_uid: str, raw_token: bytes
) -> dict[str, Any]:
    """Build one immutable, object-incarnation-bound read capability Secret."""
    if len(raw_token) != 32:
        raise ValueError("results-read capability must contain exactly 32 bytes")
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": results_read_secret_name(envelope, object_uid),
            "namespace": envelope.namespace,
            "ownerReferences": [owner_reference(envelope, object_uid)],
            "labels": _authority_labels(envelope, object_uid, "results-read"),
            "annotations": _authority_annotations(envelope),
        },
        "immutable": True,
        "type": "Opaque",
        "data": {"token": base64.b64encode(raw_token).decode("ascii")},
    }


def build_results_authority(
    envelope: ControllerEnvelope,
    object_uid: str,
    upload_public_key: str,
    read_token_sha256: str,
) -> dict[str, Any]:
    """Build one immutable atomic verifier record for the accepted workload."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": authority_name(envelope, object_uid),
            "namespace": envelope.namespace,
            "ownerReferences": [owner_reference(envelope, object_uid)],
            "labels": _authority_labels(envelope, object_uid, "results-authority"),
            "annotations": _authority_annotations(envelope),
        },
        "immutable": True,
        "data": {
            "uploadPublicKey": upload_public_key,
            "readTokenSha256": read_token_sha256,
        },
    }


def build_config_snapshot(
    envelope: ControllerEnvelope, object_uid: str, source: dict[str, Any]
) -> dict[str, Any]:
    """Verify and freeze the complete source ConfigMap content for one run."""
    metadata = source.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") != envelope.config_ref.name
        or metadata.get("namespace") != envelope.namespace
    ):
        raise ValueError("source ConfigMap identity does not match configRef")
    content: dict[str, dict[str, str]] = {}
    for field in ("data", "binaryData"):
        value = source.get(field, {})
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        ):
            raise ValueError(f"source ConfigMap {field} must be a string map")
        content[field] = dict(value)
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if actual_digest != envelope.config_ref.sha256:
        raise ValueError("source ConfigMap content digest does not match configRef")
    snapshot = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": config_snapshot_name(envelope, object_uid),
            "namespace": envelope.namespace,
            "ownerReferences": [owner_reference(envelope, object_uid)],
            "labels": _authority_labels(envelope, object_uid, "config-snapshot"),
            "annotations": {
                **_authority_annotations(envelope),
                "aiperf.nvidia.com/content-sha256": actual_digest,
            },
        },
        "immutable": True,
    }
    snapshot.update(content)
    return snapshot


def _workload_metadata(envelope: ControllerEnvelope, object_uid: str) -> dict[str, Any]:
    return {
        "name": workload_name(envelope),
        "namespace": envelope.namespace,
        "ownerReferences": [owner_reference(envelope, object_uid)],
        "labels": {"aiperf.nvidia.com/role": "workload"},
        "annotations": {
            "aiperf.nvidia.com/job-id": envelope.job_id,
            "aiperf.nvidia.com/run-id": envelope.run_id,
            "aiperf.nvidia.com/sha256": envelope_sha256(envelope),
        },
    }


def build_workload_identity(
    envelope: ControllerEnvelope, object_uid: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the per-run ServiceAccount and least-authority RBAC resources."""
    name = workload_name(envelope)
    metadata = _workload_metadata(envelope, object_uid)
    service_account = {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": metadata,
    }
    role = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": metadata,
        "rules": [
            {
                "apiGroups": ["aiperf.nvidia.com"],
                "resources": ["aiperfjobs/status"],
                "resourceNames": [envelope.job_id],
                "verbs": ["patch"],
            },
        ],
    }
    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": metadata,
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": name,
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": name,
                "namespace": envelope.namespace,
            }
        ],
    }
    return service_account, role, binding


def build_jobset(
    envelope: ControllerEnvelope, results_upload_base_url: str, object_uid: str
) -> dict[str, Any]:
    """Build controller+sidecar and indexed cell JobSets from submitted envelope fields only."""
    controller = _role_by_name(envelope, "controller")
    sidecar = _role_by_name(envelope, "results-sidecar")
    cell = _role_by_name(envelope, "cell")
    controller_bootstrap = _role_bootstrap(controller)
    sidecar_bootstrap = _role_bootstrap(sidecar)
    common = {
        "restartPolicy": "Never",
        "serviceAccountName": workload_name(envelope),
        "securityContext": {"runAsUser": 0},
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": envelope.job_id,
            "namespace": envelope.namespace,
            "ownerReferences": [owner_reference(envelope, object_uid)],
            "labels": {
                "aiperf.nvidia.com/run-id": envelope.run_id,
                "aiperf.nvidia.com/role": "jobset",
            },
            "annotations": {"aiperf.nvidia.com/sha256": envelope_sha256(envelope)},
        },
        "spec": {
            "network": {"enableDNSHostnames": True},
            "replicatedJobs": [
                _job(
                    "controller",
                    [
                        _container(
                            controller,
                            envelope,
                            controller_bootstrap,
                            results_upload_base_url,
                            object_uid,
                        ),
                        _container(
                            sidecar,
                            envelope,
                            sidecar_bootstrap,
                            results_upload_base_url,
                            object_uid,
                        ),
                    ],
                    _volumes(
                        [controller_bootstrap, sidecar_bootstrap],
                        envelope,
                        object_uid,
                        has_results=True,
                    ),
                    {
                        **common,
                        "automountServiceAccountToken": False,
                    },
                ),
                *[
                    _job(
                        f"cell-{bootstrap.cell_id}",
                        [
                            _container(
                                cell,
                                envelope,
                                bootstrap,
                                results_upload_base_url,
                                object_uid,
                                bootstrap.cell_id,
                            )
                        ],
                        _volumes([bootstrap], envelope, object_uid),
                        {**common, "automountServiceAccountToken": False},
                    )
                    for bootstrap in envelope.cell_bootstraps
                ],
            ],
        },
    }


def envelope_sha256(envelope: ControllerEnvelope) -> str:
    """Return the stable digest of the typed envelope's canonical JSON form."""
    canonical = json.dumps(
        envelope.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _contains(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def validate_jobset_identity(desired: dict[str, Any], existing: dict[str, Any]) -> None:
    """Accept an existing JobSet only when its complete desired projection matches."""
    actual_metadata = existing.get("metadata")
    desired_metadata = desired["metadata"]
    if (
        not isinstance(actual_metadata, dict)
        or existing.get("apiVersion") != desired.get("apiVersion")
        or existing.get("kind") != desired.get("kind")
        or actual_metadata.get("name") != desired_metadata.get("name")
        or actual_metadata.get("namespace") != desired_metadata.get("namespace")
        or actual_metadata.get("labels") != desired_metadata.get("labels")
        or actual_metadata.get("annotations") != desired_metadata.get("annotations")
        or actual_metadata.get("ownerReferences")
        != desired_metadata.get("ownerReferences")
        or existing.get("spec") != desired.get("spec")
    ):
        raise ValueError("existing JobSet does not match submitted envelope")


def _job(
    name: str,
    containers: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    pod_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "groupName": "default",
        "replicas": 1,
        "template": {
            "spec": {
                "template": {
                    "spec": {**pod_spec, "containers": containers, "volumes": volumes}
                }
            }
        },
    }


def validate_references(
    envelope: ControllerEnvelope,
    metadata_by_name: dict[str, dict[str, Any]],
    object_uid: str | None = None,
) -> None:
    """Validate only supplied Secret metadata; Secret `.data` is never accessed."""
    for role in envelope.roles:
        if role.name == "cell" or role.bootstrap is None:
            continue
        metadata = metadata_by_name.get(role.bootstrap.secret_name)
        if metadata is None:
            raise ValueError(f"bootstrap Secret metadata missing for {role.name}")
        validate_bootstrap_metadata(role.bootstrap, metadata)
        if object_uid is not None and metadata.get("metadata", {}).get(
            "ownerReferences"
        ) != [owner_reference(envelope, object_uid)]:
            raise ValueError(
                "bootstrap Secret owner reference does not match AIPerfJob"
            )
        if (
            metadata.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/run-id")
            != envelope.run_id
        ):
            raise ValueError("bootstrap Secret run-id label does not match envelope")
    for bootstrap in envelope.cell_bootstraps:
        metadata = metadata_by_name.get(bootstrap.secret_name)
        if metadata is None:
            raise ValueError(
                f"bootstrap Secret metadata missing for cell {bootstrap.cell_id}"
            )
        validate_bootstrap_metadata(bootstrap, metadata)
        if object_uid is not None and metadata.get("metadata", {}).get(
            "ownerReferences"
        ) != [owner_reference(envelope, object_uid)]:
            raise ValueError(
                "bootstrap Secret owner reference does not match AIPerfJob"
            )
        if (
            metadata.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/run-id")
            != envelope.run_id
        ):
            raise ValueError("bootstrap Secret run-id label does not match envelope")


def submitted_status(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Return the initial recognized lifecycle status without bootstrap content."""
    return {"phase": "Pending", "runId": envelope.run_id, "jobSet": envelope.job_id}
