# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile immutable native envelopes into the three-role JobSet topology."""

from __future__ import annotations

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
    cell_id: int | None = None,
) -> dict[str, Any]:
    """Project one immutable role without interpreting its command, argv, or image."""
    environment: dict[str, Any] = dict(role.environment)
    environment["AIPERF_CELL_LAUNCHER"] = "k8s"
    environment["AIPERF_ROLE_BOOTSTRAP_FILE"] = bootstrap.mount_path
    if role.name == "controller":
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
    return {
        "name": role.name,
        "image": envelope.image_digest,
        "command": role.command,
        "args": role.argv,
        "env": [
            {"name": key, **({"value": value} if isinstance(value, str) else value)}
            for key, value in sorted(environment.items())
        ],
        "volumeMounts": [
            {
                "name": f"bootstrap-{bootstrap_name(bootstrap)}",
                "mountPath": bootstrap.mount_path,
                "subPath": "bootstrap",
                "readOnly": True,
            },
            {"name": "config", "mountPath": "/etc/aiperf/config", "readOnly": True},
        ],
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
) -> list[dict[str, Any]]:
    return [
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
        {"name": "config", "configMap": {"name": envelope.config_ref.name}},
    ]


def bootstrap_name(bootstrap: BootstrapReference | CellBootstrapReference) -> str:
    if isinstance(bootstrap, CellBootstrapReference):
        return f"cell-{bootstrap.cell_id}"
    return bootstrap.role


def build_jobset(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Build controller+sidecar and indexed cell JobSets from submitted envelope fields only."""
    controller = _role_by_name(envelope, "controller")
    sidecar = _role_by_name(envelope, "results-sidecar")
    cell = _role_by_name(envelope, "cell")
    controller_bootstrap = _role_bootstrap(controller)
    sidecar_bootstrap = _role_bootstrap(sidecar)
    common = {
        "restartPolicy": "Never",
        "serviceAccountName": "aiperf-workload",
        "securityContext": {"runAsUser": 0},
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": envelope.job_id,
            "namespace": envelope.namespace,
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
                        _container(controller, envelope, controller_bootstrap),
                        _container(sidecar, envelope, sidecar_bootstrap),
                    ],
                    _volumes([controller_bootstrap, sidecar_bootstrap], envelope),
                    common,
                ),
                *[
                    _job(
                        f"cell-{bootstrap.cell_id}",
                        [_container(cell, envelope, bootstrap, bootstrap.cell_id)],
                        _volumes([bootstrap], envelope),
                        common,
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


def validate_jobset_identity(
    envelope: ControllerEnvelope, jobset: dict[str, Any]
) -> None:
    """Accept an existing JobSet only when its immutable run identity matches."""
    metadata = jobset.get("metadata", {})
    labels = metadata.get("labels", {})
    annotations = metadata.get("annotations", {})
    if (
        metadata.get("name") != envelope.job_id
        or metadata.get("namespace") != envelope.namespace
        or labels.get("aiperf.nvidia.com/run-id") != envelope.run_id
        or labels.get("aiperf.nvidia.com/role") != "jobset"
        or annotations.get("aiperf.nvidia.com/sha256") != envelope_sha256(envelope)
    ):
        raise ValueError("existing JobSet identity does not match submitted envelope")


def _job(
    name: str,
    containers: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    pod_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
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
    envelope: ControllerEnvelope, metadata_by_name: dict[str, dict[str, Any]]
) -> None:
    """Validate only supplied Secret metadata; Secret `.data` is never accessed."""
    for role in envelope.roles:
        if role.name == "cell" or role.bootstrap is None:
            continue
        metadata = metadata_by_name.get(role.bootstrap.secret_name)
        if metadata is None:
            raise ValueError(f"bootstrap Secret metadata missing for {role.name}")
        validate_bootstrap_metadata(role.bootstrap, metadata)
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
