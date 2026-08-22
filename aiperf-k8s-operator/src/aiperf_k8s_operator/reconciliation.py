# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile immutable native envelopes into the three-role JobSet topology."""

from __future__ import annotations

from typing import Any

from .contract import ControllerEnvelope, RoleEnvelope, validate_bootstrap_metadata


def _container(role: RoleEnvelope, envelope: ControllerEnvelope) -> dict[str, Any]:
    """Project one immutable role without interpreting its command, argv, or image."""
    return {
        "name": role.name,
        "image": envelope.image_digest,
        "command": role.command,
        "args": role.argv,
        "env": [{"name": key, "value": value} for key, value in sorted(role.environment.items())],
        "volumeMounts": [
            {"name": f"bootstrap-{role.name}", "mountPath": role.bootstrap.mount_path, "readOnly": True},
            {"name": "config", "mountPath": "/etc/aiperf/config", "readOnly": True},
        ],
    }


def _role_by_name(envelope: ControllerEnvelope, name: str) -> RoleEnvelope:
    return next(role for role in envelope.roles if role.name == name)


def _volumes(roles: list[RoleEnvelope], envelope: ControllerEnvelope) -> list[dict[str, Any]]:
    return [
        *[
            {"name": f"bootstrap-{role.name}", "secret": {"secretName": role.bootstrap.secret_name}}
            for role in roles
        ],
        {"name": "config", "configMap": {"name": envelope.config_ref.name}},
    ]


def build_jobset(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Build controller+sidecar and indexed cell JobSets from submitted envelope fields only."""
    controller = _role_by_name(envelope, "controller")
    sidecar = _role_by_name(envelope, "results-sidecar")
    cell = _role_by_name(envelope, "cell")
    common = {
        "restartPolicy": "Never",
        "serviceAccountName": "aiperf-workload",
        "enableDNSHostnames": True,
    }
    return {
        "apiVersion": "jobset.x-k8s.io/v1alpha2",
        "kind": "JobSet",
        "metadata": {
            "name": envelope.job_id,
            "namespace": envelope.namespace,
            "labels": {"aiperf.nvidia.com/run-id": envelope.run_id},
        },
        "spec": {
            "replicatedJobs": [
                {
                    "name": "controller",
                    "replicas": 1,
                    "template": {
                        "spec": {
                            **common,
                            "containers": [_container(controller, envelope), _container(sidecar, envelope)],
                            "volumes": _volumes([controller, sidecar], envelope),
                        }
                    },
                },
                {
                    "name": "cell",
                    "replicas": envelope.cells,
                    "template": {
                        "spec": {
                            **common,
                            "containers": [_container(cell, envelope)],
                            "volumes": _volumes([cell], envelope),
                        }
                    },
                },
            ]
        },
    }


def validate_references(envelope: ControllerEnvelope, metadata_by_name: dict[str, dict[str, Any]]) -> None:
    """Validate only supplied Secret metadata; Secret `.data` is never accessed."""
    for role in envelope.roles:
        metadata = metadata_by_name.get(role.bootstrap.secret_name)
        if metadata is None:
            raise ValueError(f"bootstrap Secret metadata missing for {role.name}")
        validate_bootstrap_metadata(role.bootstrap, metadata)


def submitted_status(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Return the initial recognized lifecycle status without bootstrap content."""
    return {"phase": "Pending", "runId": envelope.run_id, "jobSet": envelope.job_id}
