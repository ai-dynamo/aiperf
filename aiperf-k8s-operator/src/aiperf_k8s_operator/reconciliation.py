# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile immutable native envelopes into JobSet resources."""

from __future__ import annotations

from typing import Any

from .contract import ControllerEnvelope, RoleEnvelope, validate_bootstrap_metadata


def _container(role: RoleEnvelope, envelope: ControllerEnvelope) -> dict[str, Any]:
    """Project one immutable role without interpreting its argv or image."""
    return {
        "name": role.name,
        "image": envelope.image_digest,
        "command": [role.command],
        "args": role.argv,
        "env": [
            {"name": key, "value": value}
            for key, value in sorted(role.environment.items())
        ],
        "volumeMounts": [
            {
                "name": "bootstrap",
                "mountPath": role.bootstrap.mount_path,
                "readOnly": True,
            }
        ],
    }


def _replicated_job(role: RoleEnvelope, envelope: ControllerEnvelope) -> dict[str, Any]:
    replicas = envelope.cells if role.name == "cell" else 1
    return {
        "name": role.name,
        "replicas": replicas,
        "template": {
            "spec": {
                "restartPolicy": "Never",
                "serviceAccountName": "aiperf-workload",
                "containers": [_container(role, envelope)],
                "volumes": [
                    {
                        "name": "bootstrap",
                        "secret": {"secretName": role.bootstrap.secret_name},
                    }
                ],
            }
        },
    }


def build_jobset(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Build the three-role JobSet from immutable client-submitted data only."""
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
                _replicated_job(role, envelope) for role in envelope.roles
            ]
        },
    }


def validate_references(
    envelope: ControllerEnvelope, metadata_by_name: dict[str, dict[str, Any]]
) -> None:
    """Validate only caller-supplied Secret metadata; Secret data is never accessed."""
    for role in envelope.roles:
        metadata = metadata_by_name.get(role.bootstrap.secret_name)
        if metadata is None:
            raise ValueError(f"bootstrap Secret metadata missing for {role.name}")
        validate_bootstrap_metadata(role.bootstrap, metadata)


def submitted_status(envelope: ControllerEnvelope) -> dict[str, Any]:
    """Return a monotonic initial status without duplicating bootstrap bytes."""
    return {"phase": "Submitted", "runId": envelope.run_id, "jobSet": envelope.job_id}
