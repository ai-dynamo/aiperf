# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict native-k8s/v1 contract loading and metadata-only validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "native-k8s/v1"
_ROLES = frozenset({"controller", "cell", "results-sidecar"})
_CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "contracts" / "native-k8s" / "v1"


class ConfigReference(BaseModel):
    """Reference to immutable benchmark configuration material."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(min_length=1)


class BootstrapReference(BaseModel):
    """Reference-only bootstrap material; no Secret bytes cross this boundary."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    secret_name: str = Field(alias="secretName", min_length=1)
    role: Literal["controller", "cell", "results-sidecar"]
    mount_path: str = Field(alias="mountPath", pattern=r"^/")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RoleEnvelope(BaseModel):
    """One executable role from the immutable controller envelope."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Literal["controller", "cell", "results-sidecar"]
    command: list[str] = Field(min_length=1)
    argv: list[str]
    environment: dict[str, str]
    bootstrap: BootstrapReference

    @model_validator(mode="after")
    def bootstrap_matches_role(self) -> RoleEnvelope:
        """Reject a reference that could mount one role's bootstrap into another."""
        if self.bootstrap.role != self.name:
            raise ValueError("bootstrap.role must equal role name")
        return self


class ControllerEnvelope(BaseModel):
    """The native v1 workload definition accepted by the independent operator."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contract_version: Literal[CONTRACT_VERSION] = Field(alias="contractVersion")
    run_id: str = Field(alias="runId", min_length=1)
    namespace: str = Field(min_length=1)
    job_id: str = Field(alias="jobId", min_length=1)
    image_digest: str = Field(alias="imageDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    cells: int = Field(ge=1)
    artifact_root: str = Field(alias="artifactRoot", min_length=1)
    config_ref: ConfigReference = Field(alias="configRef")
    controller_address: str = Field(alias="controllerAddress", min_length=1)
    roles: list[RoleEnvelope]

    @model_validator(mode="after")
    def requires_exact_v1_roles(self) -> ControllerEnvelope:
        """Make aggregator/hierarchical roles impossible in the v1 operator."""
        names = {role.name for role in self.roles}
        if names != _ROLES or len(self.roles) != len(_ROLES):
            raise ValueError("native-k8s/v1 requires exactly controller, cell, and results-sidecar roles")
        return self


def _schema(name: str) -> dict[str, Any]:
    with (_CONTRACT_ROOT / name).open(encoding="utf-8") as source:
        return json.load(source)


def validate_envelope(payload: dict[str, Any]) -> ControllerEnvelope:
    """Validate caller JSON against its checked-in schema and strict local model."""
    errors = sorted(Draft202012Validator(_schema("controller-envelope.schema.json")).iter_errors(payload), key=str)
    if errors:
        raise ValueError(errors[0].message)
    return ControllerEnvelope.model_validate(payload)


def validate_bootstrap_metadata(reference: BootstrapReference, metadata: dict[str, Any]) -> None:
    """Validate supplied Secret metadata without reading, listing, hashing, or logging `.data`."""
    if metadata.get("immutable") is not True:
        raise ValueError("bootstrap Secret must be immutable")
    object_metadata = metadata.get("metadata", {})
    if object_metadata.get("name") != reference.secret_name:
        raise ValueError("bootstrap Secret name does not match envelope")
    labels = object_metadata.get("labels", {})
    annotations = object_metadata.get("annotations", {})
    if labels.get("aiperf.nvidia.com/role") != reference.role:
        raise ValueError("bootstrap Secret role label does not match envelope")
    if annotations.get("aiperf.nvidia.com/sha256") != reference.sha256:
        raise ValueError("bootstrap Secret digest annotation does not match envelope")
