# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict native-k8s/v1 contract loading and metadata-only validation."""

from __future__ import annotations

import json
from importlib import resources
from ipaddress import IPv6Address
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "native-k8s/v1"
_ROLES = frozenset({"controller", "cell", "results-sidecar"})
_CONTRACT_PACKAGE = "aiperf_k8s_operator.contracts.v1"
_DNS_LABEL = r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
_ARTIFACT_ROOT = r"^/results(?:/[A-Za-z0-9_-][A-Za-z0-9._-]{0,62})*$"


def _is_valid_controller_coordinate(address: str) -> bool:
    """Accept one unambiguous TCP host/port coordinate, including bracketed IPv6."""
    coordinate = address.removeprefix("tcp://")
    if "://" in coordinate:
        return False
    if coordinate.startswith("["):
        host, separator, port = coordinate[1:].partition("]:")
        if not separator:
            return False
        try:
            IPv6Address(host)
        except ValueError:
            return False
    else:
        host, separator, port = coordinate.rpartition(":")
        if (
            not separator
            or not host
            or any(character in host for character in "/:[]")
            or not host.isascii()
            or not host.isprintable()
            or any(character.isspace() for character in host)
        ):
            return False
    return port.isascii() and port.isdecimal() and 0 < int(port) < 65536


class ConfigReference(BaseModel):
    """Reference to immutable benchmark configuration material."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(pattern=_DNS_LABEL)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BootstrapReference(BaseModel):
    """Reference-only bootstrap material; no Secret bytes cross this boundary."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    secret_name: str = Field(alias="secretName", min_length=1)
    role: Literal["controller", "cell"]
    mount_path: str = Field(alias="mountPath", pattern=r"^/")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CellBootstrapReference(BaseModel):
    """Reference-only bootstrap material for one numbered cellular worker."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cell_id: int = Field(alias="cellId", ge=0)
    secret_name: str = Field(alias="secretName", min_length=1)
    role: Literal["cell"] = "cell"
    mount_path: str = Field(alias="mountPath", pattern=r"^/")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RoleEnvelope(BaseModel):
    """One executable role from the immutable controller envelope."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: Literal["controller", "cell", "results-sidecar"]
    command: list[str] = Field(min_length=1)
    argv: list[str]
    environment: dict[str, str]
    bootstrap: BootstrapReference | None = None

    @model_validator(mode="after")
    def bootstrap_matches_role(self) -> RoleEnvelope:
        """Reject a reference that could mount one role's bootstrap into another."""
        if self.name != "controller":
            if self.bootstrap is not None:
                raise ValueError(f"{self.name} must not carry a role bootstrap")
            return self
        if self.bootstrap is None:
            raise ValueError("controller role requires bootstrap")
        if self.bootstrap.role != self.name:
            raise ValueError("bootstrap.role must equal role name")
        return self


class ControllerEnvelope(BaseModel):
    """The native v1 workload definition accepted by the independent operator."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contract_version: Literal[CONTRACT_VERSION] = Field(alias="contractVersion")
    run_id: str = Field(alias="runId", pattern=_DNS_LABEL)
    namespace: str = Field(pattern=_DNS_LABEL)
    job_id: str = Field(alias="jobId", pattern=_DNS_LABEL)
    image_digest: str = Field(alias="imageDigest", pattern=r"^sha256:[0-9a-f]{64}$")
    image_reference: str = Field(
        alias="imageReference",
        pattern=(
            r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?"
            r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
            r"@sha256:[0-9a-f]{64}$"
        ),
    )
    cells: int = Field(ge=1)
    artifact_root: str = Field(
        alias="artifactRoot", pattern=_ARTIFACT_ROOT, max_length=1024
    )
    config_ref: ConfigReference = Field(alias="configRef")
    controller_address: str = Field(alias="controllerAddress", min_length=1)
    roles: list[RoleEnvelope]
    cell_bootstraps: list[CellBootstrapReference] = Field(alias="cellBootstraps")

    @model_validator(mode="after")
    def requires_exact_v1_roles(self) -> ControllerEnvelope:
        """Make aggregator/hierarchical roles impossible in the v1 operator."""
        reference_digest = self.image_reference.rsplit("@", maxsplit=1)[-1]
        if reference_digest != self.image_digest:
            raise ValueError("imageReference digest must equal imageDigest")
        if not _is_valid_controller_coordinate(self.controller_address):
            raise ValueError(
                "controllerAddress must be tcp://HOST:PORT or tcp://[IPv6]:PORT"
            )
        names = {role.name for role in self.roles}
        if names != _ROLES or len(self.roles) != len(_ROLES):
            raise ValueError(
                "native-k8s/v1 requires exactly controller, cell, and results-sidecar roles"
            )
        if len(self.cell_bootstraps) != self.cells or any(
            bootstrap.cell_id != cell_id
            for cell_id, bootstrap in enumerate(self.cell_bootstraps)
        ):
            raise ValueError("cellBootstraps must contain each cell id exactly once")
        secret_names = [
            *(
                role.bootstrap.secret_name
                for role in self.roles
                if role.bootstrap is not None
            ),
            *(bootstrap.secret_name for bootstrap in self.cell_bootstraps),
        ]
        if len(secret_names) != len(set(secret_names)):
            raise ValueError("bootstrap Secret names must be unique")
        return self


def _schema(name: str) -> dict[str, Any]:
    with (
        resources.files(_CONTRACT_PACKAGE)
        .joinpath(name)
        .open(encoding="utf-8") as source
    ):
        return json.load(source)


def validate_envelope(payload: dict[str, Any]) -> ControllerEnvelope:
    """Validate caller JSON against its checked-in schema and strict local model."""
    errors = sorted(
        Draft202012Validator(_schema("controller-envelope.schema.json")).iter_errors(
            payload
        ),
        key=str,
    )
    if errors:
        raise ValueError(errors[0].message)
    return ControllerEnvelope.model_validate(payload)


def validate_bootstrap_metadata(
    reference: BootstrapReference, metadata: dict[str, Any]
) -> None:
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
