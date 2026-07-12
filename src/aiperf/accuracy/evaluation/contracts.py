# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict Python mirror of evaluator-worker protocol v2.

The Rust DTOs in ``aiperf-accuracy::provider_protocol`` are the wire authority.
This module deliberately contains logical service and semantic-operation names,
but no upstream URL, credential, HTTP method, or caller-selected forwarding
target.  Unknown fields and non-canonical values fail closed at the worker
boundary.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, TypeAlias

from aiperf.accuracy.evaluation.canonical import (
    CANONICAL_JSON_CODEC,
    canonical_dumps,
    canonical_sha256,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_OPEN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE_RE = re.compile(
    r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+(?:;[ -~]+)?$"
)
_FORBIDDEN_AUTHORITY_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "base_url",
        "bearer_token",
        "credential",
        "credentials",
        "endpoint",
        "endpoint_url",
        "headers",
        "host",
        "http_method",
        "proxy_url",
        "secret",
        "token",
        "upstream",
        "url",
    }
)


def strict_object(
    value: Any,
    *,
    field_name: str,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a mapping only when it has the exact permitted field image."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{field_name} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{field_name} is missing field(s): {', '.join(missing)}")
    return value


def require_open_id(value: Any, field_name: str) -> str:
    """Validate a protocol-extensible identifier."""
    if not isinstance(value, str) or not _OPEN_ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} must match {_OPEN_ID_RE.pattern!r}")
    return value


def require_opaque_id(value: Any, field_name: str) -> str:
    """Validate a bounded opaque correlation identifier."""
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or value != value.strip()
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise ValueError(f"{field_name} must be a bounded opaque identifier")
    canonical_dumps(value)
    return value


def require_string(value: Any, field_name: str, *, max_bytes: int = 4096) -> str:
    """Validate one non-empty bounded canonical string."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    canonical_dumps(value)
    return value


def require_sha256(value: Any, field_name: str) -> str:
    """Validate a lowercase SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def require_non_negative_int(value: Any, field_name: str) -> int:
    """Validate an unsigned JSON integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    """Validate a positive JSON integer while rejecting booleans."""
    result = require_non_negative_int(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return result


def require_finite(value: Any, field_name: str) -> float:
    """Validate and normalize one finite public numeric value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


class ArtifactVisibility(StrEnum):
    """Provider artifact visibility request; Rust remains report authority."""

    RESTRICTED = "restricted"
    PUBLIC_PROJECTION = "public_projection"


class ExecutionGranularity(StrEnum):
    """Evaluator-native schedulable unit granularity."""

    CASE = "case"
    HOST_BATCH = "host_batch"


class SchedulingMode(StrEnum):
    """Who instantiates concrete evaluation occurrences."""

    FINITE = "finite"
    RUST_OCCURRENCES = "rust_occurrences"


class ResponseMode(StrEnum):
    """Requested host response delivery."""

    TERMINAL = "terminal"
    STREAMING = "streaming"


class HostOperationDisposition(StrEnum):
    """Rust terminal classification for one host operation."""

    COMPLETED = "completed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"
    ALREADY_TERMINAL = "already_terminal"


class CaseOutcomeKind(StrEnum):
    """Provider-semantic terminal classification for one case."""

    COMPLETED = "completed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EvaluationWorkerIdentity:
    """Immutable self-description returned by ``hello``."""

    evaluator_protocol: int
    provider_id: str
    distribution_id: str
    package: str
    package_version: str
    provider_source_sha256: str
    worker_source_sha256: str
    dependency_lock_sha256: str
    python_version: str
    launch_nonce: str
    operations: tuple[str, ...]
    oci_digest: str | None = None

    def __post_init__(self) -> None:
        if self.evaluator_protocol != 2:
            raise ValueError("evaluator_protocol must be 2")
        require_open_id(self.provider_id, "provider_id")
        require_open_id(self.distribution_id, "distribution_id")
        for name in ("package", "package_version", "python_version"):
            require_string(getattr(self, name), name, max_bytes=512)
        require_string(self.launch_nonce, "launch_nonce", max_bytes=512)
        if len(self.launch_nonce.encode("utf-8")) < 32:
            raise ValueError("launch_nonce must contain at least 32 bytes")
        for name in (
            "provider_source_sha256",
            "worker_source_sha256",
            "dependency_lock_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.oci_digest is not None:
            if not self.oci_digest.startswith("sha256:"):
                raise ValueError("oci_digest must use sha256")
            require_sha256(self.oci_digest.removeprefix("sha256:"), "oci_digest")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("worker operations contain duplicates")

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "evaluator_protocol": self.evaluator_protocol,
            "provider_id": self.provider_id,
            "distribution_id": self.distribution_id,
            "package": self.package,
            "package_version": self.package_version,
            "provider_source_sha256": self.provider_source_sha256,
            "worker_source_sha256": self.worker_source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "python_version": self.python_version,
            "launch_nonce": self.launch_nonce,
            "operations": list(self.operations),
        }
        if self.oci_digest is not None:
            result["oci_digest"] = self.oci_digest
        return result


@dataclass(frozen=True)
class EvaluationPlanRequest:
    """Factory-validated authored plan request."""

    session_id: str
    provider_id: str
    distribution_id: str
    config_schema_version: int
    config_schema_sha256: str
    provider_config: JsonValue
    reproducible: bool

    @classmethod
    def from_wire(cls, value: Any) -> EvaluationPlanRequest:
        item = strict_object(
            value,
            field_name="evaluation plan request",
            allowed={
                "session_id",
                "provider_id",
                "distribution_id",
                "config_schema_version",
                "config_schema_sha256",
                "provider_config",
                "reproducible",
            },
            required={
                "session_id",
                "provider_id",
                "distribution_id",
                "config_schema_version",
                "config_schema_sha256",
                "provider_config",
                "reproducible",
            },
        )
        if not isinstance(item["reproducible"], bool):
            raise TypeError("reproducible must be a boolean")
        canonical_dumps(item["provider_config"])
        _reject_connection_authority(item["provider_config"])
        return cls(
            session_id=require_opaque_id(item["session_id"], "session_id"),
            provider_id=require_open_id(item["provider_id"], "provider_id"),
            distribution_id=require_open_id(item["distribution_id"], "distribution_id"),
            config_schema_version=require_positive_int(
                item["config_schema_version"], "config_schema_version"
            ),
            config_schema_sha256=require_sha256(
                item["config_schema_sha256"], "config_schema_sha256"
            ),
            provider_config=item["provider_config"],
            reproducible=item["reproducible"],
        )


@dataclass(frozen=True)
class AssetRequirement:
    """Provider-declared immutable asset requirement."""

    asset_id: str
    source_kind: str
    immutable_revision: str
    content_sha256: str
    media_type: str
    visibility: ArtifactVisibility = ArtifactVisibility.RESTRICTED

    def __post_init__(self) -> None:
        require_string(self.asset_id, "asset_id", max_bytes=256)
        require_string(self.source_kind, "source_kind", max_bytes=256)
        require_string(self.immutable_revision, "immutable_revision", max_bytes=512)
        require_sha256(self.content_sha256, "content_sha256")
        _require_media_type(self.media_type)

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "asset_id": self.asset_id,
            "source_kind": self.source_kind,
            "immutable_revision": self.immutable_revision,
            "content_sha256": self.content_sha256,
            "media_type": self.media_type,
            "visibility": self.visibility.value,
        }


@dataclass(frozen=True)
class ResolvedAsset:
    """Rust-resolved read-only asset binding."""

    asset_id: str
    contained_path: str
    content_sha256: str
    immutable_revision: str
    media_type: str

    @classmethod
    def from_wire(cls, value: Any) -> ResolvedAsset:
        item = strict_object(
            value,
            field_name="resolved asset",
            allowed={
                "asset_id",
                "contained_path",
                "content_sha256",
                "immutable_revision",
                "media_type",
            },
            required={
                "asset_id",
                "contained_path",
                "content_sha256",
                "immutable_revision",
                "media_type",
            },
        )
        path = require_string(item["contained_path"], "contained_path")
        if "\x00" in path or not PurePosixPath(path).is_absolute():
            raise ValueError("contained_path must be an absolute contained worker path")
        return cls(
            asset_id=require_string(item["asset_id"], "asset_id", max_bytes=256),
            contained_path=path,
            content_sha256=require_sha256(item["content_sha256"], "content_sha256"),
            immutable_revision=require_string(
                item["immutable_revision"], "immutable_revision", max_bytes=512
            ),
            media_type=_require_media_type(item["media_type"]),
        )


@dataclass(frozen=True)
class HostCapabilityRequirement:
    """One typed Rust-hosted capability required by a plan."""

    capability_id: str
    schema_sha256: str
    required: bool = True

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "capability_id": require_open_id(self.capability_id, "capability_id"),
            "schema_sha256": require_sha256(self.schema_sha256, "schema_sha256"),
            "required": self.required,
        }


@dataclass(frozen=True)
class LogicalServiceRequirement:
    """Operations required on one logical Rust-owned model service."""

    service_id: str
    purpose: str
    operations: tuple[str, ...]
    allows_restricted_payload: bool = False

    def __post_init__(self) -> None:
        require_open_id(self.service_id, "service_id")
        require_open_id(self.purpose, "purpose")
        if not self.operations or len(set(self.operations)) != len(self.operations):
            raise ValueError("logical service operations must be non-empty and unique")
        for operation in self.operations:
            require_open_id(operation, "semantic_operation_id")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "service_id": self.service_id,
            "purpose": self.purpose,
            "operations": list(self.operations),
            "allows_restricted_payload": self.allows_restricted_payload,
        }


@dataclass(frozen=True)
class EvaluationQueueCredits:
    """Provider-proposed hard worker queue bounds."""

    units: int
    host_operations: int
    host_operations_per_unit: int
    stream_events: int
    sandboxes: int
    processes: int
    artifacts: int
    artifact_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "units",
            "host_operations",
            "host_operations_per_unit",
            "stream_events",
            "artifacts",
            "artifact_bytes",
        ):
            require_non_negative_int(getattr(self, name), name)
        require_non_negative_int(self.sandboxes, "sandboxes")
        require_non_negative_int(self.processes, "processes")
        if self.host_operations_per_unit > self.host_operations:
            raise ValueError("per-unit host-operation credit exceeds global credit")

    def validate_plan(self) -> None:
        """Reject zero capacities that are invalid in a proposed plan."""
        for name in (
            "units",
            "host_operations",
            "host_operations_per_unit",
            "stream_events",
            "artifacts",
            "artifact_bytes",
        ):
            require_positive_int(getattr(self, name), name)

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            name: getattr(self, name)
            for name in (
                "units",
                "host_operations",
                "host_operations_per_unit",
                "stream_events",
                "sandboxes",
                "processes",
                "artifacts",
                "artifact_bytes",
            )
        }


@dataclass(frozen=True)
class AggregationPolicy:
    """Frozen provider-owned denominator and reducer semantics."""

    policy_id: str
    exclude_infrastructure: bool
    exclude_cancelled: bool
    definition: JsonValue

    def __post_init__(self) -> None:
        require_string(self.policy_id, "policy_id", max_bytes=256)
        canonical_dumps(self.definition)

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "policy_id": self.policy_id,
            "exclude_infrastructure": self.exclude_infrastructure,
            "exclude_cancelled": self.exclude_cancelled,
            "definition": self.definition,
        }


@dataclass(frozen=True)
class EvaluationPlan:
    """Side-effect-free provider planning result."""

    assets: tuple[AssetRequirement, ...]
    host_requirements: tuple[HostCapabilityRequirement, ...]
    logical_services: tuple[LogicalServiceRequirement, ...]
    aggregation_policy: AggregationPolicy
    execution_granularity: ExecutionGranularity
    scheduling_mode: SchedulingMode
    queue_credits: EvaluationQueueCredits
    max_total_host_operations: int
    max_total_stream_events: int
    finite_unit_count: int | None = None
    finite_case_count: int | None = None

    def __post_init__(self) -> None:
        self.queue_credits.validate_plan()
        require_positive_int(
            self.max_total_host_operations, "max_total_host_operations"
        )
        require_non_negative_int(
            self.max_total_stream_events, "max_total_stream_events"
        )
        if self.queue_credits.host_operations > self.max_total_host_operations:
            raise ValueError(
                "outstanding host-operation credits exceed the total operation envelope"
            )
        if self.scheduling_mode is SchedulingMode.FINITE:
            if self.finite_unit_count is None or self.finite_case_count is None:
                raise ValueError("finite plan requires exact unit/case counts")
            require_positive_int(self.finite_unit_count, "finite_unit_count")
            require_positive_int(self.finite_case_count, "finite_case_count")
        elif self.finite_unit_count is not None or self.finite_case_count is not None:
            raise ValueError("rust_occurrences plan cannot report finite counts")

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "assets": [item.to_wire() for item in self.assets],
            "host_requirements": [item.to_wire() for item in self.host_requirements],
            "logical_services": [item.to_wire() for item in self.logical_services],
            "aggregation_policy": self.aggregation_policy.to_wire(),
            "execution_granularity": self.execution_granularity.value,
            "scheduling_mode": self.scheduling_mode.value,
            "queue_credits": self.queue_credits.to_wire(),
            "max_total_host_operations": self.max_total_host_operations,
            "max_total_stream_events": self.max_total_stream_events,
        }
        if self.finite_unit_count is not None:
            result["finite_unit_count"] = self.finite_unit_count
        if self.finite_case_count is not None:
            result["finite_case_count"] = self.finite_case_count
        return result


@dataclass(frozen=True)
class CaseTemplateDescriptor:
    """Model-safe frozen provider case template."""

    template_id: str
    task: str
    source: str

    def __post_init__(self) -> None:
        require_opaque_id(self.template_id, "template_id")
        require_string(self.task, "task")
        require_string(self.source, "source")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "template_id": self.template_id,
            "task": self.task,
            "source": self.source,
        }


@dataclass(frozen=True)
class ExecutionUnitTemplateDescriptor:
    """Frozen unit template and ordered case-template membership."""

    unit_template_id: str
    case_template_ids: tuple[str, ...]
    granularity: ExecutionGranularity
    scheduling_class: str

    def __post_init__(self) -> None:
        require_opaque_id(self.unit_template_id, "unit_template_id")
        if not self.case_template_ids or len(set(self.case_template_ids)) != len(
            self.case_template_ids
        ):
            raise ValueError("case_template_ids must be non-empty and unique")
        for item in self.case_template_ids:
            require_opaque_id(item, "case_template_id")
        require_string(self.scheduling_class, "scheduling_class")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "unit_template_id": self.unit_template_id,
            "case_template_ids": list(self.case_template_ids),
            "granularity": self.granularity.value,
            "scheduling_class": self.scheduling_class,
        }


@dataclass(frozen=True)
class CaseOccurrenceDescriptor:
    """Concrete occurrence of one provider case template."""

    case_id: str
    template_id: str
    issue_ordinal: int
    phase_id: str
    cycle_index: int

    def __post_init__(self) -> None:
        require_opaque_id(self.case_id, "case_id")
        require_opaque_id(self.template_id, "template_id")
        require_non_negative_int(self.issue_ordinal, "issue_ordinal")
        require_open_id(self.phase_id, "phase_id")
        require_non_negative_int(self.cycle_index, "cycle_index")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "case_id": self.case_id,
            "template_id": self.template_id,
            "issue_ordinal": self.issue_ordinal,
            "phase_id": self.phase_id,
            "cycle_index": self.cycle_index,
        }


@dataclass(frozen=True)
class ExecutionUnitOccurrence:
    """Concrete schedulable unit occurrence."""

    unit_id: str
    unit_template_id: str
    cases: tuple[CaseOccurrenceDescriptor, ...]

    def __post_init__(self) -> None:
        require_opaque_id(self.unit_id, "unit_id")
        require_opaque_id(self.unit_template_id, "unit_template_id")
        if not self.cases or len({case.case_id for case in self.cases}) != len(
            self.cases
        ):
            raise ValueError("unit cases must be non-empty and unique")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "unit_id": self.unit_id,
            "unit_template_id": self.unit_template_id,
            "cases": [case.to_wire() for case in self.cases],
        }


@dataclass(frozen=True)
class UnitOccurrenceRequest:
    """Rust-authored deterministic unit occurrence request."""

    unit_template_id: str
    phase_id: str
    issue_ordinal: int
    cycle_index: int

    @classmethod
    def from_wire(cls, value: Any) -> UnitOccurrenceRequest:
        item = strict_object(
            value,
            field_name="unit occurrence request",
            allowed={"unit_template_id", "phase_id", "issue_ordinal", "cycle_index"},
            required={"unit_template_id", "phase_id", "issue_ordinal", "cycle_index"},
        )
        return cls(
            unit_template_id=require_opaque_id(
                item["unit_template_id"], "unit_template_id"
            ),
            phase_id=require_open_id(item["phase_id"], "phase_id"),
            issue_ordinal=require_non_negative_int(
                item["issue_ordinal"], "issue_ordinal"
            ),
            cycle_index=require_non_negative_int(item["cycle_index"], "cycle_index"),
        )


@dataclass(frozen=True)
class EvaluationUnitPage:
    """Bounded ordered page of finite occurrences."""

    items: tuple[ExecutionUnitOccurrence, ...]
    next_offset: int
    done: bool

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "items": [item.to_wire() for item in self.items],
            "next_offset": self.next_offset,
            "done": self.done,
        }


@dataclass(frozen=True)
class EvaluationIdentityOverlay:
    """One ordered source overlay bound into evaluation identity."""

    overlay_id: str
    artifact_content_sha256: str

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "overlay_id": require_open_id(self.overlay_id, "overlay_id"),
            "artifact_content_sha256": require_sha256(
                self.artifact_content_sha256, "overlay artifact_content_sha256"
            ),
        }


@dataclass(frozen=True)
class EvaluationIdentityComponent:
    """One immutable provider identity component."""

    name: str
    version: str
    source_sha256: str
    source_commit: str | None = None
    base_source_sha256: str | None = None
    overlay_policy: str | None = None
    overlays: tuple[EvaluationIdentityOverlay, ...] = ()

    def __post_init__(self) -> None:
        require_string(self.name, "component.name")
        require_string(self.version, "component.version")
        require_sha256(self.source_sha256, "component.source_sha256")
        if self.overlays:
            if (
                self.source_commit is None
                or self.base_source_sha256 is None
                or self.overlay_policy != "aiperf-unified-diff-overlay-v1"
            ):
                raise ValueError(
                    "overlaid component requires source commit, base source, and overlay policy"
                )
            if len(self.source_commit) not in {40, 64} or any(
                character not in "0123456789abcdef"
                for character in self.source_commit
            ):
                raise ValueError("component source_commit is not lowercase hex")
            require_sha256(
                self.base_source_sha256, "component.base_source_sha256"
            )
            ids = [item.overlay_id for item in self.overlays]
            if len(set(ids)) != len(ids):
                raise ValueError("component overlays contain duplicate IDs")
            for overlay in self.overlays:
                overlay.to_wire()
        elif any(
            value is not None
            for value in (
                self.source_commit,
                self.base_source_sha256,
                self.overlay_policy,
            )
        ):
            raise ValueError("non-overlaid component cannot claim overlay provenance")

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "name": require_string(self.name, "component.name"),
            "version": require_string(self.version, "component.version"),
            "source_sha256": require_sha256(
                self.source_sha256, "component.source_sha256"
            ),
            "overlays": [item.to_wire() for item in self.overlays],
        }
        if self.overlays:
            assert self.source_commit is not None
            assert self.base_source_sha256 is not None
            assert self.overlay_policy is not None
            result.update(
                {
                    "source_commit": self.source_commit,
                    "base_source_sha256": self.base_source_sha256,
                    "overlay_policy": self.overlay_policy,
                }
            )
        return result


@dataclass(frozen=True)
class EvaluationHostIdentity:
    """Rust host/isolation identity bound into a session."""

    runner_sha256: str
    capability_inventory_sha256: str
    schema_inventory_sha256: str
    isolation_proof_sha256: str

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            name: require_sha256(getattr(self, name), name)
            for name in (
                "runner_sha256",
                "capability_inventory_sha256",
                "schema_inventory_sha256",
                "isolation_proof_sha256",
            )
        }


@dataclass(frozen=True)
class EvaluationHostBinding:
    """Rust-authored immutable host and route identity supplied at binding."""

    host: EvaluationHostIdentity
    route_map_sha256: str
    prepared_endpoints_sha256: str
    sandbox_sha256: str | None = None

    @classmethod
    def from_wire(cls, value: Any) -> EvaluationHostBinding:
        item = strict_object(
            value,
            field_name="evaluation host binding",
            allowed={
                "host",
                "route_map_sha256",
                "prepared_endpoints_sha256",
                "sandbox_sha256",
            },
            required={"host", "route_map_sha256", "prepared_endpoints_sha256"},
        )
        host = strict_object(
            item["host"],
            field_name="evaluation host identity",
            allowed={
                "runner_sha256",
                "capability_inventory_sha256",
                "schema_inventory_sha256",
                "isolation_proof_sha256",
            },
            required={
                "runner_sha256",
                "capability_inventory_sha256",
                "schema_inventory_sha256",
                "isolation_proof_sha256",
            },
        )
        sandbox = item.get("sandbox_sha256")
        return cls(
            host=EvaluationHostIdentity(
                runner_sha256=require_sha256(host["runner_sha256"], "runner_sha256"),
                capability_inventory_sha256=require_sha256(
                    host["capability_inventory_sha256"],
                    "capability_inventory_sha256",
                ),
                schema_inventory_sha256=require_sha256(
                    host["schema_inventory_sha256"], "schema_inventory_sha256"
                ),
                isolation_proof_sha256=require_sha256(
                    host["isolation_proof_sha256"], "isolation_proof_sha256"
                ),
            ),
            route_map_sha256=require_sha256(
                item["route_map_sha256"], "route_map_sha256"
            ),
            prepared_endpoints_sha256=require_sha256(
                item["prepared_endpoints_sha256"], "prepared_endpoints_sha256"
            ),
            sandbox_sha256=(
                None if sandbox is None else require_sha256(sandbox, "sandbox_sha256")
            ),
        )


@dataclass(frozen=True)
class EvaluationIdentity:
    """Complete frozen provider/template/host identity graph."""

    worker: EvaluationWorkerIdentity
    config_schema_sha256: str
    resolved_config_sha256: str
    dataset: EvaluationIdentityComponent
    components: tuple[EvaluationIdentityComponent, ...]
    ordered_manifest_sha256: str
    case_templates: tuple[CaseTemplateDescriptor, ...]
    unit_templates: tuple[ExecutionUnitTemplateDescriptor, ...]
    policies: JsonValue
    host: EvaluationHostIdentity
    route_map_sha256: str
    prepared_endpoints_sha256: str
    sandbox_sha256: str | None = None
    canonical_json_codec: str = CANONICAL_JSON_CODEC

    def __post_init__(self) -> None:
        if self.canonical_json_codec != CANONICAL_JSON_CODEC:
            raise ValueError("unsupported canonical JSON codec")
        for name in (
            "config_schema_sha256",
            "resolved_config_sha256",
            "ordered_manifest_sha256",
            "route_map_sha256",
            "prepared_endpoints_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if self.sandbox_sha256 is not None:
            require_sha256(self.sandbox_sha256, "sandbox_sha256")
        case_ids = {item.template_id for item in self.case_templates}
        if len(case_ids) != len(self.case_templates):
            raise ValueError("case templates contain duplicate identities")
        unit_ids = {item.unit_template_id for item in self.unit_templates}
        if len(unit_ids) != len(self.unit_templates):
            raise ValueError("unit templates contain duplicate identities")
        if any(
            case_id not in case_ids
            for unit in self.unit_templates
            for case_id in unit.case_template_ids
        ):
            raise ValueError("unit template references an unknown case template")
        canonical_dumps(self.policies)

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "canonical_json_codec": self.canonical_json_codec,
            "worker": self.worker.to_wire(),
            "config_schema_sha256": self.config_schema_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "dataset": self.dataset.to_wire(),
            "components": [item.to_wire() for item in self.components],
            "ordered_manifest_sha256": self.ordered_manifest_sha256,
            "case_templates": [item.to_wire() for item in self.case_templates],
            "unit_templates": [item.to_wire() for item in self.unit_templates],
            "policies": self.policies,
            "host": self.host.to_wire(),
            "route_map_sha256": self.route_map_sha256,
            "prepared_endpoints_sha256": self.prepared_endpoints_sha256,
        }
        if self.sandbox_sha256 is not None:
            result["sandbox_sha256"] = self.sandbox_sha256
        return result


@dataclass(frozen=True)
class CallContext:
    """Immutable lineage attached to a provider-requested host effect."""

    session_id: str
    unit_id: str
    case_id: str
    semantic_attempt_id: str
    logical_call_id: str

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            name: require_opaque_id(getattr(self, name), name)
            for name in (
                "session_id",
                "unit_id",
                "case_id",
                "semantic_attempt_id",
                "logical_call_id",
            )
        }


@dataclass(frozen=True)
class RestrictedDisclosure:
    """Fail-closed disclosure scope for a restricted auxiliary body."""

    service_id: str
    purpose: str
    allow_content_logging: bool = False
    allow_cache: bool = False
    allow_public_hash: bool = False

    def __post_init__(self) -> None:
        require_open_id(self.service_id, "service_id")
        require_open_id(self.purpose, "purpose")
        if self.allow_content_logging or self.allow_cache or self.allow_public_hash:
            raise ValueError(
                "restricted disclosure cannot allow logging, cache, or public hash"
            )

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "service_id": self.service_id,
            "purpose": self.purpose,
            "allow_content_logging": self.allow_content_logging,
            "allow_cache": self.allow_cache,
            "allow_public_hash": self.allow_public_hash,
        }


@dataclass(frozen=True)
class RestrictedInferencePayload:
    """Sensitive body confined to one declared auxiliary route."""

    body: JsonValue
    disclosure: RestrictedDisclosure

    def to_wire(self) -> dict[str, JsonValue]:
        canonical_dumps(self.body)
        return {"body": self.body, "disclosure": self.disclosure.to_wire()}


@dataclass(frozen=True)
class HostOperationRequest:
    """Typed logical host effect requested by a provider."""

    operation_id: str
    context: CallContext
    service_id: str
    purpose: str
    semantic_operation_id: str
    payload: JsonValue
    response_mode: ResponseMode
    idempotency_key: str
    restricted_payload: RestrictedInferencePayload | None = None
    deadline_ms: int | None = None

    def __post_init__(self) -> None:
        require_opaque_id(self.operation_id, "operation_id")
        require_open_id(self.service_id, "service_id")
        require_open_id(self.purpose, "purpose")
        require_open_id(self.semantic_operation_id, "semantic_operation_id")
        require_string(self.idempotency_key, "idempotency_key", max_bytes=512)
        canonical_dumps(self.payload)
        _reject_connection_authority(self.payload)
        if self.deadline_ms is not None:
            require_non_negative_int(self.deadline_ms, "deadline_ms")
        if self.restricted_payload is not None:
            disclosure = self.restricted_payload.disclosure
            if (
                disclosure.service_id != self.service_id
                or disclosure.purpose != self.purpose
            ):
                raise ValueError(
                    "restricted payload disclosure does not match route/purpose"
                )

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {
            "operation_id": self.operation_id,
            "context": self.context.to_wire(),
            "service_id": self.service_id,
            "purpose": self.purpose,
            "semantic_operation_id": self.semantic_operation_id,
            "payload": self.payload,
            "response_mode": self.response_mode.value,
            "idempotency_key": self.idempotency_key,
        }
        if self.restricted_payload is not None:
            result["restricted_payload"] = self.restricted_payload.to_wire()
        if self.deadline_ms is not None:
            result["deadline_ms"] = self.deadline_ms
        return result


@dataclass(frozen=True)
class HostOperationCancelRequest:
    """Provider request to cancel a queued or active host operation."""

    operation_id: str
    semantic_attempt_id: str
    reason: str

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "operation_id": require_opaque_id(self.operation_id, "operation_id"),
            "semantic_attempt_id": require_opaque_id(
                self.semantic_attempt_id, "semantic_attempt_id"
            ),
            "reason": require_string(self.reason, "reason"),
        }


@dataclass(frozen=True)
class EvaluationError:
    """Safe structured evaluation infrastructure error."""

    stage: str
    error_kind: str
    retryable: bool
    message: str

    def __post_init__(self) -> None:
        require_open_id(self.stage, "error.stage")
        require_open_id(self.error_kind, "error.error_kind")
        require_string(self.message, "error.message")

    @classmethod
    def from_wire(cls, value: Any) -> EvaluationError:
        item = strict_object(
            value,
            field_name="evaluation error",
            allowed={"stage", "error_kind", "retryable", "message"},
            required={"stage", "error_kind", "retryable", "message"},
        )
        if not isinstance(item["retryable"], bool):
            raise TypeError("evaluation error retryable must be a boolean")
        return cls(
            stage=require_open_id(item["stage"], "error.stage"),
            error_kind=require_open_id(item["error_kind"], "error.error_kind"),
            retryable=item["retryable"],
            message=require_string(item["message"], "error.message"),
        )

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "stage": self.stage,
            "error_kind": self.error_kind,
            "retryable": self.retryable,
            "message": self.message,
        }


@dataclass(frozen=True)
class HostOperationUsage:
    """Rust-authoritative normalized token usage."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    @classmethod
    def from_wire(cls, value: Any) -> HostOperationUsage:
        item = strict_object(
            value,
            field_name="host operation usage",
            allowed={
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_tokens",
            },
        )
        values: dict[str, int | None] = {}
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cached_tokens",
        ):
            raw = item.get(name)
            values[name] = None if raw is None else require_non_negative_int(raw, name)
        return cls(**values)

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            name: value
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_tokens",
            )
            if (value := getattr(self, name)) is not None
        }


@dataclass(frozen=True)
class HostOperationTerminal:
    """Exactly one terminal for an emitted host operation."""

    operation_id: str
    semantic_attempt_id: str
    disposition: HostOperationDisposition
    usage: HostOperationUsage
    observed_output: bool
    result: JsonValue | None = None
    error: EvaluationError | None = None

    @classmethod
    def from_wire(cls, value: Any) -> HostOperationTerminal:
        item = strict_object(
            value,
            field_name="host operation terminal",
            allowed={
                "operation_id",
                "semantic_attempt_id",
                "disposition",
                "result",
                "error",
                "usage",
                "observed_output",
            },
            required={
                "operation_id",
                "semantic_attempt_id",
                "disposition",
                "usage",
                "observed_output",
            },
        )
        try:
            disposition = HostOperationDisposition(item["disposition"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid host terminal disposition") from error
        if not isinstance(item["observed_output"], bool):
            raise TypeError("observed_output must be a boolean")
        result = item.get("result")
        if result is not None:
            canonical_dumps(result)
        resolved_error = (
            None
            if item.get("error") is None
            else EvaluationError.from_wire(item["error"])
        )
        if disposition is HostOperationDisposition.COMPLETED:
            if "result" not in item or resolved_error is not None:
                raise ValueError("completed terminal requires result and no error")
        elif disposition in {
            HostOperationDisposition.INFRASTRUCTURE_ERROR,
            HostOperationDisposition.CANCELLED,
        }:
            if result is not None or resolved_error is None:
                raise ValueError("failed/cancelled terminal requires only error")
        elif result is not None or resolved_error is not None:
            raise ValueError("already-terminal acknowledgement has no result/error")
        return cls(
            operation_id=require_opaque_id(item["operation_id"], "operation_id"),
            semantic_attempt_id=require_opaque_id(
                item["semantic_attempt_id"], "semantic_attempt_id"
            ),
            disposition=disposition,
            usage=HostOperationUsage.from_wire(item["usage"]),
            observed_output=item["observed_output"],
            result=result,
            error=resolved_error,
        )


@dataclass(frozen=True)
class HostOperationEvent:
    """Typed Rust-to-provider host event."""

    kind: str
    operation_id: str
    stream_sequence: int | None = None
    delta: JsonValue | None = None
    usage: HostOperationUsage | None = None
    terminal: HostOperationTerminal | None = None
    semantic_attempt_id: str | None = None
    already_terminal: bool | None = None

    @classmethod
    def from_wire(cls, value: Any) -> HostOperationEvent:
        if not isinstance(value, dict):
            raise TypeError("host operation event must be an object")
        kind = value.get("kind")
        if kind == "stream_delta":
            item = strict_object(
                value,
                field_name="stream delta",
                allowed={"kind", "operation_id", "stream_sequence", "delta"},
                required={"kind", "operation_id", "stream_sequence", "delta"},
            )
            canonical_dumps(item["delta"])
            return cls(
                kind=kind,
                operation_id=require_opaque_id(item["operation_id"], "operation_id"),
                stream_sequence=require_non_negative_int(
                    item["stream_sequence"], "stream_sequence"
                ),
                delta=item["delta"],
            )
        if kind == "usage":
            item = strict_object(
                value,
                field_name="usage event",
                allowed={"kind", "operation_id", "usage"},
                required={"kind", "operation_id", "usage"},
            )
            return cls(
                kind=kind,
                operation_id=require_opaque_id(item["operation_id"], "operation_id"),
                usage=HostOperationUsage.from_wire(item["usage"]),
            )
        if kind == "terminal":
            item = strict_object(
                value,
                field_name="terminal event",
                allowed={"kind", "terminal"},
                required={"kind", "terminal"},
            )
            terminal = HostOperationTerminal.from_wire(item["terminal"])
            return cls(kind=kind, operation_id=terminal.operation_id, terminal=terminal)
        if kind == "cancellation_acknowledged":
            item = strict_object(
                value,
                field_name="cancellation acknowledgement",
                allowed={
                    "kind",
                    "operation_id",
                    "semantic_attempt_id",
                    "already_terminal",
                },
                required={
                    "kind",
                    "operation_id",
                    "semantic_attempt_id",
                    "already_terminal",
                },
            )
            if not isinstance(item["already_terminal"], bool):
                raise TypeError("already_terminal must be a boolean")
            return cls(
                kind=kind,
                operation_id=require_opaque_id(item["operation_id"], "operation_id"),
                semantic_attempt_id=require_opaque_id(
                    item["semantic_attempt_id"], "semantic_attempt_id"
                ),
                already_terminal=item["already_terminal"],
            )
        raise ValueError(f"unknown host operation event kind {kind!r}")


@dataclass(frozen=True)
class ProviderScore:
    """Complete provider-native score plus reviewed optional public projection."""

    value: JsonValue
    public_projection: JsonValue | None = None

    def __post_init__(self) -> None:
        canonical_dumps(self.value)
        if self.public_projection is not None:
            canonical_dumps(self.public_projection)

    def to_wire(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {"value": self.value}
        if self.public_projection is not None:
            result["public_projection"] = self.public_projection
        return result


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a provider-declared artifact."""

    artifact_id: str
    path: str
    visibility: ArtifactVisibility = ArtifactVisibility.RESTRICTED

    def __post_init__(self) -> None:
        require_open_id(self.artifact_id, "artifact_id")
        _validate_relative_path(self.path, "artifact.path")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "visibility": self.visibility.value,
        }


@dataclass(frozen=True)
class CaseOutcome:
    """Exactly one provider-semantic terminal for a concrete case."""

    case_id: str
    kind: CaseOutcomeKind
    scores: dict[str, ProviderScore] = field(default_factory=dict)
    numeric_metrics: dict[str, float] = field(default_factory=dict)
    primary_score: str | None = None
    annotations: JsonValue | None = None
    error: EvaluationError | None = None
    cancellation_stage: str | None = None
    cancellation_reason: str | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        require_opaque_id(self.case_id, "case_id")
        if self.kind is CaseOutcomeKind.COMPLETED:
            if (
                not self.scores
                or self.error
                or self.cancellation_stage
                or self.cancellation_reason
            ):
                raise ValueError(
                    "completed case requires scores and no error/cancellation"
                )
            if self.primary_score is not None and self.primary_score not in self.scores:
                raise ValueError("primary_score is absent from scores")
            for name in (*self.scores, *self.numeric_metrics):
                require_open_id(name, "score name")
            for name, number in self.numeric_metrics.items():
                require_finite(number, f"numeric_metrics.{name}")
            if self.annotations is not None:
                canonical_dumps(self.annotations)
        elif self.kind is CaseOutcomeKind.INFRASTRUCTURE_ERROR:
            if self.error is None or self.scores or self.numeric_metrics:
                raise ValueError("infrastructure case requires only an error")
        else:
            if (
                self.cancellation_stage is None
                or self.cancellation_reason is None
                or self.error is not None
                or self.scores
                or self.numeric_metrics
            ):
                raise ValueError("cancelled case requires stage/reason only")
            require_open_id(self.cancellation_stage, "cancellation_stage")
            require_string(self.cancellation_reason, "cancellation_reason")

    def to_wire(self) -> dict[str, JsonValue]:
        if self.kind is CaseOutcomeKind.COMPLETED:
            completed: dict[str, JsonValue] = {
                "scores": {
                    name: score.to_wire() for name, score in self.scores.items()
                },
                "numeric_metrics": {
                    name: require_finite(value, name)
                    for name, value in self.numeric_metrics.items()
                },
            }
            if self.primary_score is not None:
                completed["primary_score"] = self.primary_score
            if self.annotations is not None:
                completed["annotations"] = self.annotations
            outcome: dict[str, JsonValue] = {
                "kind": "completed",
                "completed": completed,
            }
        elif self.kind is CaseOutcomeKind.INFRASTRUCTURE_ERROR:
            assert self.error is not None
            outcome = {"kind": "infrastructure_error", "error": self.error.to_wire()}
        else:
            assert self.cancellation_stage is not None
            assert self.cancellation_reason is not None
            outcome = {
                "kind": "cancelled",
                "stage": self.cancellation_stage,
                "reason": self.cancellation_reason,
            }
        return {
            "case_id": self.case_id,
            "outcome": outcome,
            "artifact_refs": [item.to_wire() for item in self.artifact_refs],
        }


@dataclass(frozen=True)
class AggregateMetric:
    """Provider-authored aggregate with a finite public value."""

    scorer: str
    reducer: str
    metric: str
    value: float
    scored_count: int
    unscored_count: int
    definition: JsonValue

    def __post_init__(self) -> None:
        for name in ("scorer", "reducer", "metric"):
            require_string(getattr(self, name), name)
        require_finite(self.value, "aggregate.value")
        require_non_negative_int(self.scored_count, "scored_count")
        require_non_negative_int(self.unscored_count, "unscored_count")
        canonical_dumps(self.definition)

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "scorer": self.scorer,
            "reducer": self.reducer,
            "metric": self.metric,
            "value": require_finite(self.value, "aggregate.value"),
            "scored_count": self.scored_count,
            "unscored_count": self.unscored_count,
            "definition": self.definition,
        }


@dataclass(frozen=True)
class EvaluationArtifactManifestEntry:
    """One provider-declared staged artifact candidate."""

    artifact_id: str
    path: str
    media_type: str
    visibility: ArtifactVisibility
    size_bytes: int
    artifact_content_sha256: str

    def __post_init__(self) -> None:
        require_open_id(self.artifact_id, "artifact_id")
        _validate_relative_path(self.path, "artifact.path")
        _require_media_type(self.media_type)
        require_positive_int(self.size_bytes, "size_bytes")
        require_sha256(self.artifact_content_sha256, "artifact_content_sha256")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "media_type": self.media_type,
            "visibility": self.visibility.value,
            "size_bytes": self.size_bytes,
            "artifact_content_sha256": self.artifact_content_sha256,
        }


@dataclass(frozen=True)
class EvaluationFinishCandidate:
    """Provider manifest candidate returned before Rust artifact sealing."""

    identity: EvaluationIdentity
    outcomes: tuple[CaseOutcome, ...]
    aggregates: tuple[AggregateMetric, ...]
    artifacts: tuple[EvaluationArtifactManifestEntry, ...]
    provider_bundle: ArtifactRef
    normalized_result_sha256: str

    def __post_init__(self) -> None:
        if len({item.case_id for item in self.outcomes}) != len(self.outcomes):
            raise ValueError("finish candidate contains duplicate case outcomes")
        if len({item.artifact_id for item in self.artifacts}) != len(self.artifacts):
            raise ValueError("finish candidate contains duplicate artifacts")
        if self.provider_bundle.artifact_id not in {
            item.artifact_id for item in self.artifacts
        }:
            raise ValueError("provider bundle is absent from artifact manifest")
        require_sha256(self.normalized_result_sha256, "normalized_result_sha256")

    def to_wire(self) -> dict[str, JsonValue]:
        return {
            "identity": self.identity.to_wire(),
            "outcomes": [item.to_wire() for item in self.outcomes],
            "aggregates": [item.to_wire() for item in self.aggregates],
            "artifacts": [item.to_wire() for item in self.artifacts],
            "provider_bundle": self.provider_bundle.to_wire(),
            "normalized_result_sha256": self.normalized_result_sha256,
        }


# Compatibility spelling used by the neutral session protocol in the design.
EvaluationBundle = EvaluationFinishCandidate


@dataclass(frozen=True)
class ScopedProxyBinding:
    """Rust-issued loopback-only compatibility proxy binding."""

    local_locator: str
    grant_id: str
    session_id: str
    secret: str
    service_ids: tuple[str, ...]
    purposes: tuple[str, ...]
    semantic_operation_ids: tuple[str, ...]
    process_scope_sha256: str
    max_operations: int
    max_concurrent_operations: int
    max_request_bytes: int
    max_response_bytes: int
    max_stream_events: int
    expires_after_ms: int

    @classmethod
    def from_wire(cls, value: Any) -> ScopedProxyBinding:
        item = strict_object(
            value,
            field_name="scoped proxy binding",
            allowed={"local_locator", "grant"},
            required={"local_locator", "grant"},
        )
        grant = strict_object(
            item["grant"],
            field_name="scoped proxy grant",
            allowed={
                "grant_id",
                "session_id",
                "secret",
                "service_ids",
                "purposes",
                "semantic_operation_ids",
                "process_scope_sha256",
                "max_operations",
                "max_concurrent_operations",
                "max_request_bytes",
                "max_response_bytes",
                "max_stream_events",
                "expires_after_ms",
            },
            required={
                "grant_id",
                "session_id",
                "secret",
                "service_ids",
                "purposes",
                "semantic_operation_ids",
                "process_scope_sha256",
                "max_operations",
                "max_concurrent_operations",
                "max_request_bytes",
                "max_response_bytes",
                "max_stream_events",
                "expires_after_ms",
            },
        )
        locator = require_string(item["local_locator"], "local_locator", max_bytes=512)
        if locator != "unix:///run/aiperf/evaluator-proxy.sock":
            raise ValueError("proxy locator must name the contained AIPerf socket")
        services = _open_id_array(grant["service_ids"], "service_ids")
        purposes = _open_id_array(grant["purposes"], "purposes")
        operations = _open_id_array(
            grant["semantic_operation_ids"], "semantic_operation_ids"
        )
        secret = require_string(grant["secret"], "proxy secret", max_bytes=512)
        if len(secret) < 32 or any(character.isspace() for character in secret):
            raise ValueError("proxy secret is weak or malformed")
        max_operations = require_positive_int(grant["max_operations"], "max_operations")
        max_concurrent_operations = require_positive_int(
            grant["max_concurrent_operations"], "max_concurrent_operations"
        )
        if max_concurrent_operations > max_operations:
            raise ValueError("max_concurrent_operations exceeds max_operations")
        return cls(
            local_locator=locator,
            grant_id=require_string(grant["grant_id"], "grant_id"),
            session_id=require_opaque_id(grant["session_id"], "session_id"),
            secret=secret,
            service_ids=services,
            purposes=purposes,
            semantic_operation_ids=operations,
            process_scope_sha256=require_sha256(
                grant["process_scope_sha256"], "process_scope_sha256"
            ),
            max_operations=max_operations,
            max_concurrent_operations=max_concurrent_operations,
            max_request_bytes=require_positive_int(
                grant["max_request_bytes"], "max_request_bytes"
            ),
            max_response_bytes=require_positive_int(
                grant["max_response_bytes"], "max_response_bytes"
            ),
            max_stream_events=require_non_negative_int(
                grant["max_stream_events"], "max_stream_events"
            ),
            expires_after_ms=require_positive_int(
                grant["expires_after_ms"], "expires_after_ms"
            ),
        )


def case_manifest_digest(
    cases: tuple[CaseTemplateDescriptor, ...],
    units: tuple[ExecutionUnitTemplateDescriptor, ...],
) -> str:
    """Hash the ordered model-safe template/unit manifest."""
    return canonical_sha256(
        {
            "cases": [case.to_wire() for case in cases],
            "units": [unit.to_wire() for unit in units],
        }
    )


def _open_id_array(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    items = tuple(require_open_id(item, field_name) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} contains duplicates")
    return items


def _opaque_id_array(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    items = tuple(require_opaque_id(item, field_name) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} contains duplicates")
    return items


def _require_media_type(value: Any) -> str:
    if not isinstance(value, str) or not _MEDIA_TYPE_RE.fullmatch(value):
        raise ValueError(f"invalid media type {value!r}")
    return value


def _validate_relative_path(value: Any, field_name: str) -> str:
    path_value = require_string(value, field_name)
    path = PurePosixPath(path_value)
    if (
        path.is_absolute()
        or not path.parts
        or "\x00" in path_value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be a contained normalized relative path")
    return path_value


def _reject_connection_authority(value: JsonValue) -> None:
    """Reject control values capable of selecting an upstream connection."""

    def walk(item: JsonValue, path: tuple[str, ...]) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = key.lower().replace("-", "_")
                if normalized in _FORBIDDEN_AUTHORITY_FIELDS:
                    raise ValueError(
                        "control payload contains forbidden connection-authority field "
                        + ".".join((*path, key))
                    )
                walk(child, (*path, key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, (*path, str(index)))

    walk(value, ())
