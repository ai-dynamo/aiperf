# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable GSM8K canary asset and bundle helpers for stock providers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

from aiperf.accuracy.evaluation.canonical import canonical_dumps, canonical_sha256
from aiperf.accuracy.evaluation.contracts import (
    AggregateMetric,
    ArtifactRef,
    ArtifactVisibility,
    CaseOutcome,
    CaseTemplateDescriptor,
    EvaluationArtifactManifestEntry,
    EvaluationFinishCandidate,
    EvaluationHostBinding,
    EvaluationIdentity,
    EvaluationIdentityComponent,
    EvaluationWorkerIdentity,
    ExecutionUnitTemplateDescriptor,
    ResolvedAsset,
    case_manifest_digest,
)

ASSET_ID = "openai_gsm8k_main_test_canary"
ASSET_REVISION = (
    "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866:main:test:first5"
)
ASSET_SHA256 = "fc9b5c03206d193c0013baf2d6344a133fe0096a2b47cd1eafdcee297dfd398a"
ASSET_MEDIA_TYPE = "application/x-ndjson"
SOURCE_LABEL = "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866"


def bind_gsm8k_asset(
    assets: Sequence[ResolvedAsset],
) -> tuple[Path, tuple[dict[str, str], ...]]:
    """Validate and decode the exact Rust-bound five-record canary asset."""
    if len(assets) != 1 or assets[0].asset_id != ASSET_ID:
        raise ValueError("GSM8K provider requires exactly its frozen canary asset")
    asset = assets[0]
    if (
        asset.content_sha256 != ASSET_SHA256
        or asset.immutable_revision != ASSET_REVISION
        or asset.media_type != ASSET_MEDIA_TYPE
    ):
        raise ValueError("GSM8K asset identity drift")
    path = Path(asset.contained_path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ASSET_SHA256:
        raise ValueError("GSM8K contained asset bytes failed worker re-verification")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            value = orjson.loads(line)
        except orjson.JSONDecodeError as error:
            raise ValueError(
                f"GSM8K asset line {line_number} is invalid JSON"
            ) from error
        if set(value) != {"question", "answer"} or not all(
            isinstance(value[field], str) for field in ("question", "answer")
        ):
            raise ValueError("GSM8K asset record has an unexpected image")
        records.append(value)
    if len(records) != 5:
        raise ValueError("GSM8K canary asset must contain exactly five records")
    return path, tuple(records)


def build_identity(
    *,
    worker: EvaluationWorkerIdentity,
    config_schema_sha256: str,
    provider_config: Any,
    case_templates: tuple[CaseTemplateDescriptor, ...],
    unit_templates: tuple[ExecutionUnitTemplateDescriptor, ...],
    components: tuple[EvaluationIdentityComponent, ...],
    policies: Any,
    host_binding: EvaluationHostBinding,
) -> EvaluationIdentity:
    """Combine provider-owned and Rust-authored immutable identity components."""
    return EvaluationIdentity(
        worker=worker,
        config_schema_sha256=config_schema_sha256,
        resolved_config_sha256=canonical_sha256(provider_config),
        dataset=EvaluationIdentityComponent(
            name="gsm8k_canary",
            version=ASSET_REVISION,
            source_sha256=ASSET_SHA256,
        ),
        components=components,
        ordered_manifest_sha256=case_manifest_digest(case_templates, unit_templates),
        case_templates=case_templates,
        unit_templates=unit_templates,
        policies=policies,
        host=host_binding.host,
        route_map_sha256=host_binding.route_map_sha256,
        prepared_endpoints_sha256=host_binding.prepared_endpoints_sha256,
        sandbox_sha256=host_binding.sandbox_sha256,
    )


def finish_candidate(
    *,
    identity: EvaluationIdentity,
    outcomes: tuple[CaseOutcome, ...],
    aggregates: tuple[AggregateMetric, ...],
    restricted_records: Any,
    staging_root: Path,
    filename: str,
    additional_artifacts: Sequence[tuple[str, Path, str, ArtifactVisibility]] = (),
    normalized_result: Any | None = None,
) -> EvaluationFinishCandidate:
    """Write a canonical restricted provider bundle and return its manifest candidate."""
    semantic = {
        "identity": identity.to_wire(),
        "outcomes": [outcome.to_wire() for outcome in outcomes],
        "aggregates": [aggregate.to_wire() for aggregate in aggregates],
        "records": restricted_records,
    }
    content = canonical_dumps(semantic)
    path = staging_root / filename
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    artifact_id = "provider_bundle"
    reference = ArtifactRef(
        artifact_id=artifact_id,
        path=filename,
        visibility=ArtifactVisibility.RESTRICTED,
    )
    manifests = [
        EvaluationArtifactManifestEntry(
            artifact_id=artifact_id,
            path=filename,
            media_type="application/json",
            visibility=ArtifactVisibility.RESTRICTED,
            size_bytes=len(content),
            artifact_content_sha256=digest,
        )
    ]
    for extra_id, extra_path, media_type, visibility in additional_artifacts:
        resolved = extra_path.resolve(strict=True)
        root = staging_root.resolve(strict=True)
        if resolved.parent != root:
            raise ValueError("provider artifact escaped the contained staging root")
        extra_content = resolved.read_bytes()
        manifests.append(
            EvaluationArtifactManifestEntry(
                artifact_id=extra_id,
                path=resolved.name,
                media_type=media_type,
                visibility=visibility,
                size_bytes=len(extra_content),
                artifact_content_sha256=hashlib.sha256(extra_content).hexdigest(),
            )
        )
    return EvaluationFinishCandidate(
        identity=identity,
        outcomes=outcomes,
        aggregates=aggregates,
        artifacts=tuple(manifests),
        provider_bundle=reference,
        normalized_result_sha256=canonical_sha256(
            normalized_result
            if normalized_result is not None
            else {
                "outcomes": [outcome.to_wire() for outcome in outcomes],
                "aggregates": [aggregate.to_wire() for aggregate in aggregates],
            }
        ),
    )
