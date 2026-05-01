#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Kubernetes CRD schema from AIPerfConfig Pydantic model.

Introspects the AIPerfConfig model to produce a complete CRD YAML that
stays in sync with the Python configuration schema. Operator-specific
fields (image, podTemplate, scheduling, etc.) and the status sub-schema
are defined statically.

Usage:
    ./tools/generate_crd.py
    ./tools/generate_crd.py --check
    ./tools/generate_crd.py --verbose
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

# Allow direct execution: add repo root to path for 'tools' package imports
if __name__ == "__main__" and "tools" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

import yaml

from tools._core import (
    GeneratedFile,
    Generator,
    GeneratorResult,
    main,
    print_step,
)

# =============================================================================
# Configuration
# =============================================================================

HELM_CRD_FILE = Path("deploy/helm/aiperf-operator/templates/crd.yaml")
HELM_SWEEP_CRD_FILE = Path("deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml")
HELM_CHART_FILE = Path("deploy/helm/aiperf-operator/Chart.yaml")
PYPROJECT_FILE = Path("pyproject.toml")

SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
)

# Keys to strip from JSON Schema that K8s CRDs don't support
_STRIP_KEYS = frozenset({"title", "examples", "$defs", "$schema"})

# Maximum recursion depth before falling back to preserve-unknown-fields.
# AIPerfSweep wraps an AIPerfJob in spec.template.spec, adding 3 levels;
# the deepest legitimate path today is spec.template.spec.benchmark.runtime.<field>
# at depth 6, but ModelItem and other inner classes go several levels deeper.
_MAX_DEPTH = 16


# =============================================================================
# JSON Schema -> K8s OpenAPI v3 Converter
# =============================================================================


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref string to its definition."""
    name = ref.rsplit("/", 1)[-1]
    if name not in defs:
        return {}
    return defs[name]


def _is_nullable_anyof(schema: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    """Check if schema is anyOf: [{real_type}, {type: null}]."""
    any_of = schema.get("anyOf")
    if not any_of or len(any_of) != 2:
        return False, None

    null_idx = None
    for i, item in enumerate(any_of):
        if item.get("type") == "null":
            null_idx = i

    if null_idx is None:
        return False, None

    real_schema = any_of[1 - null_idx]
    return True, real_schema


def _convert_schema(
    schema: dict[str, Any],
    defs: dict[str, Any],
    depth: int = 0,
) -> dict[str, Any]:
    """Convert a JSON Schema node to K8s-compatible OpenAPI v3.

    Recursively resolves $ref, handles anyOf-with-null (nullable),
    converts discriminated unions, and strips unsupported keys.
    Falls back to x-kubernetes-preserve-unknown-fields at max depth.
    """
    if not schema:
        return {}

    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], defs)
        merged = _convert_schema(resolved, defs, depth)
        if "description" in schema and schema["description"]:
            merged["description"] = schema["description"]
        # Preserve sibling x-kubernetes-preserve-unknown-fields markers
        # (Pydantic emits these alongside $ref via json_schema_extra to mark
        # narrow shorthand-accepting boundaries — see Task 4 of plan
        # 2026-04-26-aiperfconfig-strict-crd-schema.md).
        if schema.get("x-kubernetes-preserve-unknown-fields"):
            merged["x-kubernetes-preserve-unknown-fields"] = True
        return merged

    if depth > _MAX_DEPTH:
        result: dict[str, Any] = {
            "type": "object",
            "x-kubernetes-preserve-unknown-fields": True,
        }
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    is_nullable, real_type = _is_nullable_anyof(schema)
    if is_nullable and real_type is not None:
        result = _convert_schema(real_type, defs, depth)
        if "description" in schema and "description" not in result:
            result["description"] = schema["description"]
        if (
            "default" in schema
            and schema["default"] is not None
            and "default" not in result
        ):
            result["default"] = schema["default"]
        # Preserve sibling x-kubernetes-preserve-unknown-fields marker
        # (Task 5 hoisted shortcuts like model/dataset/warmup/profiling use
        # anyOf:[{},{type:null}] with this marker to opt the field out of
        # strict apiserver validation while keeping it visible in the CRD).
        if schema.get("x-kubernetes-preserve-unknown-fields"):
            result["x-kubernetes-preserve-unknown-fields"] = True
            # The marker requires type=object on the K8s side.
            result.setdefault("type", "object")
        return result

    if "anyOf" in schema and not is_nullable:
        any_of = schema["anyOf"]
        scalar_types = []
        for alt in any_of:
            if "type" in alt and alt["type"] not in ("object", "array"):
                scalar_types.append(alt["type"])
            elif "const" in alt:
                scalar_types.append(type(alt["const"]).__name__)
        if scalar_types and len(scalar_types) == len(any_of):
            result = _convert_schema(any_of[0], defs, depth)
            for key in ("default", "description"):
                if key in schema:
                    result[key] = schema[key]
            return result

        # Mixed-type anyOf: K8s structural schema allows no-type when paired
        # with x-kubernetes-preserve-unknown-fields: true (apiserver skips
        # type enforcement). Don't force type=object here — that breaks
        # leaves like `artifacts.records: list | Literal[False]` whose default
        # is a non-object scalar.
        result = {"x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    if "oneOf" in schema:
        result = {"x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    ap = schema.get("additionalProperties", {})
    if isinstance(ap, dict) and "discriminator" in ap:
        result = {"type": "object", "x-kubernetes-preserve-unknown-fields": True}
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    result = {}

    if "type" in schema:
        result["type"] = schema["type"]
    else:
        # Pydantic emits an empty/no-type schema for ``Any``-typed fields and
        # for some discriminated-union leaves. K8s structural schemas reject
        # objects without `type`, so fall back to a permissive object shape.
        result["type"] = "object"
        result["x-kubernetes-preserve-unknown-fields"] = True

    if "description" in schema:
        result["description"] = schema["description"]

    if "enum" in schema:
        result["enum"] = schema["enum"]

    if "const" in schema:
        result["enum"] = [schema["const"]]
        if "type" not in result:
            val = schema["const"]
            if isinstance(val, str):
                result["type"] = "string"
            elif isinstance(val, bool):
                result["type"] = "boolean"
            elif isinstance(val, int):
                result["type"] = "integer"

    if "default" in schema and schema["default"] is not None:
        result["default"] = schema["default"]

    for key in ("minimum", "maximum"):
        if key in schema:
            result[key] = schema[key]

    # K8s CRDs use OpenAPI v3 where exclusiveMinimum/Maximum are booleans,
    # not numbers like JSON Schema Draft 2020-12. Convert by setting the
    # boolean flag and moving the value to minimum/maximum.
    if "exclusiveMinimum" in schema:
        val = schema["exclusiveMinimum"]
        if isinstance(val, bool):
            result["exclusiveMinimum"] = val
        else:
            result["exclusiveMinimum"] = True
            result.setdefault("minimum", val)
    if "exclusiveMaximum" in schema:
        val = schema["exclusiveMaximum"]
        if isinstance(val, bool):
            result["exclusiveMaximum"] = val
        else:
            result["exclusiveMaximum"] = True
            result.setdefault("maximum", val)

    for key in ("minLength", "maxLength", "pattern"):
        if key in schema:
            result[key] = schema[key]

    if "format" in schema and schema["format"] != "path":
        result["format"] = schema["format"]

    if schema.get("type") == "object" or "properties" in schema:
        result["type"] = "object"

        if "properties" in schema:
            props = {}
            for prop_name, prop_schema in schema["properties"].items():
                props[prop_name] = _convert_schema(prop_schema, defs, depth + 1)
            if props:
                result["properties"] = props

        if "required" in schema:
            result["required"] = schema["required"]

        if "additionalProperties" in schema:
            ap = schema["additionalProperties"]
            if isinstance(ap, bool):
                if ap:
                    result["additionalProperties"] = ap
            elif isinstance(ap, dict):
                if "$ref" in ap or "type" in ap:
                    converted = _convert_schema(ap, defs, depth + 1)
                    if converted:
                        result["additionalProperties"] = converted
                elif "discriminator" in ap:
                    result["x-kubernetes-preserve-unknown-fields"] = True
                else:
                    result["additionalProperties"] = _convert_schema(
                        ap, defs, depth + 1
                    )

        if schema.get("additionalProperties") is False:
            result.pop("additionalProperties", None)

    if schema.get("type") == "array" and "items" in schema:
        result["items"] = _convert_schema(schema["items"], defs, depth + 1)

    for key in ("minItems", "maxItems"):
        if key in schema:
            result[key] = schema[key]

    for key in _STRIP_KEYS:
        result.pop(key, None)

    # Preserve x-kubernetes-preserve-unknown-fields if explicitly set on the
    # source schema. Used by Pydantic fields with json_schema_extra to mark
    # narrow shorthand boundaries (e.g. EndpointConfig.urls is an array but
    # the before-validator also accepts a single string).
    if schema.get("x-kubernetes-preserve-unknown-fields"):
        result["x-kubernetes-preserve-unknown-fields"] = True

    return result


def convert_aiperf_config_fields(
    schema: dict[str, Any], verbose: bool = False
) -> dict[str, Any]:
    """Convert AIPerfConfig's JSON Schema properties to K8s CRD spec properties."""
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    result = {}
    for name, prop_schema in properties.items():
        converted = _convert_schema(prop_schema, defs, depth=0)
        if verbose:
            print_step(f"Converted field: {name}")
        result[name] = converted

    return result


def _add_validation_rules(node: dict[str, Any], rules: tuple[dict, ...]) -> None:
    """Append ``rules`` to ``node['x-kubernetes-validations']`` (de-duped by rule text)."""
    bag = node.setdefault("x-kubernetes-validations", [])
    existing = {r.get("rule") for r in bag if isinstance(r, dict)}
    for r in rules:
        if r["rule"] not in existing:
            bag.append(r)
            existing.add(r["rule"])


def _decorate_aiperf_config_node(node: dict[str, Any]) -> None:
    """Attach CEL rules + relax ``required`` on AIPerfConfig-shaped nodes.

    Detected by shape (presence of all four shorthand-sibling keys in
    ``properties``) so AIPerfSweep's nested ``template.spec.benchmark`` is
    fixed up via the same walker as the AIPerfJob top-level benchmark.

    Bundles every cross-field invariant declared on AIPerfConfig that the
    apiserver can enforce ahead of the operator's reconcile loop:

    * Shorthand-or-canonical OR-requirement (replaces the structural
      ``required`` list — the operator's before-validator hoists shorthand
      after admission).
    * Shorthand-and-canonical mutual exclusion (you can't set both forms).
    * Cross-field rules currently encoded as Pydantic ``@model_validator``
      decorators on ``AIPerfConfig`` and ``BenchmarkConfig``: dataset-name
      uniqueness, phase-name uniqueness, phase→dataset reference integrity,
      seamless-not-on-first-phase, ``parameter_sweep_same_seed`` requires
      ``random_seed``, dashboard UI incompatible with sweeps. See the named
      validators in ``src/aiperf/config/config.py``.
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not (
        "model" in props
        and "dataset" in props
        and "warmup" in props
        and "profiling" in props
    ):
        return

    # Relax structural required: shorthand siblings cover models/datasets/phases
    # via the CEL OR-rules below. ``endpoint`` stays required (no shorthand).
    required = node.get("required")
    if isinstance(required, list):
        relaxable = {"models", "datasets", "phases"}
        node["required"] = [r for r in required if r not in relaxable]
        if not node["required"]:
            del node["required"]

    _add_validation_rules(
        node,
        (
            # Tier 1A — OR-requirements (shorthand or canonical).
            {
                "rule": "has(self.models) || has(self.model)",
                "message": (
                    "benchmark requires either 'models' (canonical: object with "
                    "'items') or 'model' (shorthand: string, list of strings, "
                    "or ModelsAdvanced object). The operator hoists 'model' "
                    "into 'models' before validation."
                ),
            },
            {
                "rule": "has(self.datasets) || has(self.dataset)",
                "message": (
                    "benchmark requires either 'datasets' (canonical: list "
                    "with named entries) or 'dataset' (shorthand: single dict, "
                    "hoisted into a one-entry 'datasets' list with "
                    "name='default')."
                ),
            },
            {
                "rule": ("has(self.phases) || has(self.warmup) || has(self.profiling)"),
                "message": (
                    "benchmark requires either 'phases' (canonical: ordered "
                    "list with named entries) or shorthand 'warmup'/"
                    "'profiling' phase dicts. Top-level 'warmup'/'profiling' "
                    "are hoisted into a [warmup, profiling] phases list "
                    "before validation."
                ),
            },
            # Tier 1A — mutual exclusion.
            {
                "rule": "!(has(self.models) && has(self.model))",
                "message": (
                    "set 'models' (canonical) OR 'model' (shorthand), not both"
                ),
            },
            {
                "rule": "!(has(self.datasets) && has(self.dataset))",
                "message": (
                    "set 'datasets' (canonical) OR 'dataset' (shorthand), not both"
                ),
            },
            {
                "rule": (
                    "!(has(self.phases) && (has(self.warmup) || has(self.profiling)))"
                ),
                "message": (
                    "use 'phases' (canonical list) OR top-level "
                    "'warmup'/'profiling' shorthand, not both"
                ),
            },
            {
                "rule": "!has(self.warmup) || has(self.profiling)",
                "message": (
                    "'warmup' shorthand requires 'profiling' alongside it; "
                    "warmup-only runs are not supported"
                ),
            },
            # Tier 2G — parameter_sweep_same_seed requires random_seed.
            {
                "rule": (
                    "!has(self.multiRun) || "
                    "!has(self.multiRun.parameterSweepSameSeed) || "
                    "!self.multiRun.parameterSweepSameSeed || "
                    "has(self.randomSeed)"
                ),
                "message": (
                    "multiRun.parameterSweepSameSeed=true requires randomSeed "
                    "to be set; without a base seed every variation already "
                    "gets a fresh draw, so 'same seed' is meaningless"
                ),
            },
            # Tier 2I — dashboard UI incompatible with sweeps.
            {
                "rule": (
                    "!has(self.sweep) || !has(self.runtime) || "
                    "!has(self.runtime.ui) || self.runtime.ui != 'dashboard'"
                ),
                "message": (
                    "Dashboard UI cannot multiplex sweep variations (live "
                    "results would overwrite each other in the console). Use "
                    "ui='simple' or 'none' with sweep configurations."
                ),
            },
            # Tier 4P/4Q/4R skipped: the array items for `phases` and
            # `datasets` are opaque (`x-kubernetes-preserve-unknown-fields`)
            # because their entries are heterogeneous Pydantic discriminated
            # unions. CEL can't dereference `phases[].name`, `datasets[].name`,
            # `phases[].dataset`, or `phases[0].seamless` through opaque
            # items, so phase/dataset name uniqueness, phase→dataset
            # reference integrity, and "seamless not on first" stay enforced
            # in the operator-side Pydantic validators
            # (validate_phase_names_unique, validate_datasets_unique_names,
            # validate_dataset_references, validate_seamless_not_on_first_phase
            # in src/aiperf/config/config.py).
        ),
    )


def _decorate_endpoint_node(node: dict[str, Any]) -> None:
    """Attach CEL rules to EndpointConfig-shaped nodes.

    Detected by shape: presence of ``urls`` + ``apiKey`` + ``connectionReuse``
    (a combination unique to ``EndpointConfig``). Mirrors the
    ``_validate_template_required`` and ``_validate_request_content_type``
    Pydantic validators in ``src/aiperf/config/endpoint.py``.
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not ("urls" in props and "apiKey" in props and "connectionReuse" in props):
        return
    _add_validation_rules(
        node,
        (
            # Tier 1B — type=template requires template.
            {
                "rule": (
                    "!has(self.type) || self.type != 'template' || has(self.template)"
                ),
                "message": (
                    "endpoint.template is required when endpoint.type='template'"
                ),
            },
            {
                "rule": (
                    "!has(self.template) || (has(self.type) && self.type == 'template')"
                ),
                "message": (
                    "endpoint.template is only used when endpoint.type='template'"
                ),
            },
            # Tier 2J — multipart/form-data is video_generation-only today.
            {
                "rule": (
                    "!has(self.requestContentType) || "
                    "self.requestContentType != 'multipart/form-data' || "
                    "(has(self.type) && self.type == 'video_generation')"
                ),
                "message": (
                    "requestContentType='multipart/form-data' is only "
                    "supported on endpoint.type='video_generation' today"
                ),
            },
            # Tier 4O — every URL must be a valid URL.
            {
                "rule": "self.urls.all(u, isURL(u))",
                "message": "every endpoint.urls entry must be a valid URL",
            },
        ),
    )


def _decorate_runtime_node(node: dict[str, Any]) -> None:
    """Attach CEL rules to RuntimeConfig-shaped nodes.

    Detected by shape: ``apiPort`` + ``apiHost`` + ``workersPerPod``. Mirrors
    ``_validate_api_host_requires_port`` (already on this file) plus
    workersMin/workers ordering.
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not ("apiPort" in props and "apiHost" in props and "workersPerPod" in props):
        return
    _add_validation_rules(
        node,
        (
            # Tier 0 — apiHost requires apiPort (was inline; now centralized).
            {
                "rule": "!has(self.apiHost) || has(self.apiPort)",
                "message": ("runtime.apiHost requires runtime.apiPort to be set"),
            },
            # Tier 1F — workersMin <= workers when both set.
            {
                "rule": (
                    "!has(self.workersMin) || !has(self.workers) || "
                    "self.workersMin <= self.workers"
                ),
                "message": ("runtime.workersMin must be <= runtime.workers"),
            },
        ),
    )


def _decorate_multirun_node(node: dict[str, Any]) -> None:
    """Attach CEL rules to AIPerfConfig.multiRun-shaped nodes.

    Detected by shape: ``numRuns`` + ``convergenceMetric`` + ``mode`` (a
    combination unique to ``MultiRunConfig`` in ``_models_runtime``).
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not ("numRuns" in props and "convergenceMetric" in props and "mode" in props):
        return
    _add_validation_rules(
        node,
        (
            # Tier 2H — adaptive convergence incompatible with mode='repeated'.
            {
                "rule": (
                    "!has(self.convergenceMetric) || "
                    "!has(self.mode) || self.mode != 'repeated'"
                ),
                "message": (
                    "adaptive convergence (convergenceMetric) is incompatible "
                    "with mode='repeated'; use mode='independent'"
                ),
            },
        ),
    )


def _decorate_artifacts_node(node: dict[str, Any]) -> None:
    """Attach CEL rules to ArtifactsConfig-shaped nodes.

    Detected by shape: ``benchmarkId`` + ``cliCommand`` + ``dir``.
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not ("benchmarkId" in props and "cliCommand" in props and "dir" in props):
        return
    _add_validation_rules(
        node,
        (
            # Tier 3K — benchmarkId immutable once set (allow operator to
            # stamp it on first reconcile when user submits CR without it).
            {
                "rule": (
                    "!has(oldSelf.benchmarkId) || "
                    "oldSelf.benchmarkId == self.benchmarkId"
                ),
                "message": (
                    "artifacts.benchmarkId is immutable once set; mutating it "
                    "would orphan artifacts already keyed by the old id"
                ),
            },
        ),
    )


def _decorate_all(node: dict[str, Any]) -> None:
    """Apply every shape-detector decorator to ``node``."""
    _decorate_aiperf_config_node(node)
    _decorate_endpoint_node(node)
    _decorate_runtime_node(node)
    _decorate_multirun_node(node)
    _decorate_artifacts_node(node)


def _ensure_type_on_preserve_unknown(node: dict[str, Any]) -> None:
    """Default ``type: object`` on any node that has the preserve-unknown marker.

    K8s structural-schema validation rejects ``x-kubernetes-preserve-unknown-fields:
    true`` without a declared ``type``, AND CEL field access compiles only on
    nodes where the apiserver knows the type. The Pydantic→OpenAPI walker leaves
    a few branches typeless (mixed-type anyOf, oneOf, sibling markers on $refs),
    so this pass closes the gap before CRD apply.
    """
    if not isinstance(node, dict):
        return
    if node.get("x-kubernetes-preserve-unknown-fields") is True and "type" not in node:
        node["type"] = "object"


def _walk_dict_apply(node: Any, fn: Any) -> None:
    """Depth-first traversal that applies ``fn`` to every dict node."""
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk_dict_apply(v, fn)
    elif isinstance(node, list):
        for item in node:
            _walk_dict_apply(item, fn)


def _status_schema() -> dict[str, Any]:
    """Return the status sub-schema."""
    return {
        "type": "object",
        "x-kubernetes-preserve-unknown-fields": True,
        "properties": {
            "observedGeneration": {
                "type": "integer",
                "format": "int64",
                "description": "Generation of the spec that was last processed",
            },
            "phase": {
                "type": "string",
                "description": "Current job phase",
                "enum": [
                    "Pending",
                    "Queued",
                    "Initializing",
                    "Running",
                    "Completed",
                    "Failed",
                    "Cancelled",
                ],
            },
            "jobId": {
                "type": "string",
                "description": "Unique job identifier",
            },
            "startTime": {
                "type": "string",
                "format": "date-time",
                "description": "Time when job started",
            },
            "completionTime": {
                "type": "string",
                "format": "date-time",
                "description": "Time when job completed",
            },
            "jobSetName": {
                "type": "string",
                "description": "Name of the managed JobSet",
            },
            "error": {
                "type": "string",
                "description": "Error message if failed",
            },
            "workers": {
                "type": "object",
                "description": "Controller-authored aggregate worker status.",
                "properties": {
                    "ready": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Dispatch-ready worker count.",
                    },
                    "total": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Declared worker count.",
                    },
                    "dispatchable": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Workers eligible to receive credits.",
                    },
                    "routerConnected": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Workers connected to the router.",
                    },
                    "readyRecordProcessors": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Ready record processors.",
                    },
                    "declaredRecordProcessors": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Declared record processors.",
                    },
                    "readyPods": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Usable worker pods.",
                    },
                    "totalPods": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Observed worker pods.",
                    },
                    "degradedPods": {
                        "type": "integer",
                        "format": "int32",
                        "description": "Usable but degraded worker pods.",
                    },
                },
            },
            "phases": {
                "type": "object",
                "description": "Progress tracking for each benchmark phase",
                "additionalProperties": {
                    "type": "object",
                    "description": "Phase progress stats",
                    "x-kubernetes-preserve-unknown-fields": True,
                },
            },
            "currentPhase": {
                "type": "string",
                "description": "Current benchmark phase (warmup, profiling, etc)",
            },
            "liveMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Live metrics updated during benchmark run",
            },
            "serverMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Server-side metrics from inference server",
            },
            "results": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Final benchmark results and metrics",
            },
            "resultsPath": {
                "type": "string",
                "description": "Path to stored results on operator PVC",
            },
            "runEpoch": {
                "type": "integer",
                "format": "int64",
                "minimum": 0,
                "description": "Epoch-seconds key of the most recent successful run. Use as {epoch} in /api/v1/results/<ns>/<name>/runs/<epoch>/ to pin historical artifacts.",
            },
            "liveSummary": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Live summary metrics updated during benchmark run",
            },
            "summary": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Final summary metrics after benchmark completion",
            },
            "resultsTtlDays": {
                "type": "integer",
                "format": "int32",
                "description": "Days to retain result files before cleanup",
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["True", "False", "Unknown"],
                        },
                        "reason": {"type": "string"},
                        "message": {"type": "string"},
                        "lastTransitionTime": {
                            "type": "string",
                            "format": "date-time",
                        },
                    },
                },
                "description": "Detailed status conditions",
            },
        },
    }


def _printer_columns() -> list[dict[str, Any]]:
    """Return additionalPrinterColumns for kubectl output."""
    return [
        {
            "name": "Phase",
            "type": "string",
            "jsonPath": ".status.phase",
        },
        {
            "name": "Stage",
            "type": "string",
            "jsonPath": ".status.currentPhase",
            "description": "Current benchmark stage (warmup, profiling)",
        },
        {
            "name": "Progress",
            "type": "string",
            "jsonPath": ".status.phases.profiling.requestsCompleted",
            "description": "Requests completed in profiling phase",
        },
        {
            "name": "QPS",
            "type": "number",
            "jsonPath": ".status.phases.profiling.requestsPerSecond",
            "description": "Requests per second in profiling phase",
        },
        {
            "name": "Age",
            "type": "date",
            "jsonPath": ".metadata.creationTimestamp",
        },
    ]


# =============================================================================
# CRD Assembly
# =============================================================================


def _deployment_config_properties() -> dict[str, Any]:
    """Generate operator-specific fields from DeploymentConfig model."""
    from aiperf.config.deployment import DeploymentConfig

    schema = DeploymentConfig.model_json_schema(by_alias=True)
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    result = {}
    for name, prop_schema in properties.items():
        result[name] = _convert_schema(prop_schema, defs, depth=0)

    return result


def _build_crd(_config_properties: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full CRD document."""
    spec_properties: dict[str, Any] = {}

    # Operator/deployment fields from DeploymentConfig model
    operator = _deployment_config_properties()
    spec_properties["image"] = operator.pop("image")
    spec_properties["imagePullPolicy"] = operator.pop("imagePullPolicy")
    spec_properties["keepFailedPods"] = {
        "type": "boolean",
        "description": (
            "Preserve failed JobSet pod attempts for debugging by disabling "
            "retries and TTL cleanup."
        ),
        "default": False,
    }

    # AIPerfConfig fields nested under benchmark key.
    # The schema is fully walked so the apiserver enforces structural validation
    # on every field. Narrow shorthand boundaries (models, distributions,
    # endpoint.urls, top-level shortcuts like model/dataset/warmup/profiling,
    # SyntheticDataset.isl/osl) carry x-kubernetes-preserve-unknown-fields:true
    # via json_schema_extra so before-validators on AIPerfConfig can normalize
    # shorthand forms (e.g. ``models: ["name"]``) on the controller side.
    from aiperf.config.config import AIPerfConfig

    aiperf_config_raw = AIPerfConfig.model_json_schema(mode="validation")
    aiperf_defs = aiperf_config_raw.get("$defs", {})
    benchmark_walked = _convert_schema(aiperf_config_raw, aiperf_defs, depth=0)

    benchmark_walked["description"] = (
        "Benchmark configuration (AIPerfConfig). Strictly typed via the\n"
        "AIPerfConfig schema, with x-kubernetes-preserve-unknown-fields: true\n"
        "at narrow shorthand boundaries (models, distributions, endpoint urls,\n"
        "top-level shortcut fields, telemetry urls).\n"
        "\n"
        "Field naming: the apiserver schema enforces camelCase (e.g.\n"
        "urlStrategy, apiKey, readyCheckTimeout). The Pydantic model also\n"
        "accepts the snake_case form (url_strategy, api_key, …) used in\n"
        "AIPerf CLI YAML — those names are accepted by the operator at parse\n"
        "time but are not advertised by this schema, so kubectl/IDE tooling\n"
        "should write camelCase. Shorthand forms (e.g. models: ['name'],\n"
        "single-phase dict, top-level warmup/profiling) are accepted at marked\n"
        "boundaries and normalized by the operator before validation."
    )
    spec_properties["benchmark"] = benchmark_walked

    # Apply every shape-detector decorator (relaxed-required + cross-field
    # CEL invariants) across the AIPerfConfig walk. Decorators detect their
    # target node by its property shape, so they fire on AIPerfJob's
    # spec.benchmark and on AIPerfSweep's spec.template.spec.benchmark from
    # the same call. See _decorate_all and the individual _decorate_* helpers.
    _walk_dict_apply(benchmark_walked, _ensure_type_on_preserve_unknown)
    _walk_dict_apply(benchmark_walked, _decorate_all)

    # Remaining operator fields (connectionsPerWorker, timeoutSeconds, etc.).
    # skipEndpointCheck lives on AIPerfJobSpec (not DeploymentConfig) and is
    # emitted as a static sibling of `cancel`, preserving insertion order.
    for key, value in operator.items():
        spec_properties[key] = value
        if key == "cancel":
            spec_properties["skipEndpointCheck"] = {
                "type": "boolean",
                "description": (
                    "Skip the operator-side endpoint reachability probe "
                    "before deploying"
                ),
                "default": False,
            }

    # Tier 3N — scheduling.queueName is immutable after admission. Kueue's
    # own contract treats queueName as immutable, so mirror that at the
    # apiserver level for a clearer rejection message.
    scheduling = spec_properties.get("scheduling")
    if isinstance(scheduling, dict):
        scheduling.setdefault("type", "object")
        _add_validation_rules(
            scheduling,
            (
                {
                    "rule": (
                        "!has(oldSelf.queueName) || oldSelf.queueName == self.queueName"
                    ),
                    "message": (
                        "scheduling.queueName is immutable once set (Kueue "
                        "treats queueName as immutable after admission)"
                    ),
                },
            ),
        )

    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {
            "name": "aiperfjobs.aiperf.nvidia.com",
            "annotations": {
                # Keep the CRD when the helm release is uninstalled so other
                # test modules (which share the package-scoped cluster) don't
                # see a Terminating CRD.
                "helm.sh/resource-policy": "keep",
            },
        },
        "spec": {
            "group": "aiperf.nvidia.com",
            "names": {
                "kind": "AIPerfJob",
                "listKind": "AIPerfJobList",
                "plural": "aiperfjobs",
                "singular": "aiperfjob",
                "shortNames": ["apj", "aiperf"],
            },
            "scope": "Namespaced",
            "versions": [
                {
                    "name": "v1alpha1",
                    "served": True,
                    "storage": True,
                    "additionalPrinterColumns": _printer_columns(),
                    "subresources": {"status": {}},
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "required": ["spec"],
                            "properties": {
                                "spec": {
                                    "type": "object",
                                    "description": (
                                        "AIPerfJob specification.\n"
                                        "\n"
                                        "spec.benchmark holds AIPerfConfig fields (models, endpoint,\n"
                                        "datasets, phases, etc.) using camelCase aliases (urlStrategy,\n"
                                        "apiKey, readyCheckTimeout, …). The underlying Pydantic model\n"
                                        "also accepts the snake_case names used in AIPerf CLI YAML,\n"
                                        "but the apiserver schema only advertises camelCase.\n"
                                        "\n"
                                        "Top-level deployment fields (image, podTemplate, scheduling,\n"
                                        "etc.) use camelCase per Kubernetes API conventions."
                                    ),
                                    "properties": spec_properties,
                                },
                                "status": _status_schema(),
                            },
                        },
                    },
                },
            ],
        },
    }


# =============================================================================
# AIPerfSweep CRD
# =============================================================================


def _aiperfsweep_status_schema() -> dict[str, Any]:
    """OpenAPI V3 schema for AIPerfSweep.status.

    The orchestrator writes phase, run counts, per-cell summaries, and refs to
    aggregated artifacts here. Most nested objects use
    ``x-kubernetes-preserve-unknown-fields`` so the schema can evolve without a
    CRD bump.
    """
    return {
        "type": "object",
        "x-kubernetes-preserve-unknown-fields": True,
        "properties": {
            "phase": {
                "type": "string",
                "enum": [
                    "Pending",
                    "Running",
                    "Aggregating",
                    "Succeeded",
                    "PartiallyFailed",
                    "Failed",
                    "Cancelled",
                ],
            },
            "runEpoch": {"type": "integer", "format": "int64"},
            "totalVariations": {"type": "integer", "format": "int32"},
            "maxTotalRuns": {"type": "integer", "format": "int32"},
            "completedRuns": {"type": "integer", "format": "int32"},
            "failedRuns": {"type": "integer", "format": "int32"},
            "currentCell": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "cells": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "aggregation": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "aggregateRef": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "runtimeRef": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "childRunEpochsRef": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "startTime": {"type": "string", "format": "date-time"},
            "completionTime": {"type": "string", "format": "date-time"},
            "lastChildEvent": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "x-kubernetes-preserve-unknown-fields": True,
                },
            },
        },
    }


def _aiperfsweep_printer_columns() -> list[dict[str, Any]]:
    """``additionalPrinterColumns`` for ``kubectl get aiperfsweeps``."""
    return [
        {"name": "Phase", "type": "string", "jsonPath": ".status.phase"},
        {
            "name": "Completed",
            "type": "integer",
            "jsonPath": ".status.completedRuns",
        },
        {
            "name": "Total",
            "type": "integer",
            "jsonPath": ".status.maxTotalRuns",
        },
        {
            "name": "Failed",
            "type": "integer",
            "jsonPath": ".status.failedRuns",
        },
        {
            "name": "Current",
            "type": "string",
            "jsonPath": ".status.currentCell.label",
        },
        {"name": "Age", "type": "date", "jsonPath": ".metadata.creationTimestamp"},
    ]


def build_aiperfsweep_crd() -> dict[str, Any]:
    """Build the CRD dict for ``aiperfsweeps.aiperf.nvidia.com``.

    Derives ``spec`` from ``AIPerfSweepSpec.model_json_schema(by_alias=True)``
    so the CRD field names follow K8s camelCase conventions, then attaches CEL
    immutability rules to the orchestration-critical top-level spec fields
    (``sweep``, ``multiRun``, ``convergence``).

    Note: ``template.spec`` is intentionally a free-form object rather than the
    strict AIPerfJobSpec schema. Like ``spec.benchmark`` on AIPerfJob, the
    sweep controller normalizes/validates the AIPerfJobSpec stamp on its side
    via Pydantic, so the CRD passes it through with
    ``x-kubernetes-preserve-unknown-fields``.
    """
    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

    raw_schema = AIPerfSweepSpec.model_json_schema(mode="validation", by_alias=True)
    defs = raw_schema.get("$defs") or {}
    spec_schema = _convert_schema(raw_schema, defs)

    # AIPerfSweepSpec wraps AIPerfJobSpec.benchmark (an AIPerfConfig). Walk the
    # whole spec tree so every shape-detected node (benchmark, endpoint,
    # runtime, multiRun, artifacts) picks up the same CEL invariants that the
    # AIPerfJob CRD does.
    _walk_dict_apply(spec_schema, _ensure_type_on_preserve_unknown)
    _walk_dict_apply(spec_schema, _decorate_all)

    properties = spec_schema.setdefault("properties", {})

    # Tier 1C — AIPerfSweep axis-combination rules (mirrors
    # ``_validate_axis_combination`` in src/aiperf/kubernetes/sweep_models.py).
    # At least one orchestration axis must be set, and convergence pulls in
    # multiRun (without ``trials``) for cooldown/seed/warmup config.
    _add_validation_rules(
        spec_schema,
        (
            {
                "rule": (
                    "has(self.sweep) || has(self.multiRun) || has(self.convergence)"
                ),
                "message": (
                    "AIPerfSweep requires at least one of sweep/multiRun/"
                    "convergence; for a single benchmark use AIPerfJob via "
                    "`aiperf kube profile`"
                ),
            },
            {
                "rule": "!has(self.convergence) || has(self.multiRun)",
                "message": (
                    "convergence requires multiRun (for cooldown/seed/warmup config)"
                ),
            },
            {
                "rule": (
                    "!has(self.convergence) || !has(self.multiRun) || "
                    "!has(self.multiRun.trials)"
                ),
                "message": (
                    "multiRun.trials must be unset when convergence is set; "
                    "convergence.maxRuns governs the per-cell trial cap"
                ),
            },
        ),
    )

    # Tier 1E — convergence.minRuns <= maxRuns (mirrors
    # ``_validate_run_bounds`` in sweep_models.py).
    if "convergence" in properties:
        properties["convergence"].setdefault("type", "object")
        _add_validation_rules(
            properties["convergence"],
            (
                {
                    "rule": (
                        "!has(self.minRuns) || !has(self.maxRuns) || "
                        "self.minRuns <= self.maxRuns"
                    ),
                    "message": ("convergence.minRuns must be <= convergence.maxRuns"),
                },
            ),
        )

    # Tier 1D — forbid sweep/multiRun inside template.spec.benchmark.
    # The orchestration axes belong on AIPerfSweep.spec, not on the per-child
    # stamp. AIPerfJobSpec's extra='forbid' already rejects unknown keys at
    # template.spec, but AIPerfConfig's own ``sweep``/``multi_run`` fields
    # (which exist for non-k8s sweep CLI) would otherwise sneak through.
    template = properties.get("template")
    if isinstance(template, dict):
        template_spec = (template.get("properties") or {}).get("spec")
        if isinstance(template_spec, dict):
            template_benchmark = (template_spec.get("properties") or {}).get(
                "benchmark"
            )
            if isinstance(template_benchmark, dict):
                _add_validation_rules(
                    template_benchmark,
                    (
                        {
                            "rule": "!has(self.sweep)",
                            "message": (
                                "template.spec.benchmark.sweep is forbidden — "
                                "set spec.sweep at the AIPerfSweep top level "
                                "instead"
                            ),
                        },
                        {
                            "rule": "!has(self.multiRun)",
                            "message": (
                                "template.spec.benchmark.multiRun is "
                                "forbidden — set spec.multiRun at the "
                                "AIPerfSweep top level instead"
                            ),
                        },
                    ),
                )

    # Attach CEL immutability rules to top-level orchestration fields.
    # Re-running a sweep with mutated axes would corrupt the run-epoch ledger
    # and produce non-comparable cells, so the apiserver must reject mutation.
    # CEL rule construction needs a declared `type` for type inference; the
    # fields are object-shaped so this is sound.
    for field in ("sweep", "multiRun", "convergence"):
        if field in properties:
            properties[field].setdefault("type", "object")
            properties[field].setdefault("x-kubernetes-validations", []).append(
                {
                    "rule": "oldSelf == self",
                    "message": f"spec.{field} is immutable after creation",
                }
            )

    return {
        "apiVersion": "apiextensions.k8s.io/v1",
        "kind": "CustomResourceDefinition",
        "metadata": {
            "name": "aiperfsweeps.aiperf.nvidia.com",
            "annotations": {
                # Match AIPerfJob: keep the CRD when the helm release is
                # uninstalled so package-scoped test clusters don't see a
                # Terminating CRD between modules.
                "helm.sh/resource-policy": "keep",
            },
        },
        "spec": {
            "group": "aiperf.nvidia.com",
            "names": {
                "kind": "AIPerfSweep",
                "listKind": "AIPerfSweepList",
                "plural": "aiperfsweeps",
                "singular": "aiperfsweep",
                "shortNames": ["aps"],
            },
            "scope": "Namespaced",
            "versions": [
                {
                    "name": "v1alpha1",
                    "served": True,
                    "storage": True,
                    "additionalPrinterColumns": _aiperfsweep_printer_columns(),
                    "subresources": {"status": {}},
                    "schema": {
                        "openAPIV3Schema": {
                            "type": "object",
                            "required": ["spec"],
                            "properties": {
                                "spec": spec_schema,
                                "status": _aiperfsweep_status_schema(),
                            },
                        },
                    },
                },
            ],
        },
    }


# =============================================================================
# YAML Rendering
# =============================================================================


class _CRDDumper(yaml.SafeDumper):
    """Custom YAML dumper for CRD output."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> Any:
    """Use literal block style for multi-line strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _bool_representer(dumper: yaml.SafeDumper, data: bool) -> Any:
    """Represent bools as true/false (not True/False)."""
    return dumper.represent_scalar(
        "tag:yaml.org,2002:bool", "true" if data else "false"
    )


def _none_representer(dumper: yaml.SafeDumper, data: None) -> Any:
    """Represent None as empty mapping for status: {}."""
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_CRDDumper.add_representer(str, _str_representer)
_CRDDumper.add_representer(bool, _bool_representer)
_CRDDumper.add_representer(type(None), _none_representer)


def _escape_helm_braces(yaml_str: str) -> str:
    """Escape bare {{...}} in descriptions so Helm doesn't interpret them.

    Jinja2 template variables like {{prompt}} in Pydantic field descriptions
    would be parsed as Go template actions by Helm. Convert them to the
    Helm literal form: {{ "{{prompt}}" }}.
    """
    import re

    # Match {{word}} that is NOT already Helm-escaped (not preceded by {{ ")
    # and NOT a Helm directive (like {{- include ... }}).
    return re.sub(
        r'\{\{(?!\s*[-".])([\w]+)\}\}',
        r'{{ "{{\1}}" }}',
        yaml_str,
    )


def render_helm_crd_yaml(crd: dict[str, Any]) -> str:
    """Render the Helm-templated CRD variant."""
    helm_crd = copy.deepcopy(crd)

    yaml_str = yaml.dump(
        helm_crd,
        Dumper=_CRDDumper,
        default_flow_style=False,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )

    # Escape any literal `{{` / `}}` in description text (e.g. AIPerfConfig.variables
    # mentions Jinja2 `{{ ... }}` syntax) BEFORE adding our own Helm directives so
    # they don't get interpreted as Go template directives at chart render time.
    yaml_str = yaml_str.replace("{{", "\x00OPEN\x00").replace("}}", "\x00CLOSE\x00")
    yaml_str = yaml_str.replace("\x00OPEN\x00", '{{ "{{" }}').replace(
        "\x00CLOSE\x00", '{{ "}}" }}'
    )

    # Helm template substitutions
    yaml_str = yaml_str.replace(
        "default: nvcr.io/nvidia/aiperf:latest",
        "default: {{ .Values.defaults.image | quote }}",
    )

    yaml_str = yaml_str.replace(
        "  name: aiperfjobs.aiperf.nvidia.com\n",
        "  name: aiperfjobs.aiperf.nvidia.com\n"
        "  labels:\n"
        '    {{- include "aiperf-operator.labels" . | nindent 4 }}\n',
    )

    # Section comments for the nested schema.
    yaml_str = yaml_str.replace(
        "              connectionsPerWorker:\n",
        "              # -- Deployment fields (camelCase, K8s convention) ---------------\n"
        "              connectionsPerWorker:\n",
    )

    # Escape bare {{word}} in descriptions so Helm doesn't parse them.
    yaml_str = _escape_helm_braces(yaml_str)

    lines = list(SPDX_HEADER)
    lines.append(yaml_str.rstrip())
    return "\n".join(lines) + "\n"


def render_helm_sweep_crd_yaml(crd: dict[str, Any]) -> str:
    """Render the AIPerfSweep CRD as a Helm chart template.

    Sibling of :func:`render_helm_crd_yaml` for AIPerfJob. Reuses the same
    dumper and brace-escape logic, then injects the standard Helm labels block
    after the CRD ``metadata.name`` line.
    """
    helm_crd = copy.deepcopy(crd)

    yaml_str = yaml.dump(
        helm_crd,
        Dumper=_CRDDumper,
        default_flow_style=False,
        sort_keys=False,
        width=120,
        allow_unicode=True,
    )

    # Escape any literal `{{` / `}}` in description text — see render_helm_crd_yaml.
    yaml_str = yaml_str.replace("{{", "\x00OPEN\x00").replace("}}", "\x00CLOSE\x00")
    yaml_str = yaml_str.replace("\x00OPEN\x00", '{{ "{{" }}').replace(
        "\x00CLOSE\x00", '{{ "}}" }}'
    )

    yaml_str = yaml_str.replace(
        "  name: aiperfsweeps.aiperf.nvidia.com\n",
        "  name: aiperfsweeps.aiperf.nvidia.com\n"
        "  labels:\n"
        '    {{- include "aiperf-operator.labels" . | nindent 4 }}\n',
    )

    yaml_str = _escape_helm_braces(yaml_str)

    lines = list(SPDX_HEADER)
    lines.append(yaml_str.rstrip())
    return "\n".join(lines) + "\n"


# =============================================================================
# Generator
# =============================================================================


def _get_project_version() -> str:
    """Read the project version from pyproject.toml."""
    import tomllib

    with PYPROJECT_FILE.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _sync_chart_app_version(version: str) -> str:
    """Return Chart.yaml content with appVersion synced to pyproject.toml."""
    import re

    content = HELM_CHART_FILE.read_text()
    return re.sub(
        r'^appVersion:\s*".*"',
        f'appVersion: "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )


class CRDGenerator(Generator):
    """Generate Kubernetes CRD from AIPerfConfig schema."""

    name = "CRD Schema"
    description = "Generate Kubernetes CRD YAML from AIPerfConfig Pydantic model"

    def generate(self) -> GeneratorResult:
        sys.path.insert(0, "src")
        from aiperf.config.config import AIPerfConfig

        schema = AIPerfConfig.model_json_schema()
        if self.verbose:
            defs = schema.get("$defs", {})
            props = schema.get("properties", {})
            print_step(
                f"JSON Schema: {len(defs)} definitions, {len(props)} top-level properties"
            )

        config_properties = convert_aiperf_config_fields(schema, verbose=self.verbose)

        crd = _build_crd(config_properties)

        helm_yaml = render_helm_crd_yaml(crd)

        sweep_crd = build_aiperfsweep_crd()
        helm_sweep_yaml = render_helm_sweep_crd_yaml(sweep_crd)

        version = _get_project_version()
        chart_yaml = _sync_chart_app_version(version)

        field_count = len(config_properties)
        return GeneratorResult(
            files=[
                GeneratedFile(HELM_CRD_FILE, helm_yaml),
                GeneratedFile(HELM_SWEEP_CRD_FILE, helm_sweep_yaml),
                GeneratedFile(HELM_CHART_FILE, chart_yaml),
            ],
            summary=f"CRD with {field_count} AIPerfConfig fields + AIPerfSweep CRD",
        )


if __name__ == "__main__":
    main(CRDGenerator)
