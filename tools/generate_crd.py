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

PLUGINS_YAML = Path("src/aiperf/plugin/plugins.yaml")


def _form_data_endpoint_types() -> list[str]:
    """Endpoint types whose plugin metadata sets ``requires_form_data``.

    Read from the registry rather than hardcoded: the Pydantic gate
    (``endpoint.py::_validate_request_content_type``) is metadata-driven, so a
    literal list here silently diverges the moment another endpoint opts in --
    which is exactly what happened with image_edit.
    """
    data = yaml.safe_load(PLUGINS_YAML.read_text()) or {}
    endpoints = (data.get("plugins") or data).get("endpoint") or {}
    names = [
        name
        for name, entry in endpoints.items()
        if isinstance(entry, dict)
        and (entry.get("metadata") or {}).get("requires_form_data")
    ]
    return sorted(names)


HELM_CRD_FILE = Path("deploy/helm/aiperf-operator/templates/crd-aiperfjob.yaml")
HELM_SWEEP_CRD_FILE = Path("deploy/helm/aiperf-operator/templates/crd-aiperfsweep.yaml")
HELM_CHART_FILE = Path("deploy/helm/aiperf-operator/Chart.yaml")
PYPROJECT_FILE = Path("pyproject.toml")

SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
)

# Keys to strip from JSON Schema that K8s CRDs don't support
_STRIP_KEYS = frozenset({"title", "examples", "$defs", "$schema"})

# Internal marker on nodes that came from a mixed-type ``anyOf``/``oneOf``
# (e.g. ``list[X] | Literal[False]``). The post-pass that defaults
# ``type: object`` on preserve-unknown nodes skips these — neither object nor
# any primitive matches every legal value. Stripped before serialization.
_MIXED_UNION_SENTINEL = "_aiperf_mixed_union"

# Maximum recursion depth before falling back to preserve-unknown-fields.
# AIPerfSweep wraps an AIPerfJob in spec.template.spec, adding 3 levels;
# the deepest legitimate path today is spec.template.spec.benchmark.runtime.<field>
# at depth 6, but ModelItem and other inner classes go several levels deeper.
_MAX_DEPTH = 16


# =============================================================================
# JSON Schema -> K8s OpenAPI v3 Converter
# =============================================================================


class CRDSchemaSource:
    """Load raw Pydantic schemas used as CRD roots."""

    def config_schema(self) -> dict[str, Any]:
        from aiperf.config.config import AIPerfConfig

        return AIPerfConfig.model_json_schema()

    def job_schema(self) -> dict[str, Any]:
        from aiperf.kubernetes.crd_models import AIPerfJobSpec

        return AIPerfJobSpec.model_json_schema(mode="validation", by_alias=True)

    def sweep_schema(self) -> dict[str, Any]:
        from aiperf.kubernetes.crd_models import AIPerfSweepSpec

        return AIPerfSweepSpec.model_json_schema(mode="validation", by_alias=True)


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a $ref string to its definition."""
    name = ref.rsplit("/", 1)[-1]
    if name not in defs:
        return {}
    return defs[name]


def _anyof_has_non_object_alt(alts: list[dict[str, Any]]) -> bool:
    """True if any alternative is structurally not an object.

    Used to distinguish two flavors of mixed-type ``anyOf``/``oneOf``:

    * ``list[ConcurrencyPhase | PoissonPhase | ...]`` items — each
      alternative is a ``$ref`` to an object type. ``type: object`` +
      preserve-unknown is the right K8s shape.
    * ``list[Format] | Literal[False]`` — at least one alternative is an
      array or primitive. No single ``type`` matches every legal value, so
      the leaf must stay typeless (with preserve-unknown) and rely on the
      operator's Pydantic validation.
    """
    for alt in alts:
        t = alt.get("type")
        if t is not None and t != "object":
            return True
        if "const" in alt and not isinstance(alt["const"], dict):
            return True
    return False


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

    # ``x-kubernetes-preserve-unknown-fields: true`` set as ``json_schema_extra``
    # on a Pydantic field is the explicit author-side signal that the field is
    # polymorphic at the CRD layer (the Pydantic ``BeforeValidator`` accepts
    # string / list / object even though the type annotation is strict). Emit
    # a typeless polymorphic node with the mixed-union sentinel so admission
    # accepts every legal shape and the post-pass leaves it typeless. Without
    # this short-circuit the recursive walker would emit the resolved strict
    # shape (e.g. ``ModelsAdvanced.items``) and reject the list/string forms
    # that recipes use (``models: [<id>]``).
    if schema.get("x-kubernetes-preserve-unknown-fields") and (
        "$ref" in schema or "type" in schema or "properties" in schema
    ):
        result: dict[str, Any] = {
            "x-kubernetes-preserve-unknown-fields": True,
            _MIXED_UNION_SENTINEL: True,
        }
        if "description" in schema and schema["description"]:
            result["description"] = schema["description"]
        return result

    if "$ref" in schema:
        resolved = _resolve_ref(schema["$ref"], defs)
        merged = _convert_schema(resolved, defs, depth)
        if "description" in schema and schema["description"]:
            merged["description"] = schema["description"]
        # Preserve sibling x-kubernetes-preserve-unknown-fields markers
        # (Pydantic emits these alongside $ref via json_schema_extra to mark
        # narrow shorthand-accepting boundaries).
        if schema.get("x-kubernetes-preserve-unknown-fields"):
            merged["x-kubernetes-preserve-unknown-fields"] = True
        # Carry the sibling `default` across, as the nullable-anyOf branch
        # below already does. Enum-typed fields reach the CRD through a $ref
        # to their $defs entry, so without this every enum default
        # (urlStrategy, connectionReuse, logging.level, ...) was dropped and
        # `kubectl explain` disagreed with the model.
        if (
            "default" in schema
            and schema["default"] is not None
            and "default" not in merged
        ):
            merged["default"] = schema["default"]
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
            # ``Any | None`` (Pydantic emits anyOf:[{}, {type:null}]) means the
            # field genuinely accepts strings, lists, or objects — that's the
            # contract for shorthand siblings like ``model: "name"``,
            # ``model: ["a","b"]``, and ``warmup: {type: concurrency, ...}``.
            # Forcing ``type: object`` here would reject the scalar/list forms
            # at admission. Mark with the mixed-union sentinel so the post-pass
            # ``_ensure_type_on_preserve_unknown`` leaves the node typeless.
            if not real_type:
                result[_MIXED_UNION_SENTINEL] = True
                result.pop("type", None)
            else:
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

        # Mixed-type anyOf: split into "all alternatives are objects (or $ref
        # to objects)" vs. "at least one alternative is array or primitive".
        # The first kind (e.g. ``phases: list[ConcurrencyPhase | PoissonPhase
        # | ...]`` items) is structurally an object — emit
        # ``type: object`` + preserve-unknown so K8s accepts dict values and
        # CEL rules can compile. The second kind (e.g.
        # ``artifacts.summary: list[Format] | Literal[False]``) cannot be
        # coerced to a single type — leave typeless and tag with
        # ``_MIXED_UNION_SENTINEL`` so the post-pass doesn't restore
        # ``type: object`` (which would reject every legal value at admission).
        if _anyof_has_non_object_alt(any_of):
            result = {
                "x-kubernetes-preserve-unknown-fields": True,
                _MIXED_UNION_SENTINEL: True,
            }
        else:
            result = {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            }
        if "description" in schema:
            result["description"] = schema["description"]
        return result

    if "oneOf" in schema:
        # Same split as anyOf above.
        if _anyof_has_non_object_alt(schema["oneOf"]):
            result = {
                "x-kubernetes-preserve-unknown-fields": True,
                _MIXED_UNION_SENTINEL: True,
            }
        else:
            result = {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
            }
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
                    # Pydantic emits ``additionalProperties: true`` for ``dict[str, Any]``.
                    # K8s structural-schema strict-decode rejects unknown nested keys
                    # under such items unless ``x-kubernetes-preserve-unknown-fields:
                    # true`` is also set — ``additionalProperties: true`` alone is not
                    # honored by the strict decoder. Translate so PodTemplateConfig
                    # list-of-dict fields (volumes, env, volumeMounts, tolerations,
                    # initContainers, hostAliases, topologySpreadConstraints,
                    # imagePullSecrets) survive strict-decode of corev1-shaped sub-keys
                    # (claimName, valueFrom.secretKeyRef, etc.).
                    result["x-kubernetes-preserve-unknown-fields"] = True
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


class KubernetesSchemaConverter:
    """Convert Pydantic JSON Schema nodes into Kubernetes OpenAPI nodes."""

    def schema_node(
        self,
        schema: dict[str, Any],
        defs: dict[str, Any],
        depth: int = 0,
    ) -> dict[str, Any]:
        return _convert_schema(schema, defs, depth)

    def aiperf_config_fields(
        self, schema: dict[str, Any], *, verbose: bool = False
    ) -> dict[str, Any]:
        return convert_aiperf_config_fields(schema, verbose=verbose)


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
    ``properties``) so AIPerfSweep's ``spec.benchmark`` is fixed up via
    the same walker as the AIPerfJob top-level benchmark.

    Shorthand siblings (``model``, ``dataset``, ``warmup``, ``profiling``)
    accept scalar / list / object values and are emitted as typeless
    ``x-kubernetes-preserve-unknown-fields`` so admission accepts all three
    shapes. CEL cannot ``has()`` a typeless field — the apiserver refuses
    to install rules that reference one. The shorthand-or-canonical
    OR-requirement and the shorthand-and-canonical mutual exclusion
    therefore stay in ``normalize_before_validation`` in
    ``src/aiperf/config/config.py``; the operator surfaces them on
    reconcile via ``status.phase=Failed``.

    The cross-field rules referencing envelope-level fields
    (``parameter_sweep_same_seed`` requires ``random_seed``; dashboard UI
    incompatible with sweeps) used to live here too, but the envelope is
    flat, so ``self.sweep`` / ``self.multiRun`` / ``self.randomSeed``
    are not in scope from a ``benchmark`` node — apiserver rejects
    such rules at install time. Both checks remain enforced via Pydantic
    ``@model_validator`` decorators on ``AIPerfConfig`` (see
    ``validate_sweep_no_dashboard_ui`` in ``src/aiperf/config/config.py``).
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
            # Tier 1A shorthand rules skipped: ``model``/``dataset``/
            # ``warmup``/``profiling`` are typeless preserve-unknown siblings
            # (must accept scalar, list, or object). CEL ``has(self.X)`` won't
            # compile against a typeless field — the apiserver refuses the
            # CRD entirely. Shorthand-or-canonical OR-requirement and
            # shorthand-and-canonical mutual exclusion stay in
            # ``normalize_before_validation`` in
            # ``src/aiperf/config/config.py`` and surface as
            # ``status.phase=Failed`` after admission.
            #
            # Tier 4P/4Q/4R skipped: the array items for ``phases`` and
            # ``datasets`` are opaque (``x-kubernetes-preserve-unknown-fields``)
            # because their entries are heterogeneous Pydantic discriminated
            # unions. CEL can't dereference ``phases[].name``, ``datasets[].name``,
            # ``phases[].dataset``, or ``phases[0].seamless`` through opaque
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
    # `endpoint.type` must stay absent when the user omits it: the before-validator
    # in config/endpoint.py auto-sets it to 'template' only when `type` is not in
    # the payload. An apiserver-injected default of 'chat' would silently defeat
    # that and then trip the type-vs-template rule attached just below.
    if isinstance(props.get("type"), dict):
        props["type"].pop("default", None)
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
                # An omitted `type` is the documented shorthand: the Pydantic
                # validator auto-sets type='template' when a template is given
                # without one (config/endpoint.py::_auto_detect_template). The
                # CRD has no `default:` for `type`, so requiring has(self.type)
                # here rejected exactly the config the shorthand produces.
                "rule": (
                    "!has(self.template) || !has(self.type) || self.type == 'template'"
                ),
                "message": (
                    "endpoint.template is only used when endpoint.type='template' "
                    "(omit type to have it inferred)"
                ),
            },
            # Tier 2J — multipart/form-data only on endpoints whose plugin
            # metadata declares requires_form_data. Derived from plugins.yaml
            # so this cannot drift from the Pydantic gate.
            {
                "rule": (
                    "!has(self.requestContentType) || "
                    "self.requestContentType != 'multipart/form-data' || "
                    "!has(self.type) || self.type in "
                    + repr(_form_data_endpoint_types()).replace("'", "'")
                ),
                "message": (
                    "requestContentType='multipart/form-data' is only "
                    "supported on endpoint types that accept form data: "
                    + ", ".join(_form_data_endpoint_types())
                ),
            },
            # Tier 4O skipped: ``urls`` is a typeless preserve-unknown field
            # (recipes pass plain string URLs that the apiserver must accept
            # without structural validation). CEL ``self.urls.all(u, isURL(u))``
            # won't compile against a typeless field; URL well-formedness is
            # enforced by the Pydantic ``EndpointConfig`` validator.
            # Tier 4 — endpoint.path must be an absolute HTTP path.
            # A bare segment like ``v1/chat/completions`` (missing leading
            # slash) silently sends to the wrong URL and surfaces as a 404
            # at request time. Catching this at admission saves one round
            # of "why is my benchmark hitting the wrong endpoint" debugging.
            {
                "rule": "!has(self.path) || self.path.startsWith('/')",
                "message": (
                    "endpoint.path must start with '/' "
                    "(e.g. '/v1/chat/completions', not 'v1/chat/completions')"
                ),
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
            # Tier 1F — workersMin <= workers when both set, compared as ints.
            #
            # Both fields are int-or-string (a Helm value may arrive quoted),
            # which makes them `dyn` to CEL. A mixed int/string comparison has
            # no overload and the rule errors out instead of validating; a
            # string/string comparison silently compares lexicographically, so
            # {workersMin: "10", workers: "9"} passes. Casting both sides
            # through int() gives one well-defined comparison for every
            # combination.
            {
                "rule": (
                    "!has(self.workersMin) || !has(self.workers) || "
                    "int(self.workersMin) <= int(self.workers)"
                ),
                "message": ("runtime.workersMin must be <= runtime.workers"),
            },
        ),
    )


def _decorate_multirun_node(node: dict[str, Any]) -> None:
    """Attach CEL rules to AIPerfConfig.multiRun-shaped nodes.

    Detected by shape: ``numRuns`` + ``convergence`` (a combination unique to
    ``MultiRunConfig`` in ``src/aiperf/config/sweep/multi_run.py``).

    The detector previously required ``convergenceMetric`` + ``mode``, a flat
    shape MultiRunConfig has not had since convergence moved into its own
    nested ``ConvergenceConfig``: it matched nothing, so the decorator emitted
    no rule on either CRD. Its old rule is gone with the shape it referenced
    (the ``repeated``/``independent`` distinction now lives on the sweep, and
    ``ConvergenceConfig.mode`` means something else entirely). What replaces
    it mirrors the live ``_check_convergence_min_runs_le_num_runs`` validator.
    """
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if not isinstance(props, dict):
        return
    if not ("numRuns" in props and "convergence" in props):
        return
    _add_validation_rules(
        node,
        (
            {
                "rule": (
                    "!has(self.convergence) || !has(self.convergence.minRuns) || "
                    "!has(self.numRuns) || "
                    "self.convergence.minRuns <= self.numRuns"
                ),
                "message": (
                    "multiRun.convergence.minRuns must be <= multiRun.numRuns; "
                    "either lower minRuns or raise numRuns"
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


def _ensure_type_on_preserve_unknown(node: dict[str, Any]) -> None:
    """Default ``type: object`` on preserve-unknown nodes that aren't union escapes.

    K8s structural-schema validation rejects ``x-kubernetes-preserve-unknown-fields:
    true`` without a declared ``type``, AND CEL field access compiles only on
    nodes where the apiserver knows the type. The Pydantic→OpenAPI walker leaves
    a few branches typeless (mixed-type anyOf, oneOf, sibling markers on $refs),
    so this pass closes the gap before CRD apply.

    Skips nodes flagged with ``_MIXED_UNION_SENTINEL`` — those came from
    ``anyOf``/``oneOf`` of mixed scalar/array types (e.g.
    ``artifacts.summary: list[X] | Literal[False]``) where neither
    ``type: object`` nor any single primitive type matches every legal value.
    K8s accepts the typeless preserve-unknown leaf for these. The sentinel is
    stripped by ``_strip_mixed_union_sentinels`` before serialization.
    """
    if not isinstance(node, dict):
        return
    if node.get("x-kubernetes-preserve-unknown-fields") is not True:
        return
    if "type" in node:
        return
    if node.get(_MIXED_UNION_SENTINEL):
        return
    node["type"] = "object"


def _strip_mixed_union_sentinels(node: dict[str, Any]) -> None:
    """Remove the internal mixed-union sentinel before serialization."""
    if isinstance(node, dict):
        node.pop(_MIXED_UNION_SENTINEL, None)


def _walk_dict_apply(node: Any, fn: Any) -> None:
    """Depth-first traversal that applies ``fn`` to every dict node."""
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk_dict_apply(v, fn)
    elif isinstance(node, list):
        for item in node:
            _walk_dict_apply(item, fn)


class CRDSchemaEnhancer:
    """Apply AIPerf-specific CRD schema decorations."""

    def decorate_all(self, node: dict[str, Any]) -> None:
        _decorate_all(node)

    def ensure_type_on_preserve_unknown(self, node: dict[str, Any]) -> None:
        _ensure_type_on_preserve_unknown(node)

    def strip_internal_sentinels(self, node: dict[str, Any]) -> None:
        _strip_mixed_union_sentinels(node)


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
            "requestsCompleted": {
                "type": "integer",
                "format": "int64",
                "minimum": 0,
                "description": (
                    "Requests completed in the latest results-producing phase. "
                    "Stable top-level projection for kubectl printer columns."
                ),
            },
            "requestsTotal": {
                "type": "integer",
                "format": "int64",
                "minimum": 0,
                "description": (
                    "Requests expected in the latest results-producing phase."
                ),
            },
            "requestsPerSecond": {
                "type": "number",
                "description": (
                    "Current throughput for the latest results-producing phase. "
                    "Stable top-level projection for kubectl printer columns."
                ),
            },
            "currentPhase": {
                "type": "string",
                "description": "Current benchmark phase (warmup, profiling, etc)",
            },
            "subPhase": {
                "type": "string",
                "description": (
                    "Controller-side outer-lifecycle state, mirrored from the "
                    "controller pod's SystemState. Distinct from `phase` "
                    "(operator's view) and `currentPhase` (inner benchmark "
                    "stage): tracks the controller's internal state through "
                    "initializing -> configuring -> ready -> profiling -> "
                    "processing -> stopping -> shutdown. Cleared (field "
                    "removed) when the job reaches a terminal phase "
                    "(Completed/Failed/Cancelled)."
                ),
                "enum": [
                    "initializing",
                    "configuring",
                    "ready",
                    "profiling",
                    "processing",
                    "stopping",
                    "shutdown",
                ],
            },
            "liveMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Live metrics updated during benchmark run",
            },
            "serverMetrics": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Curated subset of server-side metrics from the inference server, written live by the controller as the dashboard's non-WebSocket fallback. Carries only the ~20 metric names the dashboard's server-metrics panel renders, and per series only its labels and the avg/max/rate/p99_estimate/count stats. Metrics over the AIPERF_SERVER_METRICS_CR_PROJECTION_MAX_SERIES / _MAX_LABELS caps are dropped whole, and the whole projection is dropped if it exceeds _MAX_BYTES. This field is a snapshot of the latest scrape, not an accumulation. Use the live WebSocket feed or server_metrics_export.json for the full payload.",
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
            "startupIssue": {
                "type": "object",
                "description": (
                    "Highest-priority pod startup blocker observed by the "
                    "operator. Removed when the blocker clears."
                ),
                "properties": {
                    "fingerprint": {"type": "string"},
                    "podName": {"type": "string"},
                    "containerName": {"type": "string"},
                    "reason": {"type": "string"},
                    "message": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "ContainerConfig",
                            "CrashLoop",
                            "ImagePull",
                            "SchedulingConstraint",
                            "SchedulingDelay",
                        ],
                    },
                    "terminalAfterThreshold": {"type": "boolean"},
                    "firstObservedTime": {
                        "type": "string",
                        "format": "date-time",
                    },
                    "warningEmitted": {"type": "boolean"},
                },
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
            "name": "Done",
            "type": "integer",
            "jsonPath": ".status.requestsCompleted",
            "description": "Requests completed in the active results-producing phase",
        },
        {
            "name": "Total",
            "type": "integer",
            "jsonPath": ".status.requestsTotal",
            "description": "Total expected requests in the active results-producing phase",
        },
        {
            "name": "QPS",
            "type": "number",
            "jsonPath": ".status.requestsPerSecond",
            "description": "Requests per second in the active results-producing phase",
        },
        {
            "name": "TPS",
            "type": "number",
            "jsonPath": ".status.summary.output_token_throughput.avg",
            "description": "Output tokens per second (status.summary.output_token_throughput.avg)",
        },
        {
            "name": "TTFT",
            "type": "number",
            "jsonPath": ".status.summary.time_to_first_token.p50",
            "description": "Time to first token P50 in ms (status.summary.time_to_first_token.p50)",
        },
        {
            "name": "ITL",
            "type": "number",
            "jsonPath": ".status.summary.inter_token_latency.p50",
            "description": "Inter-token latency P50 in ms (status.summary.inter_token_latency.p50)",
        },
        {
            "name": "E2E",
            "type": "number",
            "jsonPath": ".status.summary.request_latency.p50",
            "description": "End-to-end request latency P50 in ms (status.summary.request_latency.p50)",
        },
        {
            "name": "Age",
            "type": "date",
            "jsonPath": ".metadata.creationTimestamp",
        },
        # -o wide columns
        {
            "name": "TTFT_P99",
            "type": "number",
            "jsonPath": ".status.summary.time_to_first_token.p99",
            "description": "Time to first token P99 in ms",
            "priority": 1,
        },
        {
            "name": "ITL_P99",
            "type": "number",
            "jsonPath": ".status.summary.inter_token_latency.p99",
            "description": "Inter-token latency P99 in ms",
            "priority": 1,
        },
        {
            "name": "E2E_P99",
            "type": "number",
            "jsonPath": ".status.summary.request_latency.p99",
            "description": "End-to-end request latency P99 in ms",
            "priority": 1,
        },
        {
            "name": "TTFT_AVG",
            "type": "number",
            "jsonPath": ".status.summary.time_to_first_token.avg",
            "description": "Time to first token average in ms",
            "priority": 1,
        },
        {
            "name": "ITL_AVG",
            "type": "number",
            "jsonPath": ".status.summary.inter_token_latency.avg",
            "description": "Inter-token latency average in ms",
            "priority": 1,
        },
        {
            "name": "E2E_P99_9",
            "type": "number",
            "jsonPath": ".status.summary.request_latency.p99_9",
            "description": "End-to-end request latency P99.9 in ms",
            "priority": 1,
        },
    ]


# =============================================================================
# CRD Assembly
# =============================================================================


def _aiperf_job_spec_properties_from_schema(
    schema: dict[str, Any],
    converter: KubernetesSchemaConverter,
) -> dict[str, Any]:
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    return {
        name: converter.schema_node(prop_schema, defs, depth=0)
        for name, prop_schema in properties.items()
    }


def _aiperf_job_spec_properties() -> dict[str, Any]:
    """Generate ``.spec.*`` fields from ``AIPerfJobSpec`` (the validation model).

    AIPerfJobSpec — not DeploymentConfig — is the source of truth: the operator
    calls ``AIPerfJobSpec.model_validate(spec)`` in ``handlers/create.py``.
    The two models share most fields but differ on a handful of descriptions
    and constraints (e.g. ``ttl_seconds_after_finished`` carries ``ge=0`` only
    on AIPerfJobSpec), and AIPerfJobSpec adds ``skip_endpoint_check`` and
    ``benchmark``. This mirrors what the AIPerfSweep CRD already does via
    ``AIPerfSweepSpec.template.spec`` (which embeds AIPerfJobSpec).
    """
    return _aiperf_job_spec_properties_from_schema(
        CRDSchemaSource().job_schema(),
        KubernetesSchemaConverter(),
    )


def _tighten_image_schema(properties: dict[str, Any]) -> None:
    image = properties.get("image")
    if isinstance(image, dict):
        image["minLength"] = 1


def _allow_int_or_string(node: dict[str, Any]) -> None:
    """Mark a node as accepting either an int or a string.

    The apiserver rejects a schema carrying ``x-kubernetes-int-or-string``
    alongside ``type``, ``format`` or numeric bounds: the value is ``dyn``, so
    per-type constraints cannot apply. Leaving them in place makes the whole
    CRD structurally invalid, which fails ``helm install`` rather than any one
    field.
    """
    for key in (
        "type",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    ):
        node.pop(key, None)
    node["x-kubernetes-int-or-string"] = True


def _loosen_runtime_scalar_coercion_node(node: dict[str, Any]) -> None:
    properties = node.get("properties")
    if not isinstance(properties, dict):
        return
    if not {"workers", "apiPort"}.issubset(properties):
        return
    for key in ("workers", "workersMin", "apiPort"):
        field = properties.get(key)
        if isinstance(field, dict):
            _allow_int_or_string(field)


def _tighten_sweep_schema(properties: dict[str, Any]) -> None:
    sweep = properties.get("sweep")
    if not isinstance(sweep, dict):
        return
    sweep.setdefault("type", "object")
    sweep.setdefault("x-kubernetes-preserve-unknown-fields", True)
    sweep["required"] = ["type"]
    sweep.setdefault("properties", {}).update(
        {
            "type": {
                "type": "string",
                "enum": [
                    "grid",
                    "zip",
                    "scenarios",
                    "sobol",
                    "latin_hypercube",
                    "adaptive_search",
                ],
            },
            # GridSweep/ZipSweep name this `parameters` and require it. The
            # schema previously advertised `variables`, which no sweep model
            # accepts or aliases: a spec written against the published CRD was
            # admitted and then failed validation in the operator. Other sweep
            # kinds (scenarios/adaptive_search) carry different keys and rely
            # on x-kubernetes-preserve-unknown-fields above.
            "parameters": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"x-kubernetes-preserve-unknown-fields": True},
                },
            },
        }
    )


_AIPERFJOB_MUTABLE_SPEC_FIELDS = frozenset({"cancel", "timeoutSeconds"})
_AIPERFSWEEP_MUTABLE_SPEC_FIELDS = frozenset({"cancel", "ttlSecondsAfterFinished"})


def _immutable_spec_field_rule(field: str) -> str:
    """Return a presence-safe transition rule for one top-level spec field."""
    return (
        f"has(oldSelf.{field}) == has(self.{field}) && "
        f"(!has(self.{field}) || oldSelf.{field} == self.{field})"
    )


def _apply_workload_spec_immutability(
    spec_schema: dict[str, Any],
    *,
    kind: str,
    mutable_fields: frozenset[str],
) -> None:
    """Freeze every top-level workload field not reconciled after creation.

    The rules live on the spec node rather than the individual property node.
    Kubernetes does not evaluate a field-scoped transition rule when an
    optional field is added or removed, while the parent spec is present on
    every update. The explicit ``has`` parity therefore rejects value changes,
    first-set-after-create, and removal without restricting initial creation.

    This helper is deliberately called only by the two workload CRD builders;
    it is not a recursive shape detector because nested benchmark fields have
    different lifecycle semantics.
    """
    properties = spec_schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"{kind} spec schema has no properties")

    unknown_mutable = mutable_fields - properties.keys()
    if unknown_mutable:
        raise ValueError(
            f"{kind} mutable spec fields are absent from the schema: "
            f"{sorted(unknown_mutable)}"
        )

    rules = tuple(
        {
            "rule": _immutable_spec_field_rule(field),
            "message": (
                f"spec.{field} is immutable after creation; create a new "
                f"{kind} to change it"
            ),
        }
        for field in properties
        if field not in mutable_fields
    )
    _add_validation_rules(spec_schema, rules)


def _build_crd_from_job_spec_properties(
    job_spec_properties: dict[str, Any],
    enhancer: CRDSchemaEnhancer,
) -> dict[str, Any]:
    spec_properties: dict[str, Any] = {}

    # AIPerfJobSpec is the operator's `.spec` validation model — see
    # handlers/create.py. Walking it gives every deployment field plus
    # skip_endpoint_check and benchmark with the descriptions and constraints
    # the operator actually enforces.
    operator = copy.deepcopy(job_spec_properties)

    # Pop benchmark for separate handling (description override + decorator
    # walk). The AIPerfConfig sub-tree is reached via $ref from AIPerfJobSpec.
    benchmark_walked = operator.pop("benchmark")

    spec_properties["image"] = operator.pop("image")
    spec_properties["imagePullPolicy"] = operator.pop("imagePullPolicy")
    _tighten_image_schema(spec_properties)

    benchmark_walked["description"] = (
        "Benchmark workload (BenchmarkConfig). Strictly typed via the\n"
        "BenchmarkConfig schema, with x-kubernetes-preserve-unknown-fields: true\n"
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

    # Remaining AIPerfJobSpec fields (resourceMode, connectionsPerWorker,
    # timeoutSeconds, ..., skipEndpointCheck) in declared model order.
    for key, value in operator.items():
        spec_properties[key] = value

    # Apply every shape-detector decorator (relaxed-required + cross-field
    # CEL invariants) across the AIPerfConfig walk. Decorators detect their
    # target node by its property shape, so they fire on AIPerfJob's
    # spec.benchmark and on AIPerfSweep's spec.template.spec.benchmark from
    # the same call. See _decorate_all and the individual _decorate_* helpers.
    _walk_dict_apply(benchmark_walked, enhancer.ensure_type_on_preserve_unknown)
    _walk_dict_apply(benchmark_walked, enhancer.decorate_all)
    _walk_dict_apply(benchmark_walked, _loosen_runtime_scalar_coercion_node)
    _walk_dict_apply(benchmark_walked, enhancer.strip_internal_sentinels)

    # Decorators were applied to the benchmark sub-tree only, so top-level
    # orchestration nodes (multiRun in particular) picked up no CEL at all on
    # this kind while the AIPerfSweep builder -- which walks its whole spec --
    # decorated the identical node. Rules de-dupe by text, so re-walking the
    # already-walked benchmark sub-tree here is a no-op.
    _walk_dict_apply(spec_properties, enhancer.decorate_all)

    job_spec_schema: dict[str, Any] = {
        "type": "object",
        "description": (
            "AIPerfJob specification.\n"
            "\n"
            "spec.benchmark holds BenchmarkConfig fields (models, endpoint,\n"
            "datasets, phases, etc.) using camelCase aliases (urlStrategy,\n"
            "apiKey, readyCheckTimeout, …). The underlying Pydantic model\n"
            "also accepts the snake_case names used in AIPerf CLI YAML,\n"
            "but the apiserver schema only advertises camelCase.\n"
            "\n"
            "Top-level deployment fields (image, podTemplate, scheduling,\n"
            "etc.) use camelCase per Kubernetes API conventions."
        ),
        "properties": spec_properties,
        "required": ["benchmark"],
    }

    # Kind-specific cardinality: AIPerfJob.spec must NOT carry a sweep block.
    # AIPerfWorkloadSpec mixes in AIPerfConfig.sweep, so the field is present
    # in the schema; the rule rejects it at admission so users see the kind
    # error before the operator's @model_validator runs.
    _add_validation_rules(
        job_spec_schema,
        (
            {
                "rule": "!has(self.sweep)",
                "message": (
                    "AIPerfJob.spec.sweep must be null/omitted. Use "
                    "kind: AIPerfSweep for parameter sweeps."
                ),
            },
            {
                "rule": (
                    "!has(self.multiRun) || "
                    "((!has(self.multiRun.numRuns) || "
                    "self.multiRun.numRuns <= 1) && "
                    "!has(self.multiRun.convergence))"
                ),
                "message": (
                    "AIPerfJob.spec.multiRun must describe one run without "
                    "convergence. Use kind: AIPerfSweep for multi-run orchestration."
                ),
            },
        ),
    )
    _apply_workload_spec_immutability(
        job_spec_schema,
        kind="AIPerfJob",
        mutable_fields=_AIPERFJOB_MUTABLE_SPEC_FIELDS,
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
                # `kubectl get all` and `kubectl get aiperf` only surface a
                # CRD that declares itself in those categories. Dropped at the
                # kube1->kube2 seam with no commit recording the decision.
                "categories": ["all", "aiperf"],
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
                                "spec": job_spec_schema,
                                "status": _status_schema(),
                            },
                        },
                    },
                },
            ],
        },
    }


def _build_crd(_config_properties: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full CRD document."""
    return _build_crd_from_job_spec_properties(
        _aiperf_job_spec_properties(),
        CRDSchemaEnhancer(),
    )


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
            "observedGeneration": {
                "type": "integer",
                "format": "int64",
                "description": "Generation of the spec that was last processed",
            },
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
            "runStates": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Breakdown of child run states by phase: pending, running, completed, failed, cancelled.",
            },
            "currentChildRef": {
                "type": "object",
                "x-kubernetes-preserve-unknown-fields": True,
                "description": "Reference to the currently-active child: name, index, label.",
            },
            "apiUrl": {
                "type": "string",
                "description": "API endpoint URL for accessing sweep results and drill-down.",
            },
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


def _build_aiperfsweep_crd_from_schema(
    raw_schema: dict[str, Any],
    converter: KubernetesSchemaConverter,
    enhancer: CRDSchemaEnhancer,
) -> dict[str, Any]:
    defs = raw_schema.get("$defs") or {}
    spec_schema = converter.schema_node(raw_schema, defs)

    # AIPerfSweepSpec wraps AIPerfJobSpec.benchmark (an AIPerfConfig). Walk the
    # whole spec tree so every shape-detected node (benchmark, endpoint,
    # runtime, multiRun, artifacts) picks up the same CEL invariants that the
    # AIPerfJob CRD does.
    _walk_dict_apply(spec_schema, enhancer.ensure_type_on_preserve_unknown)
    _walk_dict_apply(spec_schema, enhancer.decorate_all)
    _walk_dict_apply(spec_schema, _loosen_runtime_scalar_coercion_node)
    _walk_dict_apply(spec_schema, enhancer.strip_internal_sentinels)

    properties = spec_schema.setdefault("properties", {})
    _tighten_image_schema(properties)
    _tighten_sweep_schema(properties)

    # Tier 1C — AIPerfSweep axis-combination rules (mirrors
    # ``_require_sweep_on_aiperfsweep`` in src/aiperf/operator/models.py).
    # ``sweep`` is required (kind-specific cardinality vs AIPerfJob).
    _add_validation_rules(
        spec_schema,
        (
            {
                "rule": "has(self.sweep)",
                "message": (
                    "AIPerfSweep.spec.sweep is required. Use kind: AIPerfJob "
                    "for single benchmarks."
                ),
            },
        ),
    )

    # Tier 1D removed: AIPerfJobSpec.benchmark is typed as BenchmarkConfig
    # (no sweep/multi_run fields), so the type system enforces this at the
    # apiserver level via the generated structural schema — no CEL needed.

    _apply_workload_spec_immutability(
        spec_schema,
        kind="AIPerfSweep",
        mutable_fields=_AIPERFSWEEP_MUTABLE_SPEC_FIELDS,
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
                "categories": ["all", "aiperf"],
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


def build_aiperfsweep_crd() -> dict[str, Any]:
    """Build the CRD dict for ``aiperfsweeps.aiperf.nvidia.com``.

    Derives ``spec`` from ``AIPerfSweepSpec.model_json_schema(by_alias=True)``
    so the CRD field names follow K8s camelCase conventions, then attaches CEL
    immutability rules to the orchestration-critical top-level spec fields
    (``sweep``, ``multiRun``).
    """
    return _build_aiperfsweep_crd_from_schema(
        CRDSchemaSource().sweep_schema(),
        KubernetesSchemaConverter(),
        CRDSchemaEnhancer(),
    )


class CRDDocumentBuilder:
    """Build complete Kubernetes CRD documents."""

    def __init__(
        self,
        *,
        converter: KubernetesSchemaConverter | None = None,
        enhancer: CRDSchemaEnhancer | None = None,
    ) -> None:
        self.converter = converter or KubernetesSchemaConverter()
        self.enhancer = enhancer or CRDSchemaEnhancer()

    def aiperfjob_crd(self, job_schema: dict[str, Any]) -> dict[str, Any]:
        job_spec_properties = _aiperf_job_spec_properties_from_schema(
            job_schema,
            self.converter,
        )
        return _build_crd_from_job_spec_properties(job_spec_properties, self.enhancer)

    def aiperfsweep_crd(self, sweep_schema: dict[str, Any]) -> dict[str, Any]:
        return _build_aiperfsweep_crd_from_schema(
            sweep_schema,
            self.converter,
            self.enhancer,
        )


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


def _apply_chart_default_substitutions(yaml_str: str) -> str:
    """Inject .Values-driven spec defaults shared by BOTH CRD templates.

    AIPerfJob and AIPerfSweep share the workload spec shape, so a chart
    deployed with a custom image/pull-policy must default them identically
    on both kinds — applying these to only one CRD silently gives the other
    kind's CRs the hardcoded nvcr.io fallback.

    - ``spec.image``: the Pydantic default (``nvcr.io/nvidia/aiperf:latest``)
      is replaced by the chart image (``.Values.defaults.image`` wins when set).
    - ``spec.imagePullPolicy``: no Pydantic default (None = defer to K8s), so
      inject a chart-controlled CRD default from
      ``.Values.defaults.imagePullPolicy``. `with` omits the default: line
      entirely when the value is unset/null, preserving the no-default
      behavior for charts that opt out.
    """
    yaml_str = yaml_str.replace(
        "default: nvcr.io/nvidia/aiperf:latest",
        'default: {{ default (printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion)) .Values.defaults.image | quote }}',
    )
    return yaml_str.replace(
        "              imagePullPolicy:\n                type: string\n",
        "              imagePullPolicy:\n"
        "                type: string\n"
        "                {{- with .Values.defaults.imagePullPolicy }}\n"
        "                default: {{ . | quote }}\n"
        "                {{- end }}\n",
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

    yaml_str = _apply_chart_default_substitutions(yaml_str)

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
    dumper, brace-escape logic, and chart-default substitutions
    (:func:`_apply_chart_default_substitutions`), then injects the standard
    Helm labels block after the CRD ``metadata.name`` line.
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

    yaml_str = _apply_chart_default_substitutions(yaml_str)

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


class CRDYAMLRenderer:
    """Render CRD documents into Helm-safe YAML."""

    def aiperfjob_yaml(self, crd: dict[str, Any]) -> str:
        return render_helm_crd_yaml(crd)

    def aiperfsweep_yaml(self, crd: dict[str, Any]) -> str:
        return render_helm_sweep_crd_yaml(crd)


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
        source = CRDSchemaSource()
        converter = KubernetesSchemaConverter()
        enhancer = CRDSchemaEnhancer()
        builder = CRDDocumentBuilder(converter=converter, enhancer=enhancer)
        renderer = CRDYAMLRenderer()

        config_schema = source.config_schema()
        if self.verbose:
            defs = config_schema.get("$defs", {})
            props = config_schema.get("properties", {})
            print_step(
                f"JSON Schema: {len(defs)} definitions, {len(props)} top-level properties"
            )

        config_properties = converter.aiperf_config_fields(
            config_schema,
            verbose=self.verbose,
        )

        crd = builder.aiperfjob_crd(source.job_schema())
        helm_yaml = renderer.aiperfjob_yaml(crd)

        sweep_crd = builder.aiperfsweep_crd(source.sweep_schema())
        helm_sweep_yaml = renderer.aiperfsweep_yaml(sweep_crd)

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
