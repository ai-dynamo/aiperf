# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial CRD validation tests for Kubernetes workload specs.

Focuses on:
- AIPerfJob vs AIPerfSweep kind-specific ``spec.sweep`` cardinality.
- CEL validation metadata that must reject the wrong kind at admission time.
- Preserve-unknown boundaries where CEL must not dereference opaque fields.
- Generated CRD schema consistency between both workload kinds.

Out of scope (covered elsewhere):
- Sweep model validator attack matrices: tests/unit/kubernetes/test_sweep_models_adversarial.py
- ``aiperf kube validate`` file I/O paths: tests/unit/kubernetes/test_validate.py
- Handler-runtime reconciliation: tests/unit/operator/test_sweep_handler_adversarial.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from pydantic import ValidationError
from pytest import param
from tools.generate_crd import _build_crd, build_aiperfsweep_crd

from aiperf.kubernetes.crd_models import AIPerfJobSpec, AIPerfSweepSpec

# =============================================================================
# Helpers
# =============================================================================

SchemaNode = dict[str, object]

_VALID_BENCHMARK: SchemaNode = {
    "models": ["meta-llama/Llama-3.1-8B-Instruct"],
    "endpoint": {
        "type": "chat",
        "urls": [
            "http://llm-gateway.aiperf.svc.cluster.local:8000/v1/chat/completions"
        ],
    },
    "datasets": [
        {
            "name": "main",
            "type": "synthetic",
            "entries": 8,
            "prompts": {"isl": 32, "osl": 16},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 4,
            "concurrency": 2,
        }
    ],
}

_VALID_SWEEP: SchemaNode = {
    "type": "grid",
    "parameters": {"phases.profiling.concurrency": [1, 2]},
}


def _benchmark_with(**overrides: object) -> SchemaNode:
    """Return a canonical benchmark dict with adversarial key overrides."""
    return {**_VALID_BENCHMARK, **overrides}


def _job_spec(**overrides: object) -> SchemaNode:
    """Build a minimal AIPerfJobSpec dict backed by real Pydantic validation."""
    return {
        "image": "nvcr.io/nvidia/aiperf:adversarial-crd",
        "benchmark": _VALID_BENCHMARK,
        **overrides,
    }


def _sweep_spec(**overrides: object) -> SchemaNode:
    """Build a minimal AIPerfSweepSpec dict backed by real Pydantic validation."""
    return {
        "image": "nvcr.io/nvidia/aiperf:adversarial-crd",
        "benchmark": _VALID_BENCHMARK,
        "sweep": _VALID_SWEEP,
        **overrides,
    }


def _openapi_schema(crd: SchemaNode) -> SchemaNode:
    versions = cast(list[SchemaNode], cast(SchemaNode, crd["spec"])["versions"])
    version = versions[0]
    schema = cast(SchemaNode, version["schema"])
    return cast(SchemaNode, schema["openAPIV3Schema"])


def _properties(node: SchemaNode) -> SchemaNode:
    return cast(SchemaNode, node["properties"])


def _spec_node(crd: SchemaNode) -> SchemaNode:
    return cast(SchemaNode, _properties(_openapi_schema(crd))["spec"])


def _job_spec_node() -> SchemaNode:
    return _spec_node(cast(SchemaNode, _build_crd({})))


def _sweep_spec_node() -> SchemaNode:
    return _spec_node(cast(SchemaNode, build_aiperfsweep_crd()))


def _validation_rules(node: SchemaNode) -> list[SchemaNode]:
    return cast(list[SchemaNode], node.get("x-kubernetes-validations", []))


def _rule_texts(node: SchemaNode) -> set[str]:
    return {cast(str, rule["rule"]) for rule in _validation_rules(node)}


def _message_texts(node: SchemaNode) -> set[str]:
    return {cast(str, rule["message"]) for rule in _validation_rules(node)}


def _benchmark_node(spec: SchemaNode) -> SchemaNode:
    return cast(SchemaNode, _properties(spec)["benchmark"])


def _endpoint_node(spec: SchemaNode) -> SchemaNode:
    return cast(SchemaNode, _properties(_benchmark_node(spec))["endpoint"])


def _runtime_node(spec: SchemaNode) -> SchemaNode:
    return cast(SchemaNode, _properties(_benchmark_node(spec))["runtime"])


def _immutable_spec_field_rule(field: str) -> str:
    return (
        f"has(oldSelf.{field}) == has(self.{field}) && "
        f"(!has(self.{field}) || oldSelf.{field} == self.{field})"
    )


# =============================================================================
# Kind-specific Pydantic cardinality
# =============================================================================


class TestWorkloadKindPydanticCardinality:
    """Real model builds prove AIPerfJob and AIPerfSweep do not drift apart."""

    def test_aiperfjob_spec_without_sweep_validates_as_single_benchmark(self) -> None:
        spec = AIPerfJobSpec.model_validate(_job_spec())

        assert spec.sweep is None
        assert spec.benchmark.get_model_names() == ["meta-llama/Llama-3.1-8B-Instruct"]

    def test_aiperfjob_spec_with_sweep_block_rejected_with_kind_guidance(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"AIPerfJob\.spec\.sweep.*AIPerfSweep",
        ):
            AIPerfJobSpec.model_validate(_job_spec(sweep=_VALID_SWEEP))

    def test_aiperfsweep_spec_without_sweep_block_rejected_with_kind_guidance(
        self,
    ) -> None:
        with pytest.raises(
            ValidationError,
            match=r"AIPerfSweep\.spec\.sweep is required.*AIPerfJob",
        ):
            AIPerfSweepSpec.model_validate(_sweep_spec(sweep=None))

    def test_aiperfsweep_child_metadata_allowed_but_aiperfjob_rejects_it(self) -> None:
        child_metadata = {
            "labels": {"team": "perf-platform"},
            "annotations": {"aiperf.nvidia.com/runbook": "nightly-sweep"},
        }

        sweep = AIPerfSweepSpec.model_validate(
            _sweep_spec(childMetadata=child_metadata)
        )
        assert sweep.child_metadata is not None
        assert sweep.child_metadata.labels == {"team": "perf-platform"}

        with pytest.raises(ValidationError, match=r"(?i)childMetadata|extra|forbid"):
            AIPerfJobSpec.model_validate(_job_spec(childMetadata=child_metadata))

    def test_aiperfsweep_benchmark_envelope_key_smuggling_rejected(self) -> None:
        benchmark = _benchmark_with(sweep=_VALID_SWEEP)

        with pytest.raises(ValidationError, match=r"(?i)benchmark.*sweep|extra|forbid"):
            AIPerfSweepSpec.model_validate(_sweep_spec(benchmark=benchmark))


class TestWorkloadSpecUpdateContract:
    """Only controls that live reconcilers reread remain mutable."""

    @pytest.mark.parametrize(
        "model,spec_builder,kind,mutable_fields",
        [
            param(
                AIPerfJobSpec,
                _job_spec_node,
                "AIPerfJob",
                {"cancel", "timeoutSeconds"},
                id="job",
            ),
            param(
                AIPerfSweepSpec,
                _sweep_spec_node,
                "AIPerfSweep",
                {"cancel", "ttlSecondsAfterFinished"},
                id="sweep",
            ),
        ],
    )  # fmt: skip
    def test_every_create_time_field_has_presence_safe_immutability_rule(
        self,
        model: type[AIPerfJobSpec] | type[AIPerfSweepSpec],
        spec_builder: Callable[[], SchemaNode],
        kind: str,
        mutable_fields: set[str],
    ) -> None:
        spec = spec_builder()
        properties = set(_properties(spec))
        model_aliases = {
            field.alias or name for name, field in model.model_fields.items()
        }
        assert properties == model_aliases

        rules = _rule_texts(spec)
        messages = _message_texts(spec)
        for field in properties - mutable_fields:
            assert _immutable_spec_field_rule(field) in rules
            assert (
                f"spec.{field} is immutable after creation; create a new "
                f"{kind} to change it"
            ) in messages

        for field in mutable_fields:
            assert _immutable_spec_field_rule(field) not in rules

    @pytest.mark.parametrize(
        "spec_builder,field",
        [
            param(_job_spec_node, "benchmark", id="job-value-change"),
            param(_job_spec_node, "resultsTtlDays", id="job-first-set"),
            param(_sweep_spec_node, "benchmark", id="sweep-value-change"),
            param(_sweep_spec_node, "childMetadata", id="sweep-first-set"),
        ],
    )  # fmt: skip
    def test_optional_add_remove_cannot_bypass_parent_scoped_rule(
        self, spec_builder: Callable[[], SchemaNode], field: str
    ) -> None:
        spec = spec_builder()

        assert _immutable_spec_field_rule(field) in _rule_texts(spec)

    @pytest.mark.parametrize(
        "spec_builder,mutable_fields",
        [
            param(_job_spec_node, {"cancel", "timeoutSeconds"}, id="job"),
            param(
                _sweep_spec_node,
                {"cancel", "ttlSecondsAfterFinished"},
                id="sweep",
            ),
        ],
    )  # fmt: skip
    def test_mutable_controls_have_no_old_self_transition_rule(
        self,
        spec_builder: Callable[[], SchemaNode],
        mutable_fields: set[str],
    ) -> None:
        rules = _rule_texts(spec_builder())

        for field in mutable_fields:
            assert all(f"oldSelf.{field}" not in rule for rule in rules)


# =============================================================================
# Generated CRD kind-specific CEL contracts
# =============================================================================


class TestGeneratedCrdKindSpecificCel:
    """Admission-layer rules must mirror the kind-specific Pydantic validators."""

    @pytest.mark.parametrize(
        "spec_builder,expected_rule,expected_message",
        [
            param(
                _job_spec_node,
                "!has(self.sweep)",
                "AIPerfJob.spec.sweep must be null/omitted. Use kind: AIPerfSweep for parameter sweeps.",
                id="aiperfjob-forbids-sweep",
            ),
            param(
                _job_spec_node,
                "!has(self.multiRun) || ((!has(self.multiRun.numRuns) || self.multiRun.numRuns <= 1) && !has(self.multiRun.convergence))",
                "AIPerfJob.spec.multiRun must describe one run without convergence. Use kind: AIPerfSweep for multi-run orchestration.",
                id="aiperfjob-forbids-multi-run-orchestration",
            ),
            param(
                _sweep_spec_node,
                "has(self.sweep)",
                "AIPerfSweep.spec.sweep is required. Use kind: AIPerfJob for single benchmarks.",
                id="aiperfsweep-requires-sweep",
            ),
        ],
    )  # fmt: skip
    def test_spec_node_carries_kind_specific_sweep_rule(
        self,
        spec_builder: Callable[[], SchemaNode],
        expected_rule: str,
        expected_message: str,
    ) -> None:
        spec = spec_builder()

        assert expected_rule in _rule_texts(spec)
        assert expected_message in _message_texts(spec)

    def test_aiperfjob_declares_sweep_only_so_cel_can_forbid_it(self) -> None:
        spec = _job_spec_node()
        props = _properties(spec)

        assert "sweep" in props
        assert cast(SchemaNode, props["sweep"])["type"] == "object"
        assert "sweep" not in cast(list[str], spec.get("required", []))
        assert "!has(self.sweep)" in _rule_texts(spec)

    def test_aiperfsweep_sweep_schema_requires_type_without_spec_required_trap(
        self,
    ) -> None:
        spec = _sweep_spec_node()
        sweep = cast(SchemaNode, _properties(spec)["sweep"])

        assert "sweep" not in cast(list[str], spec.get("required", []))
        assert sweep["required"] == ["type"]
        assert sweep["type"] == "object"
        assert sweep["x-kubernetes-preserve-unknown-fields"] is True
        assert "has(self.sweep)" in _rule_texts(spec)

    def test_aiperfsweep_orchestration_axes_are_immutable_and_typed_for_cel(
        self,
    ) -> None:
        spec = _sweep_spec_node()
        props = _properties(spec)
        rules = _rule_texts(spec)

        for field in ("sweep", "multiRun"):
            node = cast(SchemaNode, props[field])
            assert node["type"] == "object"
            assert _immutable_spec_field_rule(field) in rules


# =============================================================================
# Preserve-unknown and CEL compile-safety gotchas
# =============================================================================


class TestPreserveUnknownCelBoundaries:
    """Opaque fields stay opaque; CEL must not reference what apiserver cannot see."""

    @pytest.mark.parametrize(
        "shortcut",
        [
            "model",
            "dataset",
            "warmup",
            "profiling",
        ],
    )  # fmt: skip
    def test_benchmark_shorthand_siblings_are_typeless_preserve_unknown(
        self, shortcut: str
    ) -> None:
        shortcut_node = cast(
            SchemaNode, _properties(_benchmark_node(_job_spec_node()))[shortcut]
        )

        assert shortcut_node["x-kubernetes-preserve-unknown-fields"] is True
        assert "type" not in shortcut_node

    @pytest.mark.parametrize(
        "spec_builder",
        [
            param(_job_spec_node, id="aiperfjob"),
            param(_sweep_spec_node, id="aiperfsweep"),
        ],
    )  # fmt: skip
    def test_benchmark_rules_do_not_reference_typeless_shorthand_fields(
        self, spec_builder: Callable[[], SchemaNode]
    ) -> None:
        benchmark = _benchmark_node(spec_builder())
        rules = _rule_texts(benchmark)

        forbidden_fragments = (
            "has(self.model)",
            "has(self.dataset)",
            "has(self.warmup)",
            "has(self.profiling)",
        )
        assert not any(
            fragment in rule for fragment in forbidden_fragments for rule in rules
        )

    @pytest.mark.parametrize(
        "array_field",
        [
            "datasets",
            "phases",
        ],
    )  # fmt: skip
    def test_benchmark_union_array_items_are_opaque_preserve_unknown(
        self, array_field: str
    ) -> None:
        array_node = cast(
            SchemaNode, _properties(_benchmark_node(_job_spec_node()))[array_field]
        )
        items = cast(SchemaNode, array_node["items"])

        assert array_node["type"] == "array"
        assert items["type"] == "object"
        assert items["x-kubernetes-preserve-unknown-fields"] is True
        assert "properties" not in items

    def test_benchmark_rules_do_not_dereference_opaque_array_items(self) -> None:
        rules = _rule_texts(_benchmark_node(_job_spec_node()))
        forbidden_fragments = (
            "self.phases.all",
            "self.datasets.all",
            "self.phases[0]",
            ".dataset",
            ".seamless",
        )

        assert not any(
            fragment in rule for fragment in forbidden_fragments for rule in rules
        )


# =============================================================================
# Generated schema consistency between workload kinds
# =============================================================================


class TestGeneratedCrdSchemaConsistency:
    """Shared workload fields must stay identical where both kinds promise parity."""

    def test_shared_benchmark_endpoint_rules_match_between_job_and_sweep(self) -> None:
        job_endpoint = _endpoint_node(_job_spec_node())
        sweep_endpoint = _endpoint_node(_sweep_spec_node())

        assert _rule_texts(sweep_endpoint) == _rule_texts(job_endpoint)
        assert _message_texts(sweep_endpoint) == _message_texts(job_endpoint)

    def test_shared_runtime_rules_match_between_job_and_sweep(self) -> None:
        job_runtime = _runtime_node(_job_spec_node())
        sweep_runtime = _runtime_node(_sweep_spec_node())

        assert _rule_texts(sweep_runtime) == _rule_texts(job_runtime)
        assert _message_texts(sweep_runtime) == _message_texts(job_runtime)

    def test_shared_benchmark_schema_keeps_same_strict_and_opaque_boundaries(
        self,
    ) -> None:
        job_props = _properties(_benchmark_node(_job_spec_node()))
        sweep_props = _properties(_benchmark_node(_sweep_spec_node()))

        for field in ("endpoint", "runtime", "models", "datasets", "phases"):
            assert sweep_props[field] == job_props[field]

    def test_generated_crds_do_not_leak_internal_mixed_union_sentinel(self) -> None:
        sentinel = "_aiperf_mixed_union"

        def walk(node: object) -> bool:
            if isinstance(node, dict):
                return sentinel in node or any(walk(value) for value in node.values())
            if isinstance(node, list):
                return any(walk(value) for value in node)
            return False

        assert walk(_build_crd({})) is False
        assert walk(build_aiperfsweep_crd()) is False

    def test_aiperfjob_only_fields_do_not_leak_to_aiperfsweep_only_surface(
        self,
    ) -> None:
        job_names = cast(SchemaNode, cast(SchemaNode, _build_crd({})["spec"])["names"])
        sweep_names = cast(
            SchemaNode, cast(SchemaNode, build_aiperfsweep_crd()["spec"])["names"]
        )

        assert job_names["kind"] == "AIPerfJob"
        assert job_names["plural"] == "aiperfjobs"
        assert job_names["shortNames"] == ["apj", "aiperf"]
        assert sweep_names["kind"] == "AIPerfSweep"
        assert sweep_names["plural"] == "aiperfsweeps"
        assert sweep_names["shortNames"] == ["aps"]
