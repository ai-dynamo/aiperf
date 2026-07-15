# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial validation tests for operator workload Pydantic specs.

Focuses on:
- AIPerfJobSpec/AIPerfSweepSpec sweep cardinality at the Python model boundary.
- Child metadata restrictions that prevent parent/identity fields from leaking to children.
- CamelCase and snake_case alias acceptance for K8s-facing deployment fields.
- Strict-vs-open subtree boundaries: typed config rejects typos, explicit escape hatches stay open.
- Non-finite sweep numeric rejection before values reach the sweep executor.

Out of scope:
- CRD schema/CEL generation: tests/unit/operator/test_aiperfsweep_crd_generation.py.
- Handler/runtime state machines: tests/unit/operator/test_sweep_handler_adversarial.py.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.operator.models import AIPerfJobSpec, AIPerfSweepSpec, AIPerfWorkloadSpec

# =============================================================================
# Helpers
# =============================================================================

_VALID_BENCHMARK: dict[str, object] = {
    "models": ["meta-llama/Llama-3-8B"],
    "endpoint": {"url": "http://vllm-router.aiperf-system:8000/v1/chat/completions"},
    "datasets": [
        {
            "name": "synthetic-main",
            "type": "synthetic",
            "entries": 4,
            "prompts": {"isl": 128, "osl": 32},
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

_VALID_SWEEP: dict[str, object] = {
    "type": "grid",
    "variables": {"benchmark.phases.profiling.concurrency": [1, 2]},
}


def _benchmark_with(**overrides: object) -> dict[str, object]:
    """Return a real validated benchmark baseline with adversarial overrides."""
    baseline = deepcopy(_VALID_BENCHMARK)
    baseline.update(overrides)
    return baseline


def _job_spec(**overrides: object) -> dict[str, object]:
    """Build a minimal AIPerfJobSpec dict with optional top-level overrides."""
    baseline: dict[str, object] = {
        "image": "nvcr.io/nvidia/aiperf:custom-test-tag",
        "benchmark": _benchmark_with(),
    }
    baseline.update(overrides)
    return baseline


def _sweep_spec(**overrides: object) -> dict[str, object]:
    """Build a minimal AIPerfSweepSpec dict with optional top-level overrides."""
    baseline: dict[str, object] = {
        "image": "nvcr.io/nvidia/aiperf:custom-test-tag",
        "benchmark": _benchmark_with(),
        "sweep": deepcopy(_VALID_SWEEP),
    }
    baseline.update(overrides)
    return baseline


# =============================================================================
# AIPerfJobSpec / AIPerfSweepSpec sweep cardinality
# =============================================================================


class TestSweepCardinality:
    """Kind-specific specs enforce exactly one owner for sweep orchestration."""

    @pytest.mark.parametrize(
        "sweep",
        [
            param(
                {"type": "grid", "variables": {"benchmark.phases.profiling.concurrency": [1, 2]}},
                id="grid",
            ),
            param(
                {"type": "scenarios", "runs": [{"name": "latency-baseline", "variables": {"target": "latency"}}]},
                id="scenarios",
            ),
            param(
                {
                    "type": "adaptive_search",
                    "searchSpace": [
                        {
                            "path": "phases.profiling.concurrency",
                            "kind": "int",
                            "lo": 1,
                            "hi": 4,
                        }
                    ],
                    "objectives": [
                        {
                            "metric": "output_token_throughput",
                            "direction": "maximize",
                        }
                    ],
                    "maxIterations": 3,
                    "nInitialPoints": 1,
                },
                id="adaptive-search",
            ),
        ],
    )  # fmt: skip
    def test_aiperfjob_spec_sweep_block_rejected_with_kind_guidance(
        self, sweep: dict[str, object]
    ) -> None:
        with pytest.raises(
            ValidationError,
            match=r"AIPerfJob\.spec\.sweep.*null.*AIPerfSweep",
        ):
            AIPerfJobSpec.model_validate(_job_spec(sweep=sweep))

    @pytest.mark.parametrize(
        "sweep_value",
        [
            param(None, id="explicit-null"),
            param("__missing__", id="missing"),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_missing_sweep_rejected_with_single_benchmark_guidance(
        self, sweep_value: object
    ) -> None:
        spec = _sweep_spec()
        if sweep_value == "__missing__":
            del spec["sweep"]
        else:
            spec["sweep"] = sweep_value

        with pytest.raises(
            ValidationError,
            match=r"AIPerfSweep\.spec\.sweep.*required.*AIPerfJob",
        ):
            AIPerfSweepSpec.model_validate(spec)


# =============================================================================
# Child metadata restrictions
# =============================================================================


class TestChildMetadataRestrictions:
    """Child metadata accepts labels/annotations only, not child identity fields."""

    def test_aiperfsweep_spec_child_metadata_defaults_to_empty_maps_when_present(
        self,
    ) -> None:
        spec = AIPerfSweepSpec.model_validate(_sweep_spec(childMetadata={}))

        assert spec.child_metadata is not None
        assert spec.child_metadata.labels == {}
        assert spec.child_metadata.annotations == {}

    @pytest.mark.parametrize(
        "forbidden_key",
        [
            param("name", id="name-managed-by-controller"),
            param("namespace", id="namespace-managed-by-parent"),
            param("uid", id="uid-generated-by-apiserver"),
            param("ownerReferences", id="owner-references-managed-by-controller"),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_child_metadata_identity_fields_rejected(
        self, forbidden_key: str
    ) -> None:
        child_metadata = {forbidden_key: "saturation-sweep-v00"}

        with pytest.raises(
            ValidationError, match=rf"(?i)childMetadata|{forbidden_key}|extra"
        ):
            AIPerfSweepSpec.model_validate(_sweep_spec(childMetadata=child_metadata))

    def test_aiperfjob_spec_child_metadata_rejected_with_field_name(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)childMetadata|extra"):
            AIPerfJobSpec.model_validate(
                _job_spec(childMetadata={"labels": {"team": "perf-platform"}})
            )


# =============================================================================
# Aliases / camelCase / deployment defaults
# =============================================================================


class TestAliasesAndDeploymentDefaults:
    """K8s-facing workload specs accept aliases and materialize safe defaults."""

    @pytest.mark.parametrize(
        "field_name,field_value,attribute_name,expected_value",
        [
            ("skipEndpointCheck", True, "skip_endpoint_check", True),
            ("ttlSecondsAfterFinished", 900, "ttl_seconds_after_finished", 900),
            ("resultsTtlDays", 30, "results_ttl_days", 30),
            param("imagePullPolicy", "IfNotPresent", "image_pull_policy", "IfNotPresent", id="camel-image-pull-policy"),
            param("resourceMode", "none", "resource_mode", "none", id="camel-resource-mode"),
            param("skip_endpoint_check", True, "skip_endpoint_check", True, id="snake-skip-endpoint-check"),
        ],
    )  # fmt: skip
    def test_aiperfworkload_spec_alias_input_sets_python_attribute(
        self,
        field_name: str,
        field_value: object,
        attribute_name: str,
        expected_value: object,
    ) -> None:
        spec = AIPerfJobSpec.model_validate(_job_spec(**{field_name: field_value}))

        assert getattr(spec, attribute_name) == expected_value

    def test_aiperfworkload_spec_deployment_defaults_are_burstable_and_bounded(
        self,
    ) -> None:
        spec = AIPerfJobSpec.model_validate({"benchmark": _benchmark_with()})

        assert spec.image == "nvcr.io/nvidia/aiperf:latest"
        assert spec.resource_mode == "burstable"
        assert spec.connections_per_worker == 100
        assert spec.timeout_seconds == 0
        assert spec.ttl_seconds_after_finished == 300
        assert spec.failure_policy.on_child_failure == "continue"
        assert spec.failure_policy.max_failures == 0


# =============================================================================
# Strict vs open subtree boundaries
# =============================================================================


class TestStrictAndOpenBoundaries:
    """Typed config rejects typos while explicit user-extension maps stay open."""

    @pytest.mark.parametrize(
        "extra_key",
        [
            param("failurePolciy", id="misspelled-failure-policy"),
            param("podTemplte", id="misspelled-pod-template"),
            param("skipEndpointChek", id="misspelled-skip-endpoint-check"),
        ],
    )  # fmt: skip
    def test_aiperfworkload_spec_top_level_typo_rejected_with_key_name(
        self, extra_key: str
    ) -> None:
        with pytest.raises(ValidationError, match=rf"(?i){extra_key}|extra|forbid"):
            AIPerfWorkloadSpec.model_validate(_job_spec(**{extra_key: True}))

    def test_aiperfworkload_spec_pod_template_typo_rejected_but_extra_pod_spec_allowed(
        self,
    ) -> None:
        good = AIPerfJobSpec.model_validate(
            _job_spec(
                podTemplate={
                    "labels": {"aiperf.nvidia.com/workload-tier": "latency"},
                    "extraPodSpec": {"schedulingGates": [{"name": "bench-ready"}]},
                }
            )
        )
        assert good.pod_template.extra_pod_spec == {
            "schedulingGates": [{"name": "bench-ready"}]
        }

        with pytest.raises(
            ValidationError, match=r"(?i)podTemplate|extraPodSpect|extra"
        ):
            AIPerfJobSpec.model_validate(
                _job_spec(podTemplate={"extraPodSpect": {"schedulingGates": []}})
            )

    def test_aiperfworkload_spec_benchmark_endpoint_extra_allows_request_payload_extensions(
        self,
    ) -> None:
        benchmark = _benchmark_with(
            endpoint={
                "url": "http://vllm-router.aiperf-system:8000/v1/chat/completions",
                "extra": {"guided_decoding_backend": "outlines", "temperature": 0},
            }
        )

        spec = AIPerfJobSpec.model_validate(_job_spec(benchmark=benchmark))

        assert spec.benchmark.endpoint.extra == {
            "guided_decoding_backend": "outlines",
            "temperature": 0,
        }


# =============================================================================
# Non-finite numeric rejection
# =============================================================================


class TestNonFiniteNumericRejection:
    """Sweep model validation rejects NaN/Inf before expansion or execution."""

    @pytest.mark.parametrize(
        "bad_value",
        [
            param(float("nan"), id="nan"),
            param(float("inf"), id="positive-infinity"),
            param(float("-inf"), id="negative-infinity"),
        ],
    )  # fmt: skip
    def test_aiperfsweep_spec_grid_variable_non_finite_rejected_with_path(
        self, bad_value: float
    ) -> None:
        sweep = {
            "type": "grid",
            "variables": {"benchmark.phases.profiling.concurrency": [1, bad_value]},
        }

        with pytest.raises(
            ValidationError,
            match=r"(?i)benchmark\.phases\.profiling\.concurrency.*finite|NaN|inf",
        ):
            AIPerfSweepSpec.model_validate(_sweep_spec(sweep=sweep))

    def test_aiperfworkload_spec_timeout_seconds_infinity_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"(?i)timeout|finite|inf"):
            AIPerfJobSpec.model_validate(_job_spec(timeoutSeconds=float("inf")))
