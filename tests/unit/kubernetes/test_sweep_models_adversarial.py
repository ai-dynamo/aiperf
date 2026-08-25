# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial spec-validation tests for AIPerfSweep (flat envelope).

Focuses on the strict-schema surface of the flat AIPerfSweepSpec envelope:

- Distribution bounds and optional-strict type (in benchmark.datasets)
- Sweep-axis key smuggling resistance
- extra='forbid' typo coverage on AIPerfSweepSpec and nested configs
- Mutual exclusivity + dependency rules between sweep/multiRun/benchmark

Out of scope (covered elsewhere):
- Handler-runtime tests: tests/unit/operator/test_sweep_handler_adversarial.py
- Positive validation paths: tests/unit/kubernetes/test_sweep_models.py
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.crd_models import AIPerfSweepSpec
from aiperf.kubernetes.sweep_models import (
    ObjectMetaPartial,
)

# ============================================================================
# Helpers
# ============================================================================

# Smallest benchmark dict that round-trips through AIPerfConfig validation.
# Reused across every test; mutate via dict-spread to make adversarial inputs.
_VALID_BENCHMARK = {
    "models": ["test-model"],
    "endpoint": {"url": "http://x"},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 8, "osl": 8},
        }
    ],
    "phases": [
        {
            "name": "default",
            "type": "concurrency",
            "kind": "profiling",
            "requests": 1,
            "concurrency": 1,
        }
    ],
}

_VALID_SWEEP = {
    "type": "grid",
    "parameters": {"phases.default.concurrency": [1, 2]},
}


def _benchmark_with(**overrides: object) -> dict:
    """Return a copy of the canonical benchmark dict with key overrides."""
    return {**_VALID_BENCHMARK, **overrides}


def _sweep(
    *,
    spec_extra: dict | None = None,
    benchmark: dict | None = None,
) -> dict:
    """Build a minimal AIPerfSweepSpec dict (flat envelope) with optional injection."""
    out: dict = {
        "sweep": _VALID_SWEEP,
        "multiRun": {"numRuns": 2},
        "image": "x:latest",
        "benchmark": benchmark if benchmark is not None else _VALID_BENCHMARK,
    }
    if spec_extra:
        out.update(spec_extra)
    return out


# ============================================================================
# Category 1 — Type confusion on typed fields
# ============================================================================


def test_image_as_number_rejected() -> None:
    """``image`` is a string; numbers must be rejected."""
    with pytest.raises(ValidationError, match=r"(?i)image|str"):
        AIPerfSweepSpec.model_validate(
            _sweep(spec_extra={"image": 12345}),
        )


def test_timeout_seconds_string_coerces() -> None:
    """``timeout_seconds: float`` accepts numeric strings (Pydantic default coercion).

    Documents observed behavior: AIPerfSweepSpec inherits AIPerfBaseModel which does
    NOT enable ``strict=True`` on its ConfigDict, so ``"600"`` coerces to 600.0.
    A non-numeric string is still rejected. If we ever turn on strict typing this
    test will flip.
    """
    spec = AIPerfSweepSpec.model_validate(
        _sweep(spec_extra={"timeoutSeconds": "600"}),
    )
    assert spec.timeout_seconds == 600.0

    with pytest.raises(ValidationError, match=r"(?i)timeout|float"):
        AIPerfSweepSpec.model_validate(
            _sweep(spec_extra={"timeoutSeconds": "not-a-number"}),
        )


def test_benchmark_as_string_rejected() -> None:
    """``benchmark`` must be a config object, not a scalar string."""
    with pytest.raises(ValidationError, match=r"(?i)benchmark|dict|object"):
        AIPerfSweepSpec.model_validate(
            _sweep(benchmark="just-a-name"),  # type: ignore[arg-type]
        )


# ============================================================================
# Category 2 — Boundary attacks on numeric ranges
# ============================================================================


@pytest.mark.parametrize(
    "num_runs, ok",
    [
        param(0, False, id="num-runs-zero-rejected"),
        param(-1, False, id="num-runs-negative-rejected"),
        param(11, False, id="num-runs-over-max-rejected"),
        param(100, False, id="num-runs-large-rejected"),
        param(1, True, id="num-runs-min-boundary-accepted"),
        param(10, True, id="num-runs-max-boundary-accepted"),
        param(5, True, id="num-runs-mid-accepted"),
    ],
)  # fmt: skip
def test_multirun_num_runs_range(num_runs: int, ok: bool) -> None:
    """``multi_run.num_runs`` is bounded ge=1, le=10."""
    data = _sweep()
    data["multiRun"] = {"numRuns": num_runs}
    if ok:
        spec = AIPerfSweepSpec.model_validate(data)
        assert spec.multi_run is not None
        assert spec.multi_run.num_runs == num_runs
    else:
        with pytest.raises(ValidationError):
            AIPerfSweepSpec.model_validate(data)


def test_multirun_cooldown_seconds_negative_rejected() -> None:
    """``cooldown_seconds`` is ge=0; tiny negatives still violate."""
    data = _sweep()
    data["multiRun"] = {"numRuns": 2, "cooldownSeconds": -0.001}
    with pytest.raises(ValidationError, match=r"(?i)cooldown|greater"):
        AIPerfSweepSpec.model_validate(data)


def test_failure_policy_max_failures_negative_rejected() -> None:
    """``failure_policy.max_failures`` is ge=0."""
    data = _sweep(spec_extra={"failurePolicy": {"maxFailures": -1}})
    with pytest.raises(ValidationError, match=r"(?i)max.?failures|greater"):
        AIPerfSweepSpec.model_validate(data)


def test_aiperfsweep_ttl_negative_rejected() -> None:
    """``AIPerfSweepSpec.ttl_seconds_after_finished`` is ge=0."""
    data = _sweep(spec_extra={"ttlSecondsAfterFinished": -1})
    with pytest.raises(ValidationError, match=r"(?i)ttl|greater"):
        AIPerfSweepSpec.model_validate(data)


# ============================================================================
# Category 3 — Key-typo and case-mutation attacks
# ============================================================================


@pytest.mark.parametrize(
    "extra_key",
    [
        param("multiRunn", id="multiRun-extra-n"),
        param("Sweep", id="sweep-capitalized"),
        param("multirun", id="multirun-no-camel"),
        param("multi_run_config", id="multirun-config-suffix"),
    ],
)  # fmt: skip
def test_aiperfsweep_top_level_typo_rejected(extra_key: str) -> None:
    """extra='forbid' on AIPerfSweepSpec catches arbitrary typos."""
    data = _sweep()
    data[extra_key] = {"numRuns": 1}
    with pytest.raises(
        ValidationError, match=r"(?i)extra|forbid|not permitted|unknown"
    ):
        AIPerfSweepSpec.model_validate(data)


def test_benchmark_field_typo_rejected() -> None:
    """``benchark:`` (missing m) on AIPerfSweepSpec is caught by extra=forbid."""
    data = {
        "sweep": _VALID_SWEEP,
        "multiRun": {"numRuns": 2},
        "image": "x:latest",
        "benchark": _VALID_BENCHMARK,
    }
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|benchark"):
        AIPerfSweepSpec.model_validate(data)


def test_pod_template_image_typo_rejected() -> None:
    """``imag`` typo inside the pod-template subtree is caught.

    podTemplate is a typed PodTemplateConfig; if its inner fields use
    extra='forbid' (the project default for BaseConfig subclasses) the typo
    reaches a forbid-extra error. This test verifies the strict surface really
    extends through nested config objects, not just the top of AIPerfSweepSpec.
    """
    bad_pod_template = {"workerImage": "x:latest", "imag": "should-not-be-here"}
    data = _sweep(spec_extra={"podTemplate": bad_pod_template})
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|imag"):
        AIPerfSweepSpec.model_validate(data)


def test_camelcase_required_alias_works() -> None:
    """``populate_by_name=True`` lets snake_case go through alongside camelCase.

    Contrast test for the typo cases above: ``image_pull_policy`` (snake) is
    accepted and stored on the model.
    """
    spec = AIPerfSweepSpec.model_validate(
        _sweep(spec_extra={"image_pull_policy": "IfNotPresent"}),
    )
    assert spec.image_pull_policy is not None
    assert spec.image_pull_policy.value == "IfNotPresent"


def test_camelcase_alias_form_works() -> None:
    """The camelCase alias ``imagePullPolicy`` is also accepted (sanity)."""
    spec = AIPerfSweepSpec.model_validate(
        _sweep(spec_extra={"imagePullPolicy": "Always"}),
    )
    assert spec.image_pull_policy is not None
    assert spec.image_pull_policy.value == "Always"


# ============================================================================
# Category 4 — Sweep-key smuggling
# ============================================================================


def test_sweep_smuggling_under_benchmark_rejected() -> None:
    """BenchmarkConfig has no ``sweep`` field; extra=forbid rejects it."""
    bench = _benchmark_with(sweep={"type": "grid", "parameters": {"x": [1, 2]}})
    data = _sweep(benchmark=bench)
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|sweep|not permitted"):
        AIPerfSweepSpec.model_validate(data)


def test_multirun_smuggling_under_benchmark_rejected() -> None:
    """BenchmarkConfig has no ``multi_run`` / ``multiRun`` field; rejected by extra=forbid."""
    bench = _benchmark_with(multi_run={"numRuns": 1})
    data = _sweep(benchmark=bench)
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|multi"):
        AIPerfSweepSpec.model_validate(data)


def test_runtime_sweep_smuggling_caught_by_runtime_extra_forbid() -> None:
    """Deeply nested ``benchmark.runtime.sweep`` doesn't slip through.

    ``runtime`` is a typed ``RuntimeConfig`` with ``extra='forbid'`` (project
    default for BaseConfig), so the apiserver still rejects the smuggled key.
    This test locks in the layered defense.
    """
    bench = _benchmark_with(
        runtime={"sweep": {"type": "grid", "parameters": {"x": [1, 2]}}}
    )
    data = _sweep(benchmark=bench)
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|sweep"):
        AIPerfSweepSpec.model_validate(data)


# ============================================================================
# Category 5 — Distribution bounds + type adversarial (in sweep context)
# ============================================================================


def _benchmark_with_isl(isl: object) -> dict:
    """Build a benchmark whose default-dataset prompts.isl is the given value."""
    return {
        **_VALID_BENCHMARK,
        "datasets": [
            {
                "name": "default",
                "type": "synthetic",
                "entries": 1,
                "prompts": {"isl": isl, "osl": 8},
            }
        ],
    }


def test_distribution_min_eq_max_accepted_in_sweep() -> None:
    """``min == max`` is degenerate-but-valid (clamps every sample to the same value)."""
    bench = _benchmark_with_isl({"mean": 100, "stddev": 30, "min": 100, "max": 100})
    spec = AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))
    isl = spec.benchmark.datasets[0].prompts.isl
    assert isl.min == isl.max == 100


def test_distribution_nan_min_rejected_in_sweep() -> None:
    """NaN bounds are non-finite — error must propagate through the full sweep validation."""
    bench = _benchmark_with_isl({"mean": 100, "stddev": 30, "min": float("nan")})
    with pytest.raises(ValidationError, match=r"(?i)finite|nan|isl|min"):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


def test_distribution_explicit_type_normal_with_value_rejected() -> None:
    """``type: normal`` + ``value:`` mismatches structure (value belongs to fixed)."""
    bench = _benchmark_with_isl({"type": "normal", "value": 5})
    with pytest.raises(ValidationError, match=r"(?i)mean|extra|forbid|normal"):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


def test_distribution_explicit_type_lognormal_with_stddev_rejected() -> None:
    """``type: lognormal`` requires median, not stddev."""
    bench = _benchmark_with_isl({"type": "lognormal", "mean": 100, "stddev": 30})
    with pytest.raises(
        ValidationError, match=r"(?i)median|stddev|extra|forbid|lognormal"
    ):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


def test_distribution_unknown_type_rejected_in_sweep() -> None:
    """``type: gaussian`` isn't a canonical distribution name.

    The discriminator raises a bare ``ValueError`` (not wrapped in
    ``ValidationError``) because Pydantic's discriminator-callable contract
    propagates the raw exception. Both forms are caught — what matters is that
    the input is rejected with a message naming the bad type.
    """
    bench = _benchmark_with_isl({"type": "gaussian", "mean": 100, "stddev": 30})
    with pytest.raises(
        (ValidationError, ValueError), match=r"(?i)gaussian|unknown|distribution|type"
    ):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


def test_distribution_min_gt_max_rejected_in_sweep() -> None:
    """``min > max`` is rejected — error path identifies the offending dataset/prompt."""
    bench = _benchmark_with_isl({"mean": 100, "stddev": 30, "min": 200, "max": 100})
    with pytest.raises(ValidationError, match=r"(?i)min|max|bounds"):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


# ============================================================================
# Category 6 — Mutual exclusivity + dependency rules
# ============================================================================


def test_no_sweep_set_rejected() -> None:
    """AIPerfSweep.spec.sweep is required by the kind-specific validator."""
    data = {
        "multiRun": {"numRuns": 1},
        "image": "x:latest",
        "benchmark": _VALID_BENCHMARK,
    }
    with pytest.raises(ValidationError, match=r"(?i)sweep is required"):
        AIPerfSweepSpec.model_validate(data)


def test_sweep_and_multirun_compose() -> None:
    """Sweep + multi_run is the standard composition path."""
    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": _VALID_SWEEP,
            "multiRun": {"numRuns": 2, "cooldownSeconds": 5},
            "image": "x:latest",
            "benchmark": _VALID_BENCHMARK,
        }
    )
    assert spec.sweep is not None
    assert spec.multi_run is not None
    assert spec.multi_run.num_runs == 2


def test_benchmark_with_both_model_and_models_rejected() -> None:
    """``model`` (singular) and ``models`` (plural) are mutually exclusive under benchmark.

    Mirrors the existing ``dataset``+``datasets`` mutex check in
    ``_check_mutual_exclusivity``.
    """
    bench = {**_VALID_BENCHMARK, "model": "from-singular"}
    with pytest.raises(ValidationError, match="'model' cannot be used with 'models'"):
        AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))


# ============================================================================
# Sanity guards
# ============================================================================


def test_object_meta_partial_rejects_top_level_typo_directly() -> None:
    """ObjectMetaPartial (still used inside podTemplate) enforces extra=forbid."""
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|name"):
        ObjectMetaPartial.model_validate({"name": "should-not-be-here"})
