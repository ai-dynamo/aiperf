# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial spec-validation tests for AIPerfSweep.

Focuses on the strict-schema surface added on ajc/k8s:
- ObjectMetaPartial / AIPerfJobSpec typing on AIPerfJobTemplate
- Distribution bounds and optional-strict type
- Sweep-axis key smuggling resistance
- extra='forbid' typo coverage

Out of scope (covered elsewhere):
- Handler-runtime tests: tests/unit/operator/test_sweep_handler_adversarial.py
- Positive validation paths: tests/unit/kubernetes/test_sweep_models.py
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.sweep_models import (
    AIPerfJobTemplate,
    AIPerfSweepSpec,
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
            "requests": 1,
            "concurrency": 1,
        }
    ],
}


def _benchmark_with(**overrides: object) -> dict:
    """Return a copy of the canonical benchmark dict with key overrides."""
    return {**_VALID_BENCHMARK, **overrides}


def _sweep(
    *,
    spec_extra: dict | None = None,
    template_spec_extra: dict | None = None,
    benchmark: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a minimal AIPerfSweepSpec dict with optional injection points."""
    template_spec: dict = {
        "benchmark": benchmark if benchmark is not None else _VALID_BENCHMARK
    }
    if template_spec_extra:
        template_spec.update(template_spec_extra)
    template: dict = {"spec": template_spec}
    if metadata is not None:
        template["metadata"] = metadata
    sweep: dict = {
        "multiRun": {"trials": 2},
        "template": template,
    }
    if spec_extra:
        sweep.update(spec_extra)
    return sweep


# ============================================================================
# Category 1 — Type confusion on typed fields
# ============================================================================


def test_metadata_labels_as_list_rejected() -> None:
    """``metadata.labels`` must be dict, not list."""
    with pytest.raises(ValidationError, match=r"(?i)labels"):
        AIPerfSweepSpec.model_validate(
            _sweep(metadata={"labels": ["team=perf"]}),
        )


def test_metadata_labels_non_string_value_rejected() -> None:
    """``metadata.labels`` is dict[str, str]; ints are rejected for values."""
    with pytest.raises(ValidationError, match=r"(?i)labels|str"):
        AIPerfSweepSpec.model_validate(
            _sweep(metadata={"labels": {"team": 5}}),
        )


def test_metadata_annotations_non_string_value_rejected() -> None:
    """Annotations also typed dict[str, str]."""
    with pytest.raises(ValidationError, match=r"(?i)annotation|str"):
        AIPerfSweepSpec.model_validate(
            _sweep(metadata={"annotations": {"note": 1.5}}),
        )


def test_metadata_as_string_rejected() -> None:
    """``metadata`` is an object, not a string scalar."""
    with pytest.raises(ValidationError, match=r"(?i)metadata|dict|object"):
        AIPerfSweepSpec.model_validate(_sweep(metadata="just-a-string"))  # type: ignore[arg-type]


def test_template_spec_image_as_number_rejected() -> None:
    """``template.spec.image`` is a string; numbers must be rejected."""
    with pytest.raises(ValidationError, match=r"(?i)image|str"):
        AIPerfSweepSpec.model_validate(
            _sweep(template_spec_extra={"image": 12345}),
        )


def test_template_spec_timeout_seconds_string_coerces() -> None:
    """``timeout_seconds: float`` accepts numeric strings (Pydantic default coercion).

    Documents observed behavior: AIPerfJobSpec inherits AIPerfBaseModel which does
    NOT enable ``strict=True`` on its ConfigDict, so ``"600"`` coerces to 600.0.
    A non-numeric string is still rejected. If we ever turn on strict typing this
    test will flip.
    """
    spec = AIPerfSweepSpec.model_validate(
        _sweep(template_spec_extra={"timeoutSeconds": "600"}),
    )
    assert spec.template.spec.timeout_seconds == 600.0

    with pytest.raises(ValidationError, match=r"(?i)timeout|float"):
        AIPerfSweepSpec.model_validate(
            _sweep(template_spec_extra={"timeoutSeconds": "not-a-number"}),
        )


def test_template_spec_benchmark_as_string_rejected() -> None:
    """``benchmark`` must be a config object, not a scalar string."""
    with pytest.raises(ValidationError, match=r"(?i)benchmark|dict|object"):
        AIPerfSweepSpec.model_validate(
            _sweep(benchmark="just-a-name"),  # type: ignore[arg-type]
        )


# ============================================================================
# Category 2 — Boundary attacks on numeric ranges
# ============================================================================


@pytest.mark.parametrize(
    "trials, ok",
    [
        param(0, False, id="trials-zero-rejected"),
        param(-1, False, id="trials-negative-rejected"),
        param(21, False, id="trials-over-max-rejected"),
        param(100, False, id="trials-large-rejected"),
        param(1, True, id="trials-min-boundary-accepted"),
        param(20, True, id="trials-max-boundary-accepted"),
        param(10, True, id="trials-mid-accepted"),
    ],
)  # fmt: skip
def test_multirun_trials_range(trials: int, ok: bool) -> None:
    """``multi_run.trials`` is bounded ge=1, le=20."""
    data = _sweep()
    data["multiRun"] = {"trials": trials}
    if ok:
        spec = AIPerfSweepSpec.model_validate(data)
        assert spec.multi_run is not None
        assert spec.multi_run.trials == trials
    else:
        with pytest.raises(ValidationError):
            AIPerfSweepSpec.model_validate(data)


def test_multirun_cooldown_seconds_negative_rejected() -> None:
    """``cooldown_seconds`` is ge=0; tiny negatives still violate."""
    data = _sweep()
    data["multiRun"] = {"trials": 2, "cooldownSeconds": -0.001}
    with pytest.raises(ValidationError, match=r"(?i)cooldown|greater"):
        AIPerfSweepSpec.model_validate(data)


@pytest.mark.parametrize(
    "threshold, ok",
    [
        param(0.0, False, id="threshold-zero-rejected"),
        param(1.0, False, id="threshold-one-rejected"),
        param(-0.1, False, id="threshold-negative-rejected"),
        param(1.5, False, id="threshold-over-one-rejected"),
        param(0.5, True, id="threshold-half-accepted"),
        param(0.001, True, id="threshold-tiny-accepted"),
        param(0.999, True, id="threshold-near-one-accepted"),
    ],
)  # fmt: skip
def test_convergence_threshold_range(threshold: float, ok: bool) -> None:
    """``convergence.threshold`` is gt=0 and lt=1 (open interval)."""
    data = _sweep()
    data["multiRun"] = {"cooldownSeconds": 1}
    data["convergence"] = {"metric": "ttft_p99", "threshold": threshold}
    if ok:
        spec = AIPerfSweepSpec.model_validate(data)
        assert spec.convergence is not None
        assert spec.convergence.threshold == threshold
    else:
        with pytest.raises(ValidationError):
            AIPerfSweepSpec.model_validate(data)


def test_failure_policy_max_failures_negative_rejected() -> None:
    """``failure_policy.max_failures`` is ge=0."""
    data = _sweep(spec_extra={"failurePolicy": {"maxFailures": -1}})
    with pytest.raises(ValidationError, match=r"(?i)max.?failures|greater"):
        AIPerfSweepSpec.model_validate(data)


def test_template_spec_ttl_negative_rejected() -> None:
    """AIPerfJobSpec.ttl_seconds_after_finished has no ge constraint by default
    (None is the default), but if it's tightened to ge=0 this test guards it.

    Documents observed behavior: AIPerfJobSpec.ttl_seconds_after_finished does
    NOT currently set ge=0. Negative values are accepted at the model layer; the
    rejection (if any) happens in the kopf controller. If a future tightening
    adds ge=0, this test will flip and assert ValidationError.
    """
    spec = AIPerfSweepSpec.model_validate(
        _sweep(template_spec_extra={"ttlSecondsAfterFinished": -1}),
    )
    # If this assertion ever flips to ValidationError, replace with:
    #   with pytest.raises(ValidationError, match=...): ...
    assert spec.template.spec.ttl_seconds_after_finished == -1


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
    data[extra_key] = {"trials": 1}
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|not permitted"):
        AIPerfSweepSpec.model_validate(data)


def test_template_spec_benchmark_field_typo_rejected() -> None:
    """``benchark:`` (missing m) on AIPerfJobSpec is caught by extra=forbid."""
    template_spec = {"benchark": _VALID_BENCHMARK}
    data = {"multiRun": {"trials": 2}, "template": {"spec": template_spec}}
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|benchmark"):
        AIPerfSweepSpec.model_validate(data)


def test_metadata_labels_typo_rejected() -> None:
    """``lables`` (typo of labels) on metadata caught by extra=forbid."""
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|lables"):
        AIPerfSweepSpec.model_validate(
            _sweep(metadata={"lables": {"team": "perf"}}),
        )


def test_pod_template_image_typo_rejected() -> None:
    """``imag`` typo inside the pod-template subtree is caught.

    podTemplate is a typed PodTemplateConfig; if its inner fields use
    extra='forbid' (the project default for BaseConfig subclasses) the typo
    reaches a forbid-extra error. This test verifies the strict surface really
    extends through nested config objects, not just the top of AIPerfJobSpec.
    """
    bad_pod_template = {"workerImage": "x:latest", "imag": "should-not-be-here"}
    data = _sweep(template_spec_extra={"podTemplate": bad_pod_template})
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|imag"):
        AIPerfSweepSpec.model_validate(data)


def test_template_spec_camelcase_required_alias_works() -> None:
    """``populate_by_name=True`` lets snake_case go through alongside camelCase.

    Contrast test for the typo cases above: ``image_pull_policy`` (snake) is
    accepted and stored on the model.
    """
    spec = AIPerfSweepSpec.model_validate(
        _sweep(template_spec_extra={"image_pull_policy": "IfNotPresent"}),
    )
    assert spec.template.spec.image_pull_policy is not None
    assert spec.template.spec.image_pull_policy.value == "IfNotPresent"


def test_template_spec_camelcase_alias_form_works() -> None:
    """The camelCase alias ``imagePullPolicy`` is also accepted (sanity)."""
    spec = AIPerfSweepSpec.model_validate(
        _sweep(template_spec_extra={"imagePullPolicy": "Always"}),
    )
    assert spec.template.spec.image_pull_policy is not None
    assert spec.template.spec.image_pull_policy.value == "Always"


# ============================================================================
# Category 4 — Sweep-key smuggling
# ============================================================================


@pytest.mark.parametrize(
    "key, payload",
    [
        param("sweep", {"type": "grid", "variables": {"x": [1, 2]}}, id="sweep-snake"),
        param("multi_run", {"trials": 1}, id="multi_run-snake"),
        param("multiRun", {"trials": 1}, id="multiRun-camel"),
    ],
)  # fmt: skip
def test_sweep_key_smuggling_under_template_spec_benchmark_rejected(
    key: str, payload: dict
) -> None:
    """Rule-4 forbids axis keys at ``template.spec.benchmark``; both naming forms blocked."""
    bench = _benchmark_with(**{key: payload})
    data = _sweep(benchmark=bench)
    with pytest.raises(ValidationError):
        AIPerfSweepSpec.model_validate(data)


def test_convergence_smuggling_under_template_spec_benchmark_rejected() -> None:
    """``convergence`` is not a BenchmarkConfig field — extra=forbid catches it."""
    bench = _benchmark_with(convergence={"metric": "ttft_p99"})
    data = _sweep(benchmark=bench)
    with pytest.raises(
        ValidationError, match=r"(?i)extra|forbid|convergence|not permitted"
    ):
        AIPerfSweepSpec.model_validate(data)


def test_runtime_sweep_smuggling_caught_by_runtime_extra_forbid() -> None:
    """Deeply nested ``benchmark.runtime.sweep`` doesn't slip through.

    Rule-4 in ``_validate_axis_combination`` only inspects the top of
    ``template.spec.benchmark`` (via ``benchmark.model_fields_set``). A nested
    ``runtime.sweep`` would bypass Rule-4 — but ``runtime`` itself is a typed
    ``RuntimeConfig`` with ``extra='forbid'`` (project default for BaseConfig),
    so the apiserver still rejects the smuggled key. This test locks in the
    layered defense.
    """
    bench = _benchmark_with(
        runtime={"sweep": {"type": "grid", "variables": {"x": [1, 2]}}}
    )
    data = _sweep(benchmark=bench)
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|sweep"):
        AIPerfSweepSpec.model_validate(data)


def test_metadata_label_named_sweep_accepted() -> None:
    """``labels`` are arbitrary string maps; a key literally named 'sweep' is fine."""
    spec = AIPerfSweepSpec.model_validate(
        _sweep(metadata={"labels": {"sweep": "ramp-A"}}),
    )
    assert spec.template.metadata.labels == {"sweep": "ramp-A"}


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
    isl = spec.template.spec.benchmark.datasets[0].prompts.isl
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


@pytest.mark.parametrize(
    "convergence_key",
    [
        param("convergence", id="convergence-snake-and-camel-same"),
    ],
)  # fmt: skip
def test_convergence_without_multirun_rejected(convergence_key: str) -> None:
    """Rule-2: ``convergence`` requires ``multi_run`` set."""
    data = {
        convergence_key: {"metric": "ttft_p99"},
        "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
    }
    with pytest.raises(ValidationError, match=r"(?i)requires|multi.?run"):
        AIPerfSweepSpec.model_validate(data)


@pytest.mark.parametrize(
    "multi_run_key",
    [
        param("multiRun", id="camel"),
        param("multi_run", id="snake"),
    ],
)  # fmt: skip
def test_convergence_with_explicit_trials_rejected(multi_run_key: str) -> None:
    """Rule-2: when convergence is set, ``multi_run.trials`` must be unset.

    Both naming forms (snake_case via populate_by_name, camelCase alias) trigger.
    """
    data = {
        multi_run_key: {"trials": 5},
        "convergence": {"metric": "ttft_p99"},
        "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
    }
    with pytest.raises(ValidationError, match=r"(?i)trials|unset"):
        AIPerfSweepSpec.model_validate(data)


def test_no_axes_set_rejected() -> None:
    """Rule-1: at least one of sweep/multi_run/convergence must be set."""
    with pytest.raises(ValidationError, match=r"(?i)at least one of"):
        AIPerfSweepSpec.model_validate(
            {"template": {"spec": {"benchmark": _VALID_BENCHMARK}}}
        )


def test_all_three_axes_set_accepted() -> None:
    """Sweep + multi_run (no trials) + convergence is the full composition path."""
    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": {"type": "grid", "variables": {"phases.x.concurrency": [1, 2]}},
            "multiRun": {"cooldownSeconds": 5},
            "convergence": {"metric": "ttft_p99"},
            "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
        }
    )
    assert spec.sweep is not None
    assert spec.multi_run is not None
    assert spec.convergence is not None


def test_benchmark_with_both_model_and_models_documents_behavior() -> None:
    """``model`` (singular) and ``models`` (plural) coexisting under benchmark.

    Documents observed behavior: AIPerfConfig defines BOTH ``model`` and
    ``models`` as fields. ``model`` has ``exclude=True`` and is hoisted into
    ``models`` only when ``models`` is absent. With both set, ``_normalize_models``
    is a no-op (``models`` already present), and ``model`` survives as a
    silently-dropped shorthand. There is currently no rejection at the AIPerfConfig
    layer for this conflict.

    If we ever decide to add a ``model``+``models`` mutual-exclusivity check
    (mirroring the existing ``dataset``+``datasets`` check in
    ``_check_mutual_exclusivity``), this test will flip and need to assert
    ValidationError.
    """
    bench = {**_VALID_BENCHMARK, "model": "from-singular"}
    spec = AIPerfSweepSpec.model_validate(_sweep(benchmark=bench))
    # Plural wins; singular is dropped silently (exclude=True on the field).
    items = spec.template.spec.benchmark.models.items
    names = [m.name for m in items]
    assert "test-model" in names
    assert "from-singular" not in names


# ============================================================================
# Sanity guards
# ============================================================================


def test_object_meta_partial_rejects_top_level_typo_directly() -> None:
    """ObjectMetaPartial enforces extra=forbid at the metadata layer in isolation."""
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|name"):
        ObjectMetaPartial.model_validate({"name": "should-not-be-here"})


def test_aiperf_job_template_rejects_top_level_typo() -> None:
    """AIPerfJobTemplate enforces extra=forbid alongside ObjectMetaPartial+AIPerfJobSpec."""
    with pytest.raises(ValidationError, match=r"(?i)extra|forbid|spcc"):
        AIPerfJobTemplate.model_validate({"spcc": {"benchmark": _VALID_BENCHMARK}})
