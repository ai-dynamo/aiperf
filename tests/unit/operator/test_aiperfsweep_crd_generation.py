# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the AIPerfSweep CRD generator.

Covers:
- Top-level shape of the generated CRD dict (kind, names, scope, schema paths).
- CEL ``x-kubernetes-validations`` immutability rules on critical spec fields.
- ``additionalPrinterColumns`` for ``kubectl get aiperfsweeps``.
- Helm chart emits a template containing the AIPerfSweep CRD.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_aiperfsweep_crd_has_required_paths():
    """Generate the AIPerfSweep CRD dict and assert schema shape."""
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    assert crd["kind"] == "CustomResourceDefinition"
    assert crd["spec"]["names"]["kind"] == "AIPerfSweep"
    assert crd["spec"]["names"]["plural"] == "aiperfsweeps"
    assert crd["spec"]["scope"] == "Namespaced"

    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    spec_props = schema["properties"]["spec"]["properties"]
    assert "sweep" in spec_props
    assert "multiRun" in spec_props
    assert "benchmark" in spec_props
    assert "failurePolicy" in spec_props
    assert "cancel" in spec_props

    status_props = schema["properties"]["status"]["properties"]
    assert status_props["observedGeneration"] == {
        "type": "integer",
        "format": "int64",
        "description": "Generation of the spec that was last processed",
    }


def test_aiperfsweep_immutability_rules_on_critical_fields():
    """Sweep axes and the workload template are immutable after creation."""
    from tools.generate_crd import (
        _immutable_spec_field_rule,
        build_aiperfsweep_crd,
    )

    crd = build_aiperfsweep_crd()
    spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]
    rules = {rule["rule"] for rule in spec.get("x-kubernetes-validations", [])}
    for field in ("sweep", "multiRun", "benchmark"):
        assert _immutable_spec_field_rule(field) in rules


def test_aiperfsweep_printer_columns_present():
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    printer = crd["spec"]["versions"][0].get("additionalPrinterColumns") or []
    names = [c["name"] for c in printer]
    assert "Phase" in names
    assert "Age" in names


def test_helm_chart_emits_aiperfsweep_crd_template():
    """The generator writes a chart template for the AIPerfSweep CRD."""
    chart_dir = Path("deploy/helm/aiperf-operator/templates")
    # Either a separate file OR an additional YAML document in the existing
    # crd.yaml is acceptable.
    candidates = list(chart_dir.glob("crd*.yaml"))
    assert candidates, "no crd*.yaml templates found"
    text = "\n---\n".join(c.read_text() for c in candidates)
    assert "AIPerfSweep" in text, "AIPerfSweep CRD not emitted in chart templates"
    assert "aiperfsweeps" in text


def test_aiperfsweep_template_benchmark_has_strict_walked_schema():
    """spec.benchmark on AIPerfSweep walks AIPerfConfig (Task 6 of plan).

    The previous blanket ``x-kubernetes-preserve-unknown-fields: true`` is
    replaced with a real walk; only narrow shorthand boundaries (models,
    endpoint.urls, top-level shortcuts) keep the marker.
    """
    from tools.generate_crd import build_aiperfsweep_crd

    crd = build_aiperfsweep_crd()
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    benchmark = schema["properties"]["spec"]["properties"]["benchmark"]

    # Top-level benchmark must NOT carry a blanket preserve-unknown marker —
    # individual fields are walked and validated by the apiserver.
    assert "properties" in benchmark, "benchmark should be a strictly walked object"
    assert benchmark.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "benchmark must not be a blanket preserve-unknown — Task 6 walks it"
    )

    # Narrow markers at known shorthand boundaries.
    bp = benchmark["properties"]
    assert bp["models"].get("x-kubernetes-preserve-unknown-fields") is True, (
        "models accepts shorthand and must keep the marker"
    )
    assert (
        bp["endpoint"]["properties"]["urls"].get("x-kubernetes-preserve-unknown-fields")
        is True
    ), "endpoint.urls accepts shorthand and must keep the marker"

    # Strict fields: runtime should be fully typed (no top-level marker).
    runtime = bp["runtime"]
    assert runtime.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "runtime should be strictly validated, no preserve-unknown blanket"
    )
    assert "properties" in runtime, "runtime should expose its real properties"

    # Top-level shortcut siblings (Task 5) are present and marked.
    for shortcut in ("model", "dataset", "warmup", "profiling"):
        assert shortcut in bp, f"{shortcut} shortcut sibling missing"
        assert bp[shortcut].get("x-kubernetes-preserve-unknown-fields") is True, (
            f"{shortcut} shortcut must carry preserve-unknown marker"
        )


def test_aiperfjob_benchmark_has_strict_walked_schema():
    """AIPerfJob spec.benchmark walks AIPerfConfig (Task 6 of plan).

    Mirrors :func:`test_aiperfsweep_template_benchmark_has_strict_walked_schema`
    but on the AIPerfJob CRD where the benchmark blanket previously lived.
    """
    from tools.generate_crd import _build_crd

    crd = _build_crd({})
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    benchmark = schema["properties"]["spec"]["properties"]["benchmark"]

    assert "properties" in benchmark, "benchmark should be a strictly walked object"
    assert benchmark.get("x-kubernetes-preserve-unknown-fields") is not True, (
        "benchmark must not be a blanket preserve-unknown — Task 6 walks it"
    )

    bp = benchmark["properties"]
    assert bp["models"].get("x-kubernetes-preserve-unknown-fields") is True
    assert (
        bp["endpoint"]["properties"]["urls"].get("x-kubernetes-preserve-unknown-fields")
        is True
    )
    assert bp["runtime"].get("x-kubernetes-preserve-unknown-fields") is not True
    assert "properties" in bp["runtime"]

    for shortcut in ("model", "dataset", "warmup", "profiling"):
        assert shortcut in bp, f"{shortcut} shortcut sibling missing"
        assert bp[shortcut].get("x-kubernetes-preserve-unknown-fields") is True


def _benchmark_node_aiperfjob() -> dict:
    from tools.generate_crd import _build_crd

    schema = _build_crd({})["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    return schema["properties"]["spec"]["properties"]["benchmark"]


def _benchmark_node_aiperfsweep() -> dict:
    from tools.generate_crd import build_aiperfsweep_crd

    schema = build_aiperfsweep_crd()["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    return schema["properties"]["spec"]["properties"]["benchmark"]


def test_benchmark_required_excludes_shorthand_canonicals_aiperfjob():
    """``models``/``datasets``/``phases`` must not be in benchmark.required.

    The operator's before-validator hoists shorthand siblings into the
    canonical names *after* the apiserver runs structural validation. Leaving
    them in ``required`` would reject every shorthand-only CR (the common
    AIPerf CLI YAML form) at apply-time.
    """
    benchmark = _benchmark_node_aiperfjob()
    required = set(benchmark.get("required", []))
    assert "endpoint" in required, "endpoint stays required (no shorthand sibling)"
    for canonical in ("models", "datasets", "phases"):
        assert canonical not in required, (
            f"{canonical} must move to a CEL rule so the shorthand sibling is accepted"
        )


def test_benchmark_required_excludes_shorthand_canonicals_aiperfsweep():
    """Same relaxation must apply to AIPerfSweep's spec.benchmark."""
    benchmark = _benchmark_node_aiperfsweep()
    required = set(benchmark.get("required", []))
    assert "endpoint" in required
    for canonical in ("models", "datasets", "phases"):
        assert canonical not in required


def test_benchmark_cel_rules_omit_typeless_shorthand_aiperfjob():
    """Shorthand-or-canonical CEL rules must NOT be emitted.

    ``model``/``dataset``/``warmup``/``profiling`` are typeless
    preserve-unknown siblings (must accept scalar, list, or object). CEL
    ``has(self.X)`` won't compile against a typeless field — the
    apiserver refuses the entire CRD if those rules are present.
    Enforcement lives in ``normalize_before_validation`` in
    ``src/aiperf/config/config.py`` instead.
    """
    benchmark = _benchmark_node_aiperfjob()
    rules = {r["rule"] for r in benchmark.get("x-kubernetes-validations", [])}
    forbidden = {
        "has(self.models) || has(self.model)",
        "has(self.datasets) || has(self.dataset)",
        "has(self.phases) || has(self.warmup) || has(self.profiling)",
    }
    assert not (rules & forbidden), (
        f"shorthand-referencing CEL rules must be removed (got {rules & forbidden})"
    )


def test_benchmark_cel_rules_omit_typeless_shorthand_aiperfsweep():
    """Inner template benchmark must also omit the typeless-shorthand rules."""
    benchmark = _benchmark_node_aiperfsweep()
    rules = {r["rule"] for r in benchmark.get("x-kubernetes-validations", [])}
    forbidden = {
        "has(self.models) || has(self.model)",
        "has(self.datasets) || has(self.dataset)",
        "has(self.phases) || has(self.warmup) || has(self.profiling)",
    }
    assert not (rules & forbidden)


def test_runtime_apihost_requires_apiport_cel_rule():
    """runtime.apiHost must require runtime.apiPort at the apiserver level."""
    benchmark = _benchmark_node_aiperfjob()
    runtime = benchmark["properties"]["runtime"]
    rules = {r["rule"] for r in runtime.get("x-kubernetes-validations", [])}
    assert "!has(self.apiHost) || has(self.apiPort)" in rules


# -----------------------------------------------------------------------------
# Tier 1A — shorthand/canonical mutual exclusion on benchmark.
# -----------------------------------------------------------------------------


def _benchmark_rules_aiperfjob() -> set[str]:
    benchmark = _benchmark_node_aiperfjob()
    return {r["rule"] for r in benchmark.get("x-kubernetes-validations", [])}


def _benchmark_rules_aiperfsweep() -> set[str]:
    benchmark = _benchmark_node_aiperfsweep()
    return {r["rule"] for r in benchmark.get("x-kubernetes-validations", [])}


def test_benchmark_mutual_exclusion_canonical_and_shorthand():
    """Shorthand-and-canonical mutual-exclusion rules must NOT be emitted.

    Same typeless-field reason as the OR-rules: the apiserver rejects the
    CRD if it sees ``has(self.model)`` etc. against a preserve-unknown
    sibling. ``normalize_before_validation`` raises a Pydantic
    ``ValueError`` when both forms are set, so the operator surfaces the
    failure on reconcile.
    """
    forbidden = {
        "!(has(self.models) && has(self.model))",
        "!(has(self.datasets) && has(self.dataset))",
        "!(has(self.phases) && (has(self.warmup) || has(self.profiling)))",
        "!has(self.warmup) || has(self.profiling)",
    }
    for rules in (_benchmark_rules_aiperfjob(), _benchmark_rules_aiperfsweep()):
        assert not (rules & forbidden), (
            f"shorthand-referencing CEL rules must be removed (got {rules & forbidden})"
        )


# -----------------------------------------------------------------------------
# Tier 1B — endpoint.template ↔ type=template; Tier 4O — URL validation;
# Tier 2J — multipart/form-data only on video_generation.
# -----------------------------------------------------------------------------


def _endpoint_node_aiperfjob() -> dict:
    return _benchmark_node_aiperfjob()["properties"]["endpoint"]


def test_endpoint_template_requires_type_template():
    """An omitted type must stay legal: Pydantic infers it from `template`.

    Requiring has(self.type) rejected exactly the shorthand the auto-detect
    validator produces, since the CRD gives `type` no default.
    """
    endpoint = _endpoint_node_aiperfjob()
    rules = {r["rule"] for r in endpoint.get("x-kubernetes-validations", [])}
    assert "!has(self.type) || self.type != 'template' || has(self.template)" in rules
    assert "!has(self.template) || !has(self.type) || self.type == 'template'" in rules


def test_endpoint_form_data_types_come_from_plugin_metadata():
    """Every endpoint declaring requires_form_data must appear in the rule.

    The gate on the Pydantic side reads plugin metadata, so a hardcoded list
    here drifts the moment another endpoint opts in -- image_edit did, and the
    rule still named only video_generation.
    """
    from tools.generate_crd import _form_data_endpoint_types

    endpoint = _endpoint_node_aiperfjob()
    rules = {r["rule"] for r in endpoint.get("x-kubernetes-validations", [])}
    form_rules = [r for r in rules if "multipart/form-data" in r]
    assert form_rules, "no multipart rule emitted"
    expected = _form_data_endpoint_types()
    assert len(expected) >= 2, "fixture expects more than one form-data endpoint"
    for name in expected:
        assert any(name in r for r in form_rules), f"{name} missing from the CEL rule"


def test_endpoint_urls_must_be_valid_urls():
    # CEL ``self.urls.all(u, isURL(u))`` won't compile against a typeless
    # preserve-unknown field (recipes pass plain string URLs that the apiserver
    # must accept without structural validation), so URL well-formedness is
    # enforced at the Pydantic ``EndpointConfig`` validator instead. This test
    # is the regression guard ensuring we do NOT bring the CEL rule back
    # without first restructuring ``urls`` as a typed array.
    endpoint = _endpoint_node_aiperfjob()
    rules = {r["rule"] for r in endpoint.get("x-kubernetes-validations", [])}
    assert "self.urls.all(u, isURL(u))" not in rules


# -----------------------------------------------------------------------------
# Tier 1C — AIPerfSweep axis-combination rules.
# Tier 1E — convergence bounds.
# -----------------------------------------------------------------------------


def _aiperfsweep_spec_node() -> dict:
    from tools.generate_crd import build_aiperfsweep_crd

    return build_aiperfsweep_crd()["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
        "properties"
    ]["spec"]


def test_aiperfsweep_axis_combination_rules():
    spec = _aiperfsweep_spec_node()
    rules = {r["rule"] for r in spec.get("x-kubernetes-validations", [])}
    assert "has(self.sweep)" in rules


def test_deployment_image_rejects_empty_string():
    from tools.generate_crd import _build_crd

    job_spec = _build_crd({})["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
        "properties"
    ]["spec"]
    for spec in (_aiperfsweep_spec_node(), job_spec):
        assert spec["properties"]["image"]["minLength"] == 1


def test_aiperfsweep_sweep_grid_parameters_are_structurally_typed():
    sweep = _aiperfsweep_spec_node()["properties"]["sweep"]
    assert sweep["required"] == ["type"]
    assert sweep["properties"]["type"]["enum"]
    assert sweep["properties"]["parameters"]["additionalProperties"]["type"] == "array"
    assert sweep["properties"]["parameters"]["additionalProperties"]["minItems"] == 1


def test_aiperfsweep_sweep_key_matches_the_grid_model_field():
    """The published key must be one GridSweep actually accepts.

    The schema advertised ``variables`` while GridSweep requires
    ``parameters`` and defines no such alias, so a spec written against the
    published CRD passed admission and then failed validation inside the
    operator -- with, at the time, no status written at all. Pin the CRD key
    to the model instead of to a literal so the two cannot drift again.
    """
    from aiperf.config.sweep.config import GridSweep

    grid_keys = {
        (field.alias or name) for name, field in GridSweep.model_fields.items()
    }
    sweep_props = set(_aiperfsweep_spec_node()["properties"]["sweep"]["properties"])

    assert sweep_props <= grid_keys, (
        f"CRD advertises sweep keys GridSweep does not accept: "
        f"{sorted(sweep_props - grid_keys)}"
    )


def test_crd_orchestration_scalars_use_native_kubernetes_types():
    from tools.generate_crd import _build_crd, build_aiperfsweep_crd

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]
        properties = spec["properties"]

        ttl = properties["ttlSecondsAfterFinished"]
        assert ttl["type"] == "integer"
        assert ttl["minimum"] == 0
        assert "x-kubernetes-int-or-string" not in ttl

        for key in ("cancel", "skipEndpointCheck"):
            node = properties[key]
            assert node["type"] == "boolean"
            assert "x-kubernetes-preserve-unknown-fields" not in node


def test_crd_runtime_scalars_keep_int_or_string_helm_compatibility():
    from tools.generate_crd import _build_crd, build_aiperfsweep_crd

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]
        runtime = spec["properties"]["benchmark"]["properties"]["runtime"]["properties"]

        for key in ("workers", "workersMin", "apiPort"):
            assert runtime[key]["x-kubernetes-int-or-string"] is True


def test_convergence_min_max_runs_bound():
    """Convergence config lives in multiRun; the structural
    convergence_metric/threshold fields are now flat on multiRun. The
    legacy ConvergenceConfig with min_runs/max_runs is no longer wired
    into the AIPerfSweepSpec, so the dedicated CEL bound is gone — the
    enforcement is at the model_validator level on the underlying
    convergence config when used.
    """
    spec = _aiperfsweep_spec_node()
    assert "convergence" not in spec.get("properties", {}), (
        "convergence is no longer a top-level spec field — moved into multiRun"
    )


# -----------------------------------------------------------------------------
# Tier 1D — forbid sweep/multiRun inside the AIPerfSweep template benchmark.
# -----------------------------------------------------------------------------


def test_template_benchmark_forbids_sweep_and_multirun():
    # AIPerfSweep.spec.benchmark is BenchmarkConfig (body-only). sweep/multiRun
    # are envelope keys that don't exist on BenchmarkConfig at all, so the
    # explicit CEL rule is redundant. Pydantic's `extra=forbid` still rejects
    # them at apply-time via the structural schema.
    benchmark = _benchmark_node_aiperfsweep()
    properties = benchmark.get("properties", {})
    assert "sweep" not in properties
    assert "multiRun" not in properties


def test_aiperfjob_benchmark_does_not_forbid_sweep_or_multirun():
    """The same rule must NOT fire on standalone AIPerfJob — sweep/multiRun
    are valid AIPerfConfig fields when not nested under AIPerfSweep."""
    rules = _benchmark_rules_aiperfjob()
    assert "!has(self.sweep)" not in rules
    assert "!has(self.multiRun)" not in rules


# -----------------------------------------------------------------------------
# Tier 1F — runtime.workersMin ≤ workers.
# -----------------------------------------------------------------------------


def test_runtime_workers_min_lte_workers():
    """Both operands must be cast: they are int-or-string, hence dyn to CEL.

    Comparing them directly errors with "no such overload" on a mixed
    int/string pair and compares lexicographically on a string pair, so
    {workers: "9", workersMin: "10"} passed as valid.
    """
    runtime = _benchmark_node_aiperfjob()["properties"]["runtime"]
    rules = {r["rule"] for r in runtime.get("x-kubernetes-validations", [])}
    assert (
        "!has(self.workersMin) || !has(self.workers) || "
        "int(self.workersMin) <= int(self.workers)" in rules
    )


# -----------------------------------------------------------------------------
# Tier 2 — sweep/UI/transport invariants.
# -----------------------------------------------------------------------------


def test_parameter_sweep_same_seed_no_longer_a_benchmark_scope_cel_rule():
    # The envelope is flat: ``self.multiRun`` and ``self.randomSeed``
    # are not in scope from a ``benchmark`` node, so the apiserver
    # rejects ``!has(self.multiRun) || ...`` rules attached there. The
    # ``validate_parameter_sweep_same_seed_requires_random_seed`` Pydantic
    # validator on AIPerfConfig still enforces this at submit time.
    rules = _benchmark_rules_aiperfjob()
    assert not any("parameterSweepSameSeed" in r for r in rules), (
        "envelope-scope rule must not be emitted on benchmark node"
    )


def test_dashboard_ui_no_longer_a_benchmark_scope_cel_rule():
    # Same envelope-flattening reason as above: ``self.sweep`` /
    # ``self.runtime`` are not in scope from a ``benchmark`` node.
    # ``validate_sweep_no_dashboard_ui`` on AIPerfConfig is the live check.
    rules = _benchmark_rules_aiperfjob()
    assert not any("dashboard" in r for r in rules), (
        "envelope-scope rule must not be emitted on benchmark node"
    )


def test_convergence_incompatible_with_repeated_mode_enforced_by_spec_model():
    """AIPerfSweep convergence-vs-repeated belongs in model validation.

    The invariant crosses ``spec.multiRun.convergence`` and
    ``spec.sweep.iterationOrder``. Keep it out of the CRD until both sides are
    typed in one CEL-friendly node, but assert the operator/local validation
    rejects the bad combination before a child workload is created.
    """
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec

    spec = {
        "image": "test:0",
        "benchmark": {
            "models": ["test/m"],
            "endpoint": {
                "type": "chat",
                "urls": ["http://x:8000/v1/chat/completions"],
            },
            "datasets": [{"name": "d", "type": "synthetic", "entries": 10}],
            "phases": [
                {
                    "name": "p",
                    "kind": "profiling",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
        },
        "sweep": {
            "type": "grid",
            "iterationOrder": "repeated",
            "parameters": {"phases.p.concurrency": [1, 2]},
        },
        "multiRun": {
            "numRuns": 3,
            "convergence": {"metric": "ttft", "minRuns": 2},
        },
    }
    with pytest.raises(ValueError, match="multi_run.convergence"):
        AIPerfSweepSpec.model_validate(spec)


# -----------------------------------------------------------------------------
# Tier 3 — immutability.
# -----------------------------------------------------------------------------


def test_scheduling_queue_name_immutable():
    from tools.generate_crd import _build_crd, _immutable_spec_field_rule

    crd = _build_crd({})
    spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]
    rules = {r["rule"] for r in spec.get("x-kubernetes-validations", [])}
    assert _immutable_spec_field_rule("scheduling") in rules


# -----------------------------------------------------------------------------
# Tier 4 — value-shape sanity. (Phase/dataset uniqueness, phase→dataset
# reference integrity, and seamless-not-first stay in the operator's Pydantic
# validators because the AIPerfConfig CRD walks `phases[]` and `datasets[]`
# items as opaque preserve-unknown blobs — CEL can't dereference item fields
# through that opacity.)
# -----------------------------------------------------------------------------


def test_array_item_internal_rules_are_intentionally_absent():
    """Sanity check: rules that need to peek into opaque array items must NOT
    have been emitted (the apiserver would reject the CRD at install).
    """
    rules = _benchmark_rules_aiperfjob()
    for forbidden in (
        "self.phases.all(p, self.phases.exists_one",
        "self.datasets.all(d, self.datasets.exists_one",
        "self.phases.all(p, !has(p.dataset)",
        "self.phases[0].seamless",
    ):
        assert not any(forbidden in r for r in rules), (
            f"rule containing '{forbidden}' would not compile against opaque "
            f"array items; keep this enforcement in the Pydantic validator"
        )


def test_crd_document_builder_matches_legacy_job_builder() -> None:
    from aiperf.config.config import AIPerfConfig
    from tools.generate_crd import (
        CRDDocumentBuilder,
        CRDSchemaSource,
        _build_crd,
        convert_aiperf_config_fields,
    )

    source = CRDSchemaSource()
    legacy = _build_crd(convert_aiperf_config_fields(AIPerfConfig.model_json_schema()))
    refactored = CRDDocumentBuilder().aiperfjob_crd(source.job_schema())

    assert refactored == legacy


def test_crd_document_builder_matches_legacy_sweep_builder() -> None:
    from tools.generate_crd import (
        CRDDocumentBuilder,
        CRDSchemaSource,
        build_aiperfsweep_crd,
    )

    source = CRDSchemaSource()

    assert (
        CRDDocumentBuilder().aiperfsweep_crd(source.sweep_schema())
        == build_aiperfsweep_crd()
    )


def test_shared_immutability_rules_present_on_both_kinds():
    """Shared create-time fields are immutable on both workload kinds."""
    from tools.generate_crd import (
        _build_crd,
        _immutable_spec_field_rule,
        build_aiperfsweep_crd,
    )

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        kind = crd["spec"]["names"]["kind"]
        spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]
        rules = {r["rule"] for r in spec.get("x-kubernetes-validations", [])}
        for field in ("benchmark", "multiRun", "scheduling"):
            assert _immutable_spec_field_rule(field) in rules, (
                f"{kind} missing spec.{field} immutability"
            )


def test_ref_typed_enum_defaults_reach_the_crd():
    """Enum fields reach the CRD via $ref, and their defaults must survive.

    The $ref branch of _convert_schema dropped the sibling `default`, so
    urlStrategy, connectionReuse and every other enum-typed field published a
    schema with no default -- kubectl explain and GitOps diffs disagreed with
    the model. The nullable-anyOf branch had always carried it.
    """
    from tools.generate_crd import _build_crd, build_aiperfsweep_crd

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        kind = crd["spec"]["names"]["kind"]
        endpoint = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
            "properties"
        ]["spec"]["properties"]["benchmark"]["properties"]["endpoint"]["properties"]
        assert endpoint["urlStrategy"].get("default") == "round_robin", kind
        assert endpoint["connectionReuse"].get("default") == "pooled", kind


def test_endpoint_type_has_no_crd_default():
    """endpoint.type must stay undefaulted so template auto-detect still fires.

    config/endpoint.py sets type='template' only when `type` is absent from
    the payload. An apiserver default of 'chat' would defeat that and then
    trip the template-vs-type CEL rule on the very shorthand it enables.
    """
    from tools.generate_crd import _build_crd, build_aiperfsweep_crd

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        endpoint = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
            "properties"
        ]["spec"]["properties"]["benchmark"]["properties"]["endpoint"]["properties"]
        assert "default" not in endpoint["type"], crd["spec"]["names"]["kind"]


def test_multirun_convergence_min_runs_rule_on_both_kinds():
    """multiRun carries the minRuns<=numRuns rule, mirroring the validator.

    The decorator's shape detector required a flat convergenceMetric+mode
    pair that MultiRunConfig has not had since convergence became a nested
    model, so it matched nothing and emitted no rule anywhere. The AIPerfJob
    builder additionally walked only its benchmark sub-tree, leaving the
    top-level multiRun node undecorated on that kind.
    """
    from tools.generate_crd import _build_crd, build_aiperfsweep_crd

    for crd in (_build_crd({}), build_aiperfsweep_crd()):
        mr = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "spec"
        ]["properties"]["multiRun"]
        rules = {r["rule"] for r in mr.get("x-kubernetes-validations", [])}
        assert any("convergence.minRuns <= self.numRuns" in r for r in rules), (
            f"{crd['spec']['names']['kind']} missing convergence.minRuns rule"
        )
