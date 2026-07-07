# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``aiperf kube sweep`` CR-builder helper.

Targets ``_build_sweep_cr_dict`` and ``_name_from_config_file`` in
``aiperf.cli_commands.kube.sweep``. The cyclopts ``@app.default`` handler
in that module is a thin wrapper around the helper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aiperf.cli_commands.kube import sweep as sweep_cmd
from aiperf.config.kube import KubeOptions


def _kube_options(**overrides) -> KubeOptions:
    """Construct minimal KubeOptions; overrides allow per-test customization."""
    base = {"image": "aiperf:test"}
    base.update(overrides)
    return KubeOptions(**base)


_MIN_BENCHMARK_YAML = """\
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: concurrency, requests: 10, concurrency: 1}
"""


def _yaml_with(extra: str = "") -> str:
    """Bare valid AIPerfConfig YAML body + caller-appended sweep/multi_run block."""
    return _MIN_BENCHMARK_YAML + extra


def _kwargs(**overrides):
    """Defaults for the non-config-file kwargs to ``_build_sweep_cr_dict``."""
    base = dict(
        multi_run_trials=None,
        cooldown_seconds=0.0,
        convergence_metric=None,
        convergence_min_runs=3,
        convergence_max_runs=10,
        convergence_threshold=0.05,
    )
    base.update(overrides)
    return base


def _all_keys(node) -> list[str]:
    """Recursively collect every dict key in a nested CR dict."""
    keys: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            keys.append(k)
            keys.extend(_all_keys(v))
    elif isinstance(node, list):
        for item in node:
            keys.extend(_all_keys(item))
    return keys


# ---------------------------------------------------------------------------
# _build_sweep_cr_dict — basic envelope & sweep block
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_emits_aiperfsweep_kind_and_api_version(
    tmp_path: Path,
) -> None:
    """Minimal sweep YAML produces an AIPerfSweep CR with hoisted spec.sweep."""
    config_file = tmp_path / "concurrency_grid.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [1, 2]
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )
    assert cr["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
    assert cr["kind"] == "AIPerfSweep"
    # Default name from filename stem ("concurrency_grid" -> "concurrency-grid").
    assert cr["metadata"]["name"] == "concurrency-grid-sweep"
    assert "sweep" in cr["spec"]
    # `variables` is a free-form user map: keys are sweep-variable paths
    # (e.g. ``random_seed``, ``phases.profiling.concurrency``) and pass
    # through verbatim -- they're NOT subject to the camelCase round-trip.
    assert cr["spec"]["sweep"]["variables"] == {"phases.profiling.concurrency": [1, 2]}


def test_build_sweep_cr_dict_multi_run_emits_camelcase_num_runs(tmp_path: Path) -> None:
    """``--multi-run-trials N`` appears under spec.multiRun.numRuns (NOT 'trials')."""
    config_file = tmp_path / "trials.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  variables: {phases.profiling.concurrency: [1, 2]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(multi_run_trials=4, cooldown_seconds=10.0),
    )
    assert cr["spec"]["multiRun"]["numRuns"] == 4
    assert "trials" not in cr["spec"]["multiRun"]
    assert cr["spec"]["multiRun"]["cooldownSeconds"] == 10.0


def test_build_sweep_cr_dict_with_convergence_round_trips_all_fields(
    tmp_path: Path,
) -> None:
    """--convergence-metric populates spec.multiRun.convergence with all four sub-fields."""
    config_file = tmp_path / "conv.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  variables: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            convergence_metric="output_token_throughput",
            convergence_min_runs=3,
            convergence_max_runs=7,
            convergence_threshold=0.05,
        ),
    )
    mr = cr["spec"]["multiRun"]
    assert mr["convergence"]["metric"] == "output_token_throughput"
    assert mr["convergence"]["minRuns"] == 3
    assert mr["convergence"]["threshold"] == 0.05
    # numRuns := max(existing, max_runs); existing is 1 here so 7 wins.
    assert mr["numRuns"] == 7


def test_build_sweep_cr_dict_envelope_fields_lift_to_spec_top_level(
    tmp_path: Path,
) -> None:
    """variables/random_seed are envelope-level fields; they MUST appear under
    spec (not under spec.benchmark)."""
    config_file = tmp_path / "envelope.yaml"
    config_file.write_text(
        _MIN_BENCHMARK_YAML.replace(
            "phases:",
            "random_seed: 42\nvariables: {model_id: foo}\nphases:",
        )
        + """\
sweep:
  type: grid
  variables: {phases.profiling.concurrency: [1, 2]}
"""
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )
    spec = cr["spec"]
    assert spec.get("randomSeed") == 42 or spec.get("random_seed") == 42
    # `variables` is forwarded as-is (envelope dict).
    assert spec.get("variables") == {"model_id": "foo"}
    # And NOT inside spec.benchmark.
    assert "random_seed" not in spec["benchmark"]
    assert "randomSeed" not in spec["benchmark"]
    assert "variables" not in spec["benchmark"]


# ---------------------------------------------------------------------------
# _build_sweep_cr_dict — strict-decoding camelCase round-trip
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_round_trips_through_pydantic_to_camelcase_keys(
    tmp_path: Path,
) -> None:
    """Every key at every nesting level inside ``spec`` MUST be camelCase
    (apiserver strict-decoding rejects snake_case even though Pydantic accepts
    both via populate_by_name).

    Excludes the immutable ``variables:`` map values (user-provided keys
    pass through verbatim) and outer envelope keys above ``spec``.
    """
    config_file = tmp_path / "strict.yaml"
    config_file.write_text(
        _yaml_with(
            """\
multi_run:
  num_runs: 3
  cooldown_seconds: 5
sweep:
  type: grid
  variables:
    phases.profiling.concurrency: [1, 2]
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )
    keys = _all_keys(cr["spec"])
    # `variables` is a free-form user map; its KEYS aren't subject to the
    # camelCase rule (they're random_seed-style sweep-variable paths).
    snake = [
        k for k in keys if "_" in k and not re.fullmatch(r"[a-z0-9_]*_[a-z0-9_]*", "")
    ]
    # Allow two-known-exempt keys: sweep.variables map keys (random_seed,
    # phases.profiling.concurrency etc.). All OTHER keys must be camelCase.
    sweep_var_keys = set(cr["spec"].get("sweep", {}).get("variables", {}))
    leaks = [k for k in snake if k not in sweep_var_keys]
    assert not leaks, f"snake_case keys leaked into spec: {leaks}"


# ---------------------------------------------------------------------------
# _build_sweep_cr_dict — adaptive_search round-trip
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_adaptive_search_round_trips_with_objectives_list(
    tmp_path: Path,
) -> None:
    """An ``AdaptiveSearchSweep`` body round-trips as ``spec.sweep.type ==
    'adaptive_search'`` with a multi-entry ``objectives: [Objective, ...]``
    list (length-1 = single-objective BO; length-N = Pareto BO).

    Exercises the current schema, where ``AdaptiveSearchSweep`` carries
    ``objectives`` and ``planner`` (replacing the legacy singular
    ``objective`` + ``algorithm`` fields).
    """
    config_file = tmp_path / "adaptive.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: adaptive_search
  planner: bayesian
  search_space:
    - {path: phases.profiling.concurrency, lo: 1, hi: 1000, kind: int}
  objectives:
    - {metric: output_token_throughput, stat: avg, direction: maximize}
    - {metric: time_to_first_token, stat: p95, direction: minimize}
  outcome_constraints:
    - {metric: request_latency, op: "<=", bound: 5000.0}
  max_iterations: 30
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )
    sweep = cr["spec"]["sweep"]
    # `type` is a Literal default ("adaptive_search") and stripped by
    # exclude_defaults; downstream consumers re-discriminate via SweepConfig.
    # Discriminator presence is asserted via the Pydantic round-trip below.
    objectives = sweep["objectives"]
    assert len(objectives) == 2
    assert objectives[0]["metric"] == "output_token_throughput"
    assert objectives[0]["direction"] == "maximize"
    # `stat` is the default ("avg") and gets stripped by exclude_defaults.
    assert objectives[1]["metric"] == "time_to_first_token"
    assert objectives[1]["stat"] == "p95"
    assert objectives[1]["direction"] == "minimize"

    assert sweep["maxIterations"] == 30
    assert sweep["searchSpace"][0]["path"] == "phases.profiling.concurrency"

    # outcome_constraints serializes through with camelCase round-trip.
    constraints = sweep["outcomeConstraints"]
    assert len(constraints) == 1
    assert constraints[0]["metric"] == "request_latency"
    assert constraints[0]["op"] == "<="
    assert constraints[0]["bound"] == 5000.0

    # Round-trip through AdaptiveSearchSweep: re-stamping the literal
    # type and validating against the model confirms the dump is
    # consumable by downstream code (operator CRD, sweep_controller).
    from aiperf.config.sweep import AdaptiveSearchSweep

    rebuilt = AdaptiveSearchSweep.model_validate({**sweep, "type": "adaptive_search"})
    assert len(rebuilt.objectives) == 2
    assert rebuilt.objectives[0].metric == "output_token_throughput"
    assert rebuilt.objectives[1].direction == "minimize"
    assert rebuilt.max_iterations == 30
    assert len(rebuilt.outcome_constraints) == 1


# ---------------------------------------------------------------------------
# _build_sweep_cr_dict — error & rejection paths
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_requires_config_file() -> None:
    """No --config <file> raises a helpful ValueError."""
    with pytest.raises(ValueError, match="--config <file>"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=None,
            kube_options=_kube_options(),
            **_kwargs(),
        )


def test_build_sweep_cr_dict_rejects_aiperfsweep_cr_input(tmp_path: Path) -> None:
    """An AIPerfSweep CR is already a sweep; reject with a pointer to kubectl."""
    config_file = tmp_path / "sweep-cr.yaml"
    config_file.write_text(
        "apiVersion: aiperf.nvidia.com/v1alpha1\nkind: AIPerfSweep\nspec: {}\n"
    )
    with pytest.raises(ValueError, match="already an AIPerfSweep CR"):
        sweep_cmd._build_sweep_cr_dict(
            config_file=config_file,
            kube_options=_kube_options(),
            **_kwargs(),
        )


# ---------------------------------------------------------------------------
# _name_from_config_file — DNS-1123 sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem,expected",
    [
        pytest.param("foo bar.yaml", "foo-bar-sweep", id="space-collapses-to-dash"),
        pytest.param("My_Crazy.Config.YAML", None, id="lowercases-and-sanitizes"),
        pytest.param("___.yaml", "aiperf-sweep", id="all-underscores-falls-back"),
        pytest.param("simple.yaml", "simple-sweep", id="simple-stem"),
        pytest.param(
            "a-very-long-stem-that-exceeds-thirty-chars-yaml.yaml",
            None,
            id="long-stem-truncated",
        ),
    ],
)  # fmt: skip
def test_name_from_config_file_sanitizes_to_dns_1123(
    stem: str, expected: str | None
) -> None:
    out = sweep_cmd._name_from_config_file(Path(stem))
    # Must end with `-sweep` and be a valid DNS-1123 label.
    assert out.endswith("-sweep")
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", out), out
    assert len(out) <= 63
    if expected is not None:
        assert out == expected


def test_name_from_config_file_empty_stem_falls_back_to_aiperf() -> None:
    """A stem that sanitizes to empty (e.g. ``___.yaml``) gets the ``aiperf`` fallback."""
    out = sweep_cmd._name_from_config_file(Path("___.yaml"))
    assert out == "aiperf-sweep"
