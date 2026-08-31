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
  parameters:
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
    assert cr["spec"]["sweep"]["parameters"] == {"phases.profiling.concurrency": [1, 2]}


def test_build_sweep_cr_dict_multi_run_emits_camelcase_num_runs(tmp_path: Path) -> None:
    """``--multi-run-trials N`` appears under spec.multiRun.numRuns (NOT 'trials')."""
    config_file = tmp_path / "trials.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2]}
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
    """--convergence-metric populates spec.multiRun.convergence with all three sub-fields."""
    config_file = tmp_path / "conv.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
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
  parameters: {phases.profiling.concurrency: [1, 2]}
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


def test_build_sweep_cr_dict_preserves_templates_for_variable_sweep(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "variable-sweep.yaml"
    config_file.write_text(
        """\
variables: {concurrency: 1, processors: 1}
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
runtime:
  record_processors_per_pod: '{{ processors }}'
phases:
  - name: profiling
    type: concurrency
    requests: 10
    concurrency: '{{ concurrency }}'
sweep:
  type: grid
  parameters:
    variables.concurrency: [8, 32]
    variables.processors: [2]
"""
    )

    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )

    benchmark = cr["spec"]["benchmark"]
    assert benchmark["phases"][0]["concurrency"] == "{{ concurrency }}"
    assert benchmark["runtime"]["recordProcessorsPerPod"] == "{{ processors }}"
    assert "record_processors_per_pod" not in benchmark["runtime"]

    from aiperf.sweep_controller.plan_builder import build_plan_from_sweep

    plan = build_plan_from_sweep(cr)
    assert [cfg.phases[0].concurrency for cfg in plan.configs] == [8, 32]
    assert [cfg.runtime.record_processors_per_pod for cfg in plan.configs] == [2, 2]


def test_build_sweep_cr_dict_preserves_origin_main_envelope_fields(
    tmp_path: Path,
) -> None:
    plot_file = tmp_path / "plot.yaml"
    plot_file.write_text(
        """\
visualization:
  multi_run_defaults: []
  single_run_defaults: []
"""
    )
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(
        """\
benchmark:
  models: [m]
  endpoint: {urls: [http://x], type: chat, streaming: false}
  datasets:
    - name: main
      type: synthetic
      prompts: {isl: 64, osl: 32, cacheBust: {target: none}}
  phases:
    - name: profiling
      type: concurrency
      requests: 10
      concurrency: 1
      trajectoryStartMinRatio: 0.0
  artifacts: {autoPlot: false}
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [1, 2]
plot: ./plot.yaml
variables: {region: us-west}
randomSeed: 7
noSweepTable: false
schemaVersion: "2.0"
"""
    )

    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(),
    )

    spec = cr["spec"]
    assert spec["schemaVersion"] == "2.0"
    assert spec["variables"] == {"region": "us-west"}
    assert spec["randomSeed"] == 7
    assert spec["noSweepTable"] is False
    assert spec["plot"]["visualization"] == {
        "multiRunDefaults": [],
        "singleRunDefaults": [],
    }
    benchmark = spec["benchmark"]
    assert benchmark["endpoint"]["streaming"] is False
    assert benchmark["datasets"][0]["prompts"]["cacheBust"] == {"target": "none"}
    assert benchmark["phases"][0]["trajectoryStartMinRatio"] == 0.0
    assert benchmark["artifacts"]["autoPlot"] is False

    from aiperf.sweep_controller.plan_builder import build_plan_from_sweep

    plan = build_plan_from_sweep(cr)
    assert plan.no_sweep_table is False
    assert plan.plot is not None
    assert plan.configs[0].endpoint._streaming_explicitly_set is True
    assert plan.configs[0].datasets[0].prompts.cache_bust._target_explicitly_set is True
    assert plan.configs[0].phases[0]._trajectory_start_min_ratio_explicitly_set is True
    assert "auto_plot" in plan.configs[0].artifacts.model_fields_set


# ---------------------------------------------------------------------------
# _build_sweep_cr_dict — strict-decoding camelCase round-trip
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_round_trips_through_pydantic_to_camelcase_keys(
    tmp_path: Path,
) -> None:
    """Every key at every nesting level inside ``spec`` MUST be camelCase
    (apiserver strict-decoding rejects snake_case even though Pydantic accepts
    both via populate_by_name).

    Excludes the ``sweep.parameters`` keys (user-provided keys
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
  parameters:
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
    sweep_var_keys = set(cr["spec"].get("sweep", {}).get("parameters", {}))
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
    # The explicitly authored discriminator survives the intent-preserving dump.
    assert sweep["type"] == "adaptive_search"
    objectives = sweep["objectives"]
    assert len(objectives) == 2
    assert objectives[0]["metric"] == "output_token_throughput"
    assert objectives[0]["direction"] == "maximize"
    # Explicit values equal to defaults survive the intent-preserving dump.
    assert objectives[0]["stat"] == "avg"
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
            kube_options=KubeOptions(),
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


def test_name_from_config_file_respects_operator_child_name_budget() -> None:
    from aiperf.sweep_controller._naming import MAX_SWEEP_NAME_LENGTH

    out = sweep_cmd._name_from_config_file(
        Path("a-very-long-sweep-configuration-name.yaml")
    )

    assert len(out) <= MAX_SWEEP_NAME_LENGTH


_INVALID_BENCHMARK_YAML = """\
models: [m]
endpoint: {urls: [http://x], type: chat, streaming: true}
datasets: [{name: main, type: synthetic, prompts: {isl: 64, osl: 32}}]
phases:
  - {name: profiling, type: poisson, duration: 10, concurrency: 8}
sweep:
  type: grid
  parameters:
    phases.profiling.concurrency: [4, 8]
"""


@pytest.mark.asyncio
async def test_sweep_dry_run_invalid_config_exits_nonzero(tmp_path: Path) -> None:
    """`--dry-run` must fail loudly: a CI gate reads the exit code, not the panel.

    The rate-controlled phase omits `rate`, so config construction raises inside
    `exit_on_error`, which renders the panel and exits 1.
    """
    from aiperf.config.flags.cli_config import KubeCLIConfig

    config_file = tmp_path / "bad.yaml"
    config_file.write_text(_INVALID_BENCHMARK_YAML)

    with pytest.raises(SystemExit) as excinfo:
        await sweep_cmd.sweep(
            cli_config=KubeCLIConfig(config_file=config_file),
            kube_options=_kube_options(),
            dry_run=True,
        )

    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# _merged_multirun_config — convergence merge and --max-runs precedence
# ---------------------------------------------------------------------------


def test_build_sweep_cr_dict_convergence_preserves_yaml_mode_and_stat(
    tmp_path: Path,
) -> None:
    """CLI convergence flags merge into the YAML block instead of replacing it.

    ``multiRun.convergence`` was previously assigned wholesale from the three
    CLI-mapped keys, so a YAML ``mode``/``stat`` silently reverted to the
    ConvergenceConfig defaults the moment ``--convergence-metric`` was passed --
    a different stopping rule than the user wrote down.
    """
    config_file = tmp_path / "conv-merge.yaml"
    config_file.write_text(
        _yaml_with(
            """\
multiRun:
  convergence:
    mode: cv
    stat: p99
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            convergence_metric="output_token_throughput",
            convergence_min_runs=3,
            convergence_max_runs=None,
            convergence_threshold=0.05,
        ),
    )
    convergence = cr["spec"]["multiRun"]["convergence"]
    assert convergence["mode"] == "cv"
    assert convergence["stat"] == "p99"
    # The three CLI-mapped keys still win.
    assert convergence["metric"] == "output_token_throughput"
    assert convergence["minRuns"] == 3
    assert convergence["threshold"] == 0.05


def test_build_sweep_cr_dict_convergence_without_max_runs_keeps_trials(
    tmp_path: Path,
) -> None:
    """``--trials N`` survives a convergence run when ``--max-runs`` is absent.

    ``--max-runs`` used to default to a concrete 10 and overwrite numRuns
    unconditionally, so ``--trials 8 --convergence-metric X`` silently ran 10.
    """
    config_file = tmp_path / "conv-trials.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            multi_run_trials=8,
            convergence_metric="output_token_throughput",
            convergence_max_runs=None,
        ),
    )
    assert cr["spec"]["multiRun"]["numRuns"] == 8


def test_build_sweep_cr_dict_convergence_without_max_runs_keeps_yaml_num_runs(
    tmp_path: Path,
) -> None:
    """A YAML ``multiRun.numRuns`` also survives when ``--max-runs`` is absent."""
    config_file = tmp_path / "conv-yaml-runs.yaml"
    config_file.write_text(
        _yaml_with(
            """\
multiRun:
  numRuns: 8
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            convergence_metric="output_token_throughput",
            convergence_max_runs=None,
        ),
    )
    assert cr["spec"]["multiRun"]["numRuns"] == 8


def test_build_sweep_cr_dict_convergence_falls_back_to_default_cap(
    tmp_path: Path,
) -> None:
    """With no --max-runs, no --trials, and no YAML numRuns, the 10 fallback applies."""
    config_file = tmp_path / "conv-default.yaml"
    config_file.write_text(
        _yaml_with(
            """\
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            convergence_metric="output_token_throughput",
            convergence_max_runs=None,
        ),
    )
    assert cr["spec"]["multiRun"]["numRuns"] == 10


def test_build_sweep_cr_dict_explicit_max_runs_still_caps_yaml_num_runs(
    tmp_path: Path,
) -> None:
    """An explicit --max-runs remains a hard cap, overriding a larger YAML numRuns."""
    config_file = tmp_path / "conv-cap.yaml"
    config_file.write_text(
        _yaml_with(
            """\
multiRun:
  numRuns: 8
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(
            convergence_metric="output_token_throughput",
            convergence_max_runs=5,
        ),
    )
    assert cr["spec"]["multiRun"]["numRuns"] == 5


def test_build_sweep_cr_dict_max_runs_caps_yaml_defined_convergence(
    tmp_path: Path,
) -> None:
    """--max-runs caps convergence configured entirely in the YAML file."""
    config_file = tmp_path / "yaml-convergence-cap.yaml"
    config_file.write_text(
        _yaml_with(
            """\
multiRun:
  numRuns: 8
  convergence:
    metric: time_to_first_token
    minRuns: 3
    threshold: 0.05
sweep:
  type: grid
  parameters: {phases.profiling.concurrency: [1, 2, 3]}
"""
        )
    )
    cr = sweep_cmd._build_sweep_cr_dict(
        config_file=config_file,
        kube_options=_kube_options(),
        **_kwargs(convergence_max_runs=5),
    )

    assert cr["spec"]["multiRun"]["numRuns"] == 5


# ---------------------------------------------------------------------------
# sweep() command — convergence_metric comes from cli_config, not a shadowed
# local param (b903289182)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_dry_run_propagates_convergence_metric_from_cli_config(
    tmp_path: Path,
) -> None:
    """sweep() must pass cli_config.convergence_metric to _build_sweep_cr_dict.

    Before b903289182, sweep() declared its own ``convergence_metric`` and
    ``convergence_threshold`` parameters.  The CLIConfig binding wins when
    cyclopts resolves the function, so those local params were always None,
    and _merged_multirun_config's convergence branch was never entered.

    After the fix, the params are removed and cli_config.convergence_metric is
    read directly, so --convergence-metric propagates.
    """
    from unittest.mock import patch as mock_patch

    from aiperf.cli_commands.kube import sweep as sweep_cmd
    from aiperf.config.flags.cli_config import CLIConfig

    config_file = tmp_path / "bench.yaml"
    config_file.write_text(_yaml_with())

    captured_kwargs: list[dict] = []

    def _fake_build(**kwargs):
        captured_kwargs.append(kwargs)
        return {"metadata": {}, "spec": {}}

    with (
        mock_patch.object(sweep_cmd, "_build_sweep_cr_dict", side_effect=_fake_build),
        mock_patch(
            "aiperf.kubernetes.cli_helpers.resolve_benchmark_namespace",
            return_value="ns-test",
        ),
        mock_patch("aiperf.kubernetes.console.emit_raw"),
    ):
        await sweep_cmd.sweep(
            cli_config=CLIConfig(
                config_file=config_file,
                convergence_metric="ttft_p99",
                convergence_threshold=0.03,
            ),
            kube_options=_kube_options(),
            convergence_min_runs=1,
            convergence_max_runs=5,
            dry_run=True,
        )

    assert len(captured_kwargs) == 1, "_build_sweep_cr_dict must be called exactly once"
    call_kwargs = captured_kwargs[0]
    assert call_kwargs["convergence_metric"] == "ttft_p99", (
        "convergence_metric was not forwarded from cli_config — "
        "it may still be coming from a now-removed local parameter shadow"
    )
    assert call_kwargs["convergence_threshold"] == pytest.approx(0.03)
