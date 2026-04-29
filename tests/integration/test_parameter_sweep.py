# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the in-process parameter sweep path.

Ported (focused) from main's PR #699 ``tests/integration/test_parameter_sweep.py``
and adapted to k8s's ``BenchmarkPlan`` / ``RunExecutor`` /
``aggregate_sweep_and_export`` shape.

The k8s port uses ``expand_sweep`` over a ``--concurrency 10,20`` magic-list,
which produces a ``SweepVariation`` whose ``label`` is the dotted path
``phases.profiling.concurrency=10`` (NOT main's ``concurrency_10``). The
orchestrator writes per-variation artifacts under
``<artifact_dir>/<variation.label>/profile_runs/run_NNNN/`` and the sweep
aggregate under ``<artifact_dir>/sweep_aggregate/``.

Out of scope here:
- Operator-mode gate (``AIPERF_OPERATOR_MANAGED=1``); covered at unit level in
  tests/unit/test_cli_runner.py::TestOperatorModeSweepGate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from pytest import param

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


# Parameter key in sweep-aggregate JSON: HEAD's expand_sweep keys per-cell
# stats by the dotted-path variation key (NOT main's bare ``concurrency``);
# this drops directly into ``per_combination_metrics[*].parameters`` and
# ``metadata.sweep_parameters[*].name``.
SWEPT_PARAM_KEY = "phases.profiling.concurrency"


def _variation_dir(base: Path, concurrency: int) -> Path:
    """Resolve the per-variation artifact directory.

    expand_sweep promotes ``--concurrency 10,20`` into
    ``phases.profiling.concurrency`` (dotted path) and the variation label
    becomes ``phases.profiling.concurrency=<value>``. The orchestrator
    writes ``<base>/<variation.label>/profile_runs/run_0001/`` per trial.
    """
    return base / f"phases.profiling.concurrency={concurrency}"


def _trial_run_path(base: Path, concurrency: int, run_index: int) -> Path:
    """Resolve a specific run's artifact dir under a variation cell."""
    return _variation_dir(base, concurrency) / "profile_runs" / f"run_{run_index:04d}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestParameterSweep:
    """End-to-end coverage for the in-process sweep path."""

    async def test_sweep_basic_concurrency_list_writes_per_variation_dirs(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``aiperf profile --concurrency 10,20`` produces per-variation dirs.

        Verifies the basic sweep cardinality contract: one variation cell
        per concurrency value, and a sweep-aggregate JSON whose filename
        matches the exporter contract from commit f2713733f.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (10, 20):
            run_dir = _trial_run_path(temp_output_dir, concurrency, run_index=1)
            assert run_dir.exists(), f"missing per-variation run dir: {run_dir}"
            json_path = run_dir / "profile_export_aiperf.json"
            assert json_path.exists(), f"missing per-cell json: {json_path}"

        sweep_dir = temp_output_dir / "sweep_aggregate"
        assert sweep_dir.exists(), "sweep_aggregate dir missing"
        sweep_json = sweep_dir / "profile_export_aiperf_sweep.json"
        sweep_csv = sweep_dir / "profile_export_aiperf_sweep.csv"
        assert sweep_json.exists(), f"sweep json missing: {sweep_json}"
        assert sweep_csv.exists(), f"sweep csv missing: {sweep_csv}"

    async def test_sweep_with_multi_run_writes_trial_subdirs_and_aggregate(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--concurrency 10,20 --num-profile-runs 2`` -> 2 trials per cell.

        Tree shape (under ``temp_output_dir``)::

            phases.profiling.concurrency=10/
              profile_runs/run_0001/profile_export_aiperf.json
              profile_runs/run_0002/profile_export_aiperf.json
              aggregate/profile_export_aiperf_aggregate.json
            phases.profiling.concurrency=20/
              profile_runs/run_0001/profile_export_aiperf.json
              profile_runs/run_0002/profile_export_aiperf.json
              aggregate/profile_export_aiperf_aggregate.json
            sweep_aggregate/
              profile_export_aiperf_sweep.json
              profile_export_aiperf_sweep.csv
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (10, 20):
            for trial_idx in (1, 2):
                run_dir = _trial_run_path(
                    temp_output_dir, concurrency, run_index=trial_idx
                )
                assert run_dir.exists(), f"missing trial dir: {run_dir}"
                assert (run_dir / "profile_export_aiperf.json").exists()

        sweep_dir = temp_output_dir / "sweep_aggregate"
        sweep_json_path = sweep_dir / "profile_export_aiperf_sweep.json"
        assert sweep_json_path.exists()

        with sweep_json_path.open() as f:
            sweep_data = json.load(f)
        # SweepAnalyzer.compute writes a list of per-combination metrics
        # whose length must equal the number of variations.
        assert "per_combination_metrics" in sweep_data
        assert len(sweep_data["per_combination_metrics"]) == 2

    async def test_sweep_cooldown_actually_delays_between_variations(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--parameter-sweep-cooldown-seconds 1`` is observable in wall-clock.

        We use a small cooldown (1s) and at least 2 variations so the
        delta is measurable but doesn't blow up the test budget. The
        absolute floor is ``num_variations - 1`` cooldowns; we assert
        the run took at least that long (with slack for overhead).
        """
        cooldown_s = 1.0
        num_variations = 2

        start = time.monotonic()
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --parameter-sweep-cooldown-seconds {cooldown_s} \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        elapsed = time.monotonic() - start

        assert result.exit_code == 0
        # cooldowns happen between variations: (n - 1) of them.
        # Allow ~10% slack below the floor — wall clock under xdist is noisy
        # but the cooldown is asyncio.sleep(...) so it's a hard lower bound
        # for the run loop itself; subtract a small tolerance for variance.
        min_cooldown_total = (num_variations - 1) * cooldown_s
        assert elapsed >= min_cooldown_total * 0.9, (
            f"elapsed={elapsed:.2f}s did not reflect "
            f"--parameter-sweep-cooldown-seconds {cooldown_s} between "
            f"{num_variations} variations (expected >= {min_cooldown_total:.2f}s)"
        )

    @pytest.mark.parametrize(
        "same_seed_flag, expect_identical",
        [
            param("--parameter-sweep-same-seed", True, id="same-seed"),
            param("--no-parameter-sweep-same-seed", False, id="distinct-seeds"),
        ],
    )  # fmt: skip
    async def test_sweep_same_seed_controls_input_determinism(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
        same_seed_flag: str,
        expect_identical: bool,
    ) -> None:
        """``--parameter-sweep-same-seed`` => identical synthesized inputs.

        Compares the synthesized ``inputs.json`` payload bytes across two
        variation cells; with same-seed they must match exactly, with
        distinct seeds they must differ (Phase 4.4's
        ``_apply_sweep_seed_derivation``).
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --random-seed 42 \
                --parameter-sweep-mode independent \
                {same_seed_flag} \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        inputs_a = _trial_run_path(temp_output_dir, 10, 1) / "inputs.json"
        inputs_b = _trial_run_path(temp_output_dir, 20, 1) / "inputs.json"
        assert inputs_a.exists() and inputs_b.exists()

        # Compare the synthesized payload list, not the entire file: the
        # top-level wrapper may carry incidental per-run metadata (run id,
        # timestamps) that shifts even at identical seeds.
        payload_a = json.loads(inputs_a.read_text()).get("data", [])
        payload_b = json.loads(inputs_b.read_text()).get("data", [])

        if expect_identical:
            assert payload_a == payload_b, (
                "same-seed sweep produced different synthesized inputs across "
                "variations — _apply_sweep_seed_derivation should have left "
                "seeds untouched"
            )
        else:
            assert payload_a != payload_b, (
                "distinct-seed sweep produced identical inputs across "
                "variations — _apply_sweep_seed_derivation should have "
                "shifted seeds by variation.index"
            )

    async def test_sweep_independent_mode_writes_variation_outer_layout(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--parameter-sweep-mode=independent`` (default) groups trials per variation.

        Verifies the contract from SweepMode docstring:
            INDEPENDENT: <base>/<variation>/profile_runs/run_NNNN/

        Variations are the OUTER loop, trials inner. With 2 variations and
        2 trials, expect: <variation_0>/profile_runs/run_0001 + run_0002,
        then <variation_1>/profile_runs/run_0001 + run_0002. No
        ``profile_runs/trial_NNNN/`` segment under the base dir.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0, (
            f"independent-mode sweep should succeed; stderr={result.stderr!r}"
        )
        v0 = _variation_dir(temp_output_dir, 10)
        v1 = _variation_dir(temp_output_dir, 20)
        assert (v0 / "profile_runs" / "run_0001").exists(), (
            "expected variation-0 trial 1 dir under independent-mode tree"
        )
        assert (v0 / "profile_runs" / "run_0002").exists()
        assert (v1 / "profile_runs" / "run_0001").exists()
        assert (v1 / "profile_runs" / "run_0002").exists()
        # Independent mode does NOT introduce a top-level profile_runs/trial_NNNN/
        # prefix - that is repeated-mode's signature path.
        assert not (temp_output_dir / "profile_runs" / "trial_0001").exists()

    async def test_sweep_repeated_mode_writes_trial_outer_layout(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """``--parameter-sweep-mode=repeated`` groups variations per trial.

        Verifies the contract from SweepMode docstring:
            REPEATED: <base>/profile_runs/trial_NNNN/<variation>/...

        Trials are the OUTER loop. With 2 variations and 2 trials, expect:
        trial_0001/<v0> + trial_0001/<v1>, then trial_0002/<v0> +
        trial_0002/<v1>. The sweep-aggregate output (mode-agnostic) still
        lives at <base>/sweep_aggregate/.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 10,20 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 5 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0, (
            f"repeated-mode sweep should succeed; stderr={result.stderr!r}"
        )
        t1 = temp_output_dir / "profile_runs" / "trial_0001"
        t2 = temp_output_dir / "profile_runs" / "trial_0002"
        # Each trial dir contains both variations
        assert (t1 / "phases.profiling.concurrency=10").exists()
        assert (t1 / "phases.profiling.concurrency=20").exists()
        assert (t2 / "phases.profiling.concurrency=10").exists()
        assert (t2 / "phases.profiling.concurrency=20").exists()
        # Sweep aggregate still lives at the base (mode-agnostic).
        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists(), "sweep aggregate should be written under both modes"
        with sweep_json.open() as f:
            data = json.load(f)
        assert "per_combination_metrics" in data
        assert len(data["per_combination_metrics"]) == 2, (
            f"expected 2 sweep cells; got {data['per_combination_metrics']!r}"
        )

    async def test_backward_compatibility_single_concurrency_no_sweep_dirs(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Single-value ``--concurrency 5`` must not create sweep artifacts.

        Adapted from main's ``test_backward_compatibility_single_concurrency``.
        The cardinality contract: a single concurrency value is NOT a sweep
        (``BenchmarkPlan.is_sweep`` is False), so the orchestrator must not
        emit any sweep-shaped directories. Existing scripts that grep for
        ``profile_export_aiperf.json`` at the artifact root keep working.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 5 \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0, "Single concurrency run should succeed"

        # Forbidden sweep-shaped dirs.
        assert not (temp_output_dir / "sweep_aggregate").exists(), (
            "Single concurrency must NOT create sweep_aggregate dir"
        )
        # HEAD's sweep variation labels are dotted-path; main's were
        # `concurrency_5`. Forbid both shapes to catch regressions either way.
        assert not (temp_output_dir / "phases.profiling.concurrency=5").exists(), (
            "Single concurrency must NOT create a per-variation cell"
        )
        assert not (temp_output_dir / "concurrency_5").exists(), (
            "Single concurrency must NOT create a main-style per-variation cell"
        )

        # Top-level artifact must exist (the pre-sweep contract).
        json_path = temp_output_dir / "profile_export_aiperf.json"
        assert json_path.exists(), "Should have JSON artifact at top level"
        with json_path.open() as f:
            run_data = json.load(f)
        assert run_data["request_count"]["avg"] == 10
        # No sweep-related metadata should leak into the single-run JSON.
        assert "sweep_index" not in run_data.get("metadata", {})
        assert "sweep_mode" not in run_data.get("metadata", {})

    async def test_sweep_aggregate_per_combination_metrics_math_invariants(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Per-combination metrics in ``profile_export_aiperf_sweep.json`` obey
        the elementary statistics invariants.

        Adapted from main's ``test_per_value_confidence_statistics`` — kept the
        sweep-aggregate slice of that test, since HEAD's ``SweepAnalyzer.compute``
        is the authoritative producer for that block. Locks:

        - ``min - epsilon <= mean <= max + epsilon``
        - ``std >= 0``
        - ``ci_low <= mean <= ci_high``
        - Each entry has ``parameters`` (with the swept key) and a non-empty
          ``metrics`` dict where each metric carries ``mean, std, min, max,
          ci_low, ci_high, unit``.

        Multi-trial path (``--num-profile-runs 2``) routes through
        ``ConfidenceAggregation``; the math invariants apply both there and in
        single-trial fallback (``_json_metric_to_stats``), so this test is the
        right place to anchor the aggregator-output contract. Two variations
        (not three) keep the parallel-xdist memory cost down — invariants
        scale with one cell, not with cell count.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists()
        with sweep_json.open() as f:
            sweep_data = json.load(f)

        per_combo = sweep_data["per_combination_metrics"]
        assert isinstance(per_combo, list)
        assert len(per_combo) == 2, (
            f"expected 2 swept variations; got {len(per_combo)}"
        )

        epsilon = 1e-9
        required_metric_fields = (
            "mean",
            "std",
            "min",
            "max",
            "ci_low",
            "ci_high",
            "unit",
        )
        found_concurrency_values: list[int] = []
        for combo in per_combo:
            assert "parameters" in combo, f"missing parameters in combo: {combo!r}"
            assert "metrics" in combo, f"missing metrics in combo: {combo!r}"
            params = combo["parameters"]
            assert SWEPT_PARAM_KEY in params, (
                f"missing {SWEPT_PARAM_KEY!r} in params: {params!r}"
            )
            found_concurrency_values.append(params[SWEPT_PARAM_KEY])
            metrics = combo["metrics"]
            assert len(metrics) > 0, f"combo {params!r} has no metrics"

            for metric_name, metric_data in metrics.items():
                for field in required_metric_fields:
                    assert field in metric_data, (
                        f"combo {params!r} metric {metric_name!r} missing {field!r}; "
                        f"keys={list(metric_data)}"
                    )
                # min <= mean <= max with floating-point tolerance.
                assert (
                    metric_data["min"] - epsilon
                    <= metric_data["mean"]
                    <= metric_data["max"] + epsilon
                ), (
                    f"combo {params!r} metric {metric_name!r}: "
                    f"min={metric_data['min']} mean={metric_data['mean']} "
                    f"max={metric_data['max']}"
                )
                assert metric_data["std"] >= 0, (
                    f"combo {params!r} metric {metric_name!r}: "
                    f"std={metric_data['std']} must be non-negative"
                )
                assert (
                    metric_data["ci_low"] - epsilon
                    <= metric_data["mean"]
                    <= metric_data["ci_high"] + epsilon
                ), (
                    f"combo {params!r} metric {metric_name!r}: "
                    f"ci_low={metric_data['ci_low']} mean={metric_data['mean']} "
                    f"ci_high={metric_data['ci_high']}"
                )
                assert isinstance(metric_data["unit"], str), (
                    f"combo {params!r} metric {metric_name!r}: unit must be str"
                )

        assert sorted(found_concurrency_values) == [2, 4], (
            f"expected concurrency values [2, 4]; got {found_concurrency_values!r}"
        )

    async def test_sweep_aggregate_structure_validation(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Sweep aggregate JSON has the four-section schema from PR #699.

        Adapted from main's ``test_sweep_aggregate_structure_validation``. Validates:

        - top-level keys: ``metadata``, ``per_combination_metrics``,
          ``best_configurations``, ``pareto_optimal``.
        - ``metadata.sweep_parameters`` is a non-empty list of
          ``{name, values}`` entries.
        - ``metadata.num_combinations`` matches the variation count.
        - ``best_configurations.best_throughput`` (if present) carries
          ``parameters``, ``metric``, ``unit``.
        - ``pareto_optimal`` is a list of parameter dicts; every entry has
          the swept key.

        Two variations keep the heavier multi-trial path's parallel-xdist
        memory cost contained.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists()
        with sweep_json.open() as f:
            sweep_data = json.load(f)

        for key in (
            "metadata",
            "per_combination_metrics",
            "best_configurations",
            "pareto_optimal",
        ):
            assert key in sweep_data, f"sweep aggregate JSON missing {key!r} key"

        # metadata.sweep_parameters is the new multi-parameter schema.
        metadata = sweep_data["metadata"]
        assert "sweep_parameters" in metadata
        sweep_parameters = metadata["sweep_parameters"]
        assert isinstance(sweep_parameters, list) and len(sweep_parameters) > 0
        assert sweep_parameters[0]["name"] == SWEPT_PARAM_KEY
        assert sweep_parameters[0]["values"] == [2, 4]
        # num_combinations matches variation count.
        assert metadata["num_combinations"] == 2

        per_combo = sweep_data["per_combination_metrics"]
        assert isinstance(per_combo, list)
        assert len(per_combo) == 2
        for combo in per_combo:
            assert isinstance(combo, dict)
            assert "parameters" in combo and "metrics" in combo
            assert SWEPT_PARAM_KEY in combo["parameters"]

        # best_configurations: per-objective single-best entries.
        # SweepAnalyzer drops a key when the underlying metric isn't present,
        # so we only assert the schema of keys that DID emit.
        best_configs = sweep_data["best_configurations"]
        assert isinstance(best_configs, dict)
        for objective_name, entry in best_configs.items():
            assert "parameters" in entry, (
                f"best_configurations[{objective_name!r}] missing 'parameters'"
            )
            assert "metric" in entry, (
                f"best_configurations[{objective_name!r}] missing 'metric'"
            )
            assert "unit" in entry, (
                f"best_configurations[{objective_name!r}] missing 'unit'"
            )
            assert entry["parameters"][SWEPT_PARAM_KEY] in (2, 4)

        pareto = sweep_data["pareto_optimal"]
        assert isinstance(pareto, list)
        for params in pareto:
            assert isinstance(params, dict)
            assert SWEPT_PARAM_KEY in params
            assert params[SWEPT_PARAM_KEY] in (2, 4)

    async def test_sweep_only_mode_single_trial_writes_one_run_per_cell(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Single-trial sweep (no ``--num-profile-runs``) writes one run per cell.

        Adapted from main's ``test_sweep_only_mode_without_confidence``. Pinned
        to ``--parameter-sweep-mode independent`` because the variations-outer
        layout is the only one that matches main's flat
        ``<base>/<variation>/...`` structure (the ``repeated`` mode default
        wraps everything in ``profile_runs/trial_NNNN/<variation>/...``).

        With the default ``--num-profile-runs=1``:

        - each variation cell gets exactly one ``profile_runs/run_0001/``
          subtree (HEAD's universal layout, regardless of ``num_profile_runs``);
        - no ``run_0002`` directory exists;
        - ``sweep_aggregate/`` is still written so plotting tooling can
          consume the combinatorial output.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (2, 4):
            cell_dir = _variation_dir(temp_output_dir, concurrency)
            assert cell_dir.exists(), f"missing variation cell: {cell_dir}"
            run1 = cell_dir / "profile_runs" / "run_0001"
            run2 = cell_dir / "profile_runs" / "run_0002"
            assert run1.exists(), (
                f"single-trial sweep should still write run_0001: {run1}"
            )
            assert not run2.exists(), (
                f"single-trial sweep must NOT write run_0002: {run2}"
            )
            json_file = run1 / "profile_export_aiperf.json"
            assert json_file.exists()

        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists(), (
            "sweep aggregate should be written even for single-trial sweeps"
        )
        with sweep_json.open() as f:
            data = json.load(f)
        assert len(data["per_combination_metrics"]) == 2, (
            "single-trial sweep aggregate should have one row per variation"
        )

    async def test_sweep_aggregate_csv_has_wide_format_with_param_columns(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Sweep aggregate CSV is wide-format: parameter columns + metric stats.

        Adapted from main's ``test_aggregate_file_generation`` CSV slice.
        Wide format: one row per variation, columns are
        ``<parameter>``, ``<metric>_mean``, ``<metric>_std``, ``<metric>_min``,
        ``<metric>_max``, ``<metric>_cv``. Locks the on-disk schema that
        downstream plotting/notebook tooling reads (``AggregateSweepCsvExporter``).
        Two variations (not three) keep parallel-xdist memory cost down.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        sweep_csv = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.csv"
        )
        assert sweep_csv.exists()
        csv_content = sweep_csv.read_text()
        lines = csv_content.strip().split("\n")
        assert len(lines) > 1, "Sweep CSV must have a header and at least one data row"

        header = lines[0]
        # Parameter column - HEAD's variation key is dotted-path.
        assert SWEPT_PARAM_KEY in header, (
            f"Sweep CSV header missing {SWEPT_PARAM_KEY!r} column; header={header!r}"
        )
        # At least one metric column with a stats suffix.
        suffixes = ("_mean", "_std", "_min", "_max", "_cv")
        assert any(s in header for s in suffixes), (
            f"Sweep CSV header missing wide-format metric suffixes "
            f"(_mean/_std/_min/_max/_cv); header={header!r}"
        )

        # Each swept value appears in the data rows.
        body = "\n".join(lines[1:])
        for value in (2, 4):
            assert str(value) in body, (
                f"Sweep CSV body missing data for concurrency={value}"
            )

    async def test_per_variation_aggregate_files_written_in_independent_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Independent-mode multi-trial sweep writes per-cell confidence aggregates.

        For each variation with >=2 successful runs, an aggregate JSON+CSV
        pair lands at ``<base>/<variation_label>/aggregate/`` (independent
        mode places aggregate inside the variation cell, mirroring
        origin/main's ``SweepConfidenceStrategy.export_aggregates``).

        Validates the per-variation aggregate JSON's confidence shape
        (``aggregation_type='confidence'``, ``num_runs=2``, swept-key
        metadata, valid ``mean/std/ci_low/ci_high`` per metric) and
        confirms the sweep aggregate is also produced alongside.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (2, 4):
            agg_dir = _variation_dir(temp_output_dir, concurrency) / "aggregate"
            agg_json = agg_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = agg_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"per-variation aggregate JSON missing for concurrency={concurrency}: "
                f"{agg_json}"
            )
            assert agg_csv.exists(), (
                f"per-variation aggregate CSV missing for concurrency={concurrency}: "
                f"{agg_csv}"
            )
            with agg_json.open() as f:
                agg_data = json.load(f)
            # AggregateConfidenceJsonExporter flattens AggregateResult
            # metadata: aggregation_type / num_profile_runs /
            # num_successful_runs / variation_label / variation_values
            # all live under the single ``metadata`` block.
            meta = agg_data.get("metadata", {})
            assert meta.get("aggregation_type") == "confidence", (
                f"expected aggregation_type='confidence'; got {agg_data!r}"
            )
            assert meta.get("num_profile_runs") == 2, (
                f"expected num_profile_runs=2 for {concurrency}; "
                f"got {meta.get('num_profile_runs')!r}"
            )
            assert meta.get("num_successful_runs") == 2
            assert meta.get("variation_label") == (
                f"phases.profiling.concurrency={concurrency}"
            )
            assert meta.get("variation_values") == {
                "phases.profiling.concurrency": concurrency
            }
            metrics = agg_data.get("metrics") or {}
            assert metrics, f"per-variation aggregate has no metrics for {concurrency}"
            for metric_name, metric in metrics.items():
                for field in ("mean", "std", "min", "max", "ci_low", "ci_high"):
                    assert field in metric, (
                        f"metric {metric_name!r} missing {field!r}: {metric!r}"
                    )
                assert metric["min"] - 1e-9 <= metric["mean"] <= metric["max"] + 1e-9
                assert metric["std"] >= 0.0
                assert metric["ci_low"] - 1e-9 <= metric["mean"] <= metric["ci_high"] + 1e-9

        # Sweep aggregate must still be present alongside.
        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists(), (
            "Sweep aggregate must coexist with per-variation aggregates"
        )

    async def test_per_variation_aggregate_files_written_in_repeated_mode(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Repeated-mode multi-trial sweep writes per-cell confidence aggregates.

        Repeated mode flips the path: per-variation aggregate dirs land
        at ``<base>/aggregate/<variation_label>/`` (aggregate is the
        outer directory, variation is the leaf), mirroring origin/main.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --num-profile-runs 2 \
                --parameter-sweep-mode repeated \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (2, 4):
            agg_dir = (
                temp_output_dir
                / "aggregate"
                / f"phases.profiling.concurrency={concurrency}"
            )
            agg_json = agg_dir / "profile_export_aiperf_aggregate.json"
            agg_csv = agg_dir / "profile_export_aiperf_aggregate.csv"
            assert agg_json.exists(), (
                f"repeated-mode per-variation aggregate JSON missing: {agg_json}"
            )
            assert agg_csv.exists(), (
                f"repeated-mode per-variation aggregate CSV missing: {agg_csv}"
            )
            with agg_json.open() as f:
                agg_data = json.load(f)
            meta = agg_data.get("metadata", {})
            assert meta.get("num_profile_runs") == 2
            assert str(meta.get("sweep_mode")).lower() == "repeated"

        # The independent-mode layout must NOT be written in repeated mode.
        for concurrency in (2, 4):
            wrong = _variation_dir(temp_output_dir, concurrency) / "aggregate"
            assert not (wrong / "profile_export_aiperf_aggregate.json").exists(), (
                f"repeated mode must not write independent-mode layout: {wrong}"
            )

    async def test_per_variation_aggregate_skipped_when_below_minimum_runs(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        temp_output_dir: Path,
    ) -> None:
        """Single-trial sweep skips per-variation aggregates without crashing.

        With ``--num-profile-runs 1`` (default), each variation has only
        1 successful run, which is below the >=2 minimum needed for
        confidence statistics. The per-variation aggregate dirs must NOT
        be written, but the sweep aggregate (which can degrade single-
        trial cells via ``_json_metric_to_stats``) must still produce.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --concurrency 2,4 \
                --parameter-sweep-mode independent \
                --request-count 10 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.exit_code == 0

        for concurrency in (2, 4):
            agg_dir = _variation_dir(temp_output_dir, concurrency) / "aggregate"
            agg_json = agg_dir / "profile_export_aiperf_aggregate.json"
            assert not agg_json.exists(), (
                f"single-trial cells must not write per-variation aggregates; "
                f"unexpected file: {agg_json}"
            )

        # Sweep aggregate is still expected (handles single-trial via
        # _json_metric_to_stats).
        sweep_json = (
            temp_output_dir / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
        )
        assert sweep_json.exists(), (
            "single-trial sweep must still write sweep_aggregate"
        )
