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
