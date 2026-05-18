# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: ``max-concurrency-under-sla --search-style grid`` finds
the SLA-feasibility boundary on a goodput-collapsing mock server.

The mock server is configured with a batch size of 8 plus aggressive
goodput collapse so per-token latency is ~5ms below concurrency=8 and
spikes to 30-50ms past the knee. The recipe's grid sweep walks
log-spaced concurrencies in [1, 1000] and the ``sla_breach_knee``
post-process artifact must report the boundary at concurrency 7 (last
feasible) -> 19 (first infeasible).

Parametrized over ``--num-profile-runs`` 1 and 2 to exercise both the
single-trial mean-fallback path in
``aiperf.orchestrator.aggregation.sweep_sla_filter.read_metric_value``
and the multi-trial flat-key path. Without the mean-fallback (regression
guard), single-trial mode silently marks every grid point infeasible
because the single-trial stats block has no ``p95`` key.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecipeFindsCollapseKnee:
    """``max-concurrency-under-sla`` grid sweep against the goodput-collapse mock."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "num_profile_runs",
        [
            pytest.param(1, id="single_trial"),
            pytest.param(2, id="multi_trial"),
        ],
    )
    async def test_grid_recipe_locates_sla_breach_knee(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
        num_profile_runs: int,
    ) -> None:
        # Mock server: max_batch=8 step=5ms gives ~5ms/token below the knee;
        # goodput-collapse with floor=0.3 collapses effective batch to 2 once
        # the queue is >2x oversubscribed so ITL spikes to ~30-50ms.
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_max_prefill_chunks_per_step=64,
            scheduler_goodput_collapse_enabled=True,
            scheduler_goodput_collapse_threshold=1.0,
            scheduler_goodput_collapse_slope=1.0,
            scheduler_goodput_collapse_floor=0.3,
            ttft=0.0,
            itl=0.0,
            workers=1,
        ) as server:
            await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --search-recipe max-concurrency-under-sla
                    --search-style grid
                    --tpot-sla-ms 12
                    --request-count 16
                    --warmup-request-count 4
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 32
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --num-profile-runs {num_profile_runs}
                    --ui none
                """,
                timeout=600.0,
            )

        breach_path = (
            (temp_output_dir / "aggregate" / "sweep_aggregate" / "sla_breach.json")
            if num_profile_runs > 1
            else (temp_output_dir / "sweep_aggregate" / "sla_breach.json")
        )
        assert breach_path.exists(), (
            f"recipe did not emit {breach_path.name} — "
            "post-process handler may not have run"
        )
        breach = orjson.loads(breach_path.read_bytes())

        # Defaults from MaxConcurrencyUnderSLA: 8 log-spaced steps in [1, 1000]
        # → [1, 3, 7, 19, 52, 139, 373, 1000].
        all_points = breach["all_points"]
        assert [p["concurrency"] for p in all_points] == [
            1, 3, 7, 19, 52, 139, 373, 1000,
        ]  # fmt: skip

        # Knee assertion: max_batch=8 puts the saturation knee at c~=8 and
        # collapse pushes ITL well past 12ms by c=19. The grid resolves the
        # boundary as max_passing=7, first_failing=19.
        assert breach["max_passing_concurrency"] == 7, breach
        assert breach["first_failing_concurrency"] == 19, breach

        # Feasibility must be strictly monotone in concurrency (collapse is
        # monotone in queue depth). A False here would mean the SUT noise
        # dominated the signal — i.e. the test is unstable, not the code.
        assert breach["monotonicity_check"] is True, breach

        # Per-point sanity: all sub-knee points feasible, all super-knee
        # infeasible. Catches regressions where the boundary collapses to
        # all-feasible or all-infeasible (the original single-trial bug).
        feasibility = {p["concurrency"]: p["feasible"] for p in all_points}
        assert feasibility[1] is True
        assert feasibility[3] is True
        assert feasibility[7] is True
        assert feasibility[19] is False
        assert feasibility[52] is False

        # The first failing point must report the ITL filter (only filter set).
        breach_record = breach["first_failing_breach"]
        assert breach_record["metric_tag"] == "inter_token_latency"
        assert breach_record["op"] == "lt"
        assert breach_record["threshold"] == 12.0
        assert breach_record["observed"] is not None
        assert breach_record["observed"] > 12.0


_COLLAPSE_MOCK_KWARGS = {
    "scheduler_enabled": True,
    "scheduler_step_ms": 5.0,
    "scheduler_max_batch_size": 8,
    "scheduler_max_prefill_chunks_per_step": 64,
    "scheduler_goodput_collapse_enabled": True,
    "scheduler_goodput_collapse_threshold": 1.0,
    "scheduler_goodput_collapse_slope": 1.0,
    "scheduler_goodput_collapse_floor": 0.3,
    "ttft": 0.0,
    "itl": 0.0,
    "workers": 1,
}


@pytest.mark.integration
@pytest.mark.asyncio
class TestAdaptiveRecipesLocateCollapseKnee:
    """Adaptive search recipes converge to the goodput-collapse knee.

    These tests exercise the adaptive paths (single-trial through
    ``read_metric_value`` + ``trial_satisfies``) end-to-end against a mock
    whose decode-batch knee sits at concurrency=8. Each recipe must select
    a "best" concurrency near the knee and not silently latch at the upper
    bound — the failure mode of the pre-fix single-trial bug.

    ``request_count=32`` (4x ``max_batch_size``) is the minimum that lets
    the scheduler actually queue at the higher concurrencies the planners
    bracket against; smaller values clamp in-flight to ``request_count``
    so the system never oversubscribes regardless of the concurrency knob.
    """

    # `smooth_isotonic` is intentionally NOT covered here: it does replicate
    # bootstrap sampling per design point (~70s/iter × up-to-30 iters), so
    # against this mock it routinely takes 15-30 minutes — too slow for the
    # default integration suite. Coverage of the read_metric_value fallback
    # is sufficient via `monotonic` and `bo`, which exercise the same
    # adaptive-search code paths. Re-enable once the planner has a
    # max-iterations CLI override or a fast-mode test hook.
    # `optuna` is also out of scope here: the slow recipe coverage already
    # exercises the shared adaptive-search paths through `bo`.
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "search_style,knee_band",
        [
            pytest.param("monotonic", (4, 16), id="monotonic"),
            pytest.param("bo", (4, 16), id="bo"),
        ],
    )
    async def test_max_concurrency_under_sla_finds_knee(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
        search_style: str,
        knee_band: tuple[int, int],
    ) -> None:
        async with mock_server_factory(**_COLLAPSE_MOCK_KWARGS) as server:
            await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --search-recipe max-concurrency-under-sla
                    --search-style {search_style}
                    --tpot-sla-ms 12
                    --request-count 32
                    --warmup-request-count 8
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 16
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=600.0,
            )

        history = self._read_search_history(temp_output_dir)
        best_concurrency = self._extract_best_concurrency(history)
        lo, hi = knee_band
        assert lo <= best_concurrency <= hi, (
            f"recipe selected concurrency={best_concurrency}, expected in [{lo}, {hi}]; "
            f"convergence_reason={history.get('convergence_reason')}"
        )

        # At least one iteration must have been marked infeasible. If all
        # iterations are feasible, the planner never saw the SLA breach
        # — that's the regression signature of the single-trial bug.
        any_infeasible = any(
            it.get("feasible") is False for it in history["iterations"]
        )
        assert any_infeasible, (
            "no iteration was marked infeasible — SLA filter likely not applied; "
            "regression of the single-trial mean-fallback fix in read_metric_value"
        )

    @pytest.mark.slow
    async def test_max_goodput_under_slo_finds_knee(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """Goodput recipe (BO over good_request_fraction)."""
        async with mock_server_factory(**_COLLAPSE_MOCK_KWARGS) as server:
            await cli.run(
                f"""
                aiperf profile
                    --model mock-model
                    --url {server.url}
                    --endpoint-type chat
                    --streaming
                    --search-recipe max-goodput-under-slo
                    --ttft-sla-ms 200
                    --tpot-sla-ms 12
                    --e2e-sla-ms 2000
                    --slo-attainment-fraction 0.9
                    --request-count 32
                    --warmup-request-count 8
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 16
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=600.0,
            )

        history = self._read_search_history(temp_output_dir)
        best_concurrency = self._extract_best_concurrency(history)
        # Goodput peaks before the collapse cliff; BO should land somewhere
        # near (but not far past) the knee at c=8.
        assert 1 <= best_concurrency <= 64, (
            f"goodput recipe selected concurrency={best_concurrency}; "
            f"expected near knee (~8). convergence_reason="
            f"{history.get('convergence_reason')}"
        )

    @staticmethod
    def _read_search_history(artifact_dir: Path) -> dict:
        path = artifact_dir / "search_history.json"
        assert path.exists(), (
            f"adaptive recipe did not emit {path.name}; planner may have "
            f"crashed before convergence"
        )
        return orjson.loads(path.read_bytes())

    @staticmethod
    def _extract_best_concurrency(history: dict) -> int:
        best_trials = history.get("best_trials")
        assert best_trials, (
            f"search_history.json has no 'best_trials' key: {history.keys()}"
        )
        best = best_trials[0]
        variation = best.get("variation_values") or {}
        concurrency = variation.get("phases.profiling.concurrency")
        assert concurrency is not None, (
            f"best.variation_values missing concurrency key: {variation}"
        )
        return int(concurrency)
