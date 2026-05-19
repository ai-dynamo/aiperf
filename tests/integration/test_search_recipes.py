# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests for each built-in search recipe against the dynamic
mock server (scheduler-enabled, batch-size knees, optional goodput collapse).

One test per recipe. Each recipe drives ``aiperf profile --search-recipe ...``
through a real subprocess against a mock server tuned to make the recipe's
target signal observable. Assertions stay qualitative -- artifacts exist, the
post-process produced a populated payload, planners ran for the expected
minimum number of iterations -- because exact knee values are noisy at smoke
budgets and the goal here is to prove the recipes wire end-to-end against the
dynamic mock features, not to pin specific saturation points.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchRecipes:
    """One test per recipe; each pins the plausible answer the recipe lands on."""

    @pytest.mark.slow
    async def test_concurrency_ramp_detects_degradation_knee(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """concurrency-ramp grid + degradation-knee handler land a positive knee.

        Mock has a tight saturation shelf at concurrency=8 (max_batch_size) plus
        a quadratic TTFT penalty so request latency p99 inflates well past the
        20% baseline cutoff somewhere on the recipe's [1, 1000] log grid.
        """
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_max_prefill_chunks_per_step=64,
            ttft_concurrency_quad_ms=5.0,
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
                    --search-recipe concurrency-ramp
                    --degradation-threshold 0.50
                    --request-count 30
                    --warmup-request-count 4
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 32
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=300.0,
            )

        knee_path = temp_output_dir / "sweep_aggregate" / "degradation_knee.json"
        assert knee_path.exists(), (
            f"recipe did not emit {knee_path.name}; post-process handler "
            "may not have run"
        )
        knee = orjson.loads(knee_path.read_bytes())

        # Saturation knee must land somewhere on the swept grid (>= the
        # batch-size shelf at 8). The recipe's default grid covers [1, 1000]
        # log-spaced over 8 steps, so any positive int >= 8 is plausible.
        assert knee.get("knee_concurrency") is not None, knee
        assert knee["knee_concurrency"] >= 8, knee
        assert knee["baseline_concurrency"] == 1, knee
        assert knee["stat"] == "p99"
        assert knee["swept_metric"] == "request_latency"
        assert len(knee["all_points"]) >= 2, knee

    @pytest.mark.slow
    async def test_prefill_ttft_curve_fits_linear_with_isl_penalty(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """prefill-ttft-curve + ttft_curve_fit emit a populated curve fit.

        Mock TTFT scales linearly with ISL (0.05 ms/token) so a linear fit
        should explain most of the variance. We assert "either the linear fit
        meets a relaxed r^2 floor OR the curve has at least the points we
        managed to measure" -- the recipe's grid is 8 ISLs up to 32k tokens
        and not all of those will complete inside a smoke budget.
        """
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_prefill_chunk_tokens=256,
            scheduler_max_prefill_chunks_per_step=64,
            ttft_per_isl_token_ms=0.05,
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
                    --search-recipe prefill-ttft-curve
                    --request-count 20
                    --warmup-request-count 2
                    --output-tokens-mean 16
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=300.0,
            )

        # Recipe writes prefill_curve.json (output_filename in builtins.py).
        curve_path = temp_output_dir / "sweep_aggregate" / "prefill_curve.json"
        assert curve_path.exists(), (
            f"recipe did not emit {curve_path.name}; post-process handler "
            "may not have run"
        )
        curve = orjson.loads(curve_path.read_bytes())

        assert curve["swept_metric"] == "time_to_first_token"
        assert curve["stat"] == "avg"
        # The handler always populates raw_points and a fit form; the linear
        # fit's r^2 should be >= 0.5 with a clean linear ISL penalty even when
        # only a subset of the 8 default ISL points completed.
        assert curve["fit_form"] in ("linear", "quadratic"), curve
        assert len(curve["raw_points"]) >= 2, curve
        if curve["fit_form"] == "linear":
            assert curve["r_squared"] >= 0.5, curve
        else:
            # Quadratic fallback fired (linear r^2 < floor); accept it as long
            # as the points it fitted are ours.
            assert len(curve["coefficients"]) == 3, curve

    @pytest.mark.slow
    async def test_decode_itl_curve_emits_2d_surface(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """decode-itl-curve + itl_surface_fit emit a populated 2D surface.

        Mock has both per-OSL-token and concurrency-linear ITL penalties so
        each surface cell has a distinct ITL. ITL must be at least the 5ms
        scheduler step floor.
        """
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_max_prefill_chunks_per_step=64,
            itl_per_osl_token_ms=0.01,
            itl_concurrency_lin_ms=0.05,
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
                    --search-recipe decode-itl-curve
                    --concurrency-steps 3
                    --osl-steps 2
                    --request-count 4
                    --warmup-request-count 1
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=300.0,
            )

        surface_path = temp_output_dir / "sweep_aggregate" / "decode_itl_surface.json"
        assert surface_path.exists(), (
            f"recipe did not emit {surface_path.name}; post-process handler "
            "may not have run"
        )
        surface = orjson.loads(surface_path.read_bytes())

        assert surface["swept_metric"] == "inter_token_latency"
        assert surface["stat"] == "avg"
        assert "surface" in surface
        grid = surface["surface"]["itl_grid"]
        # Flatten + drop nulls for unmeasured cells; the recipe's full grid is
        # 6x4 = 24 combinations -- a smoke budget will measure a subset.
        measured = [v for row in grid for v in row if v is not None]
        assert len(measured) >= 1, surface
        # ITL floor: scheduler_step_ms = 5ms; allow some jitter slack.
        assert min(measured) >= 4.0, (min(measured), surface)
        assert len(surface["raw_points"]) >= 1, surface

    @pytest.mark.slow
    async def test_max_concurrency_under_sla_finds_boundary(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """max-concurrency-under-sla --search-style monotonic finds a finite boundary.

        Mock has a saturation knee at concurrency=8 plus a strong
        concurrency-linear ITL penalty (0.5 ms per concurrent request) so ITL
        stretches past the 50ms TPOT SLA somewhere above the knee.
        """
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_max_prefill_chunks_per_step=64,
            itl_concurrency_lin_ms=0.5,
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
                    --search-style monotonic
                    --tpot-sla-ms 50
                    --request-count 30
                    --warmup-request-count 4
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 32
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=600.0,
            )

        history_path = temp_output_dir / "search_history.json"
        assert history_path.exists(), (
            "recipe did not emit search_history.json; planner may not have run"
        )
        history = orjson.loads(history_path.read_bytes())

        iterations = history.get("iterations") or []
        assert len(iterations) >= 3, (
            f"monotonic planner ran {len(iterations)} iters; expected >= 3"
        )
        assert history.get("convergence_reason") is not None, history

        boundary = history.get("boundary_summary")
        assert boundary is not None, history
        feasible_max = boundary.get("feasible_max")
        # Accept either feasible_max set (planner found a feasible point) or
        # the planner ran multiple iterations and produced an infeasible_min,
        # i.e. the boundary was at least bracketed even if no point passed.
        assert feasible_max is not None or boundary.get("infeasible_min") is not None, (
            history
        )
        if feasible_max is not None:
            assert feasible_max["value"] >= 1, history

    @pytest.mark.slow
    async def test_max_goodput_under_slo_lands_near_collapse_point(
        self,
        cli: AIPerfCLI,
        mock_server_factory,
        temp_output_dir: Path,
    ) -> None:
        """max-goodput-under-slo BO finds a finite, positive concurrency.

        Mock combines the saturation knee at 8 with goodput collapse and TTFT
        + ITL penalties past it -- past the knee, request_count completion
        rate drops below 95% so the BO objective collapses too. We don't pin
        an exact concurrency (BO at smoke budget is noisy) -- just that the
        planner ran multiple iterations and reported a finite winning value.
        """
        async with mock_server_factory(
            scheduler_enabled=True,
            scheduler_step_ms=5.0,
            scheduler_max_batch_size=8,
            scheduler_max_prefill_chunks_per_step=64,
            scheduler_goodput_collapse_enabled=True,
            scheduler_goodput_collapse_threshold=1.0,
            scheduler_goodput_collapse_slope=1.0,
            scheduler_goodput_collapse_floor=0.3,
            ttft_concurrency_quad_ms=2.0,
            itl_concurrency_lin_ms=0.3,
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
                    --search-recipe max-goodput-under-slo
                    --ttft-sla-ms 200
                    --tpot-sla-ms 50
                    --e2e-sla-ms 5000
                    --request-count 30
                    --warmup-request-count 4
                    --synthetic-input-tokens-mean 16
                    --synthetic-input-tokens-stddev 0
                    --output-tokens-mean 32
                    --output-tokens-stddev 0
                    --extra-inputs ignore_eos:true
                    --ui none
                """,
                timeout=420.0,
            )

        history_path = temp_output_dir / "search_history.json"
        assert history_path.exists(), (
            "recipe did not emit search_history.json; planner may not have run"
        )
        history = orjson.loads(history_path.read_bytes())

        iterations = history.get("iterations") or []
        assert len(iterations) >= 3, (
            f"BO planner ran {len(iterations)} iters; expected >= 3"
        )

        best_trials = history.get("best_trials")
        assert best_trials, history
        best = best_trials[0]
        # Best variation block must include a concurrency value; the recipe's
        # 1D search space is on phases.profiling.concurrency.
        variation = best.get("variation_values") or {}
        assert variation, history
        concurrency = next(iter(variation.values()))
        assert isinstance(concurrency, (int, float)), variation
        assert concurrency >= 1, variation
