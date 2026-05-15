# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search recipe coverage for the v1 config converter."""

import pytest
from pytest import param

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.sweep import AdaptiveSearchSweep, GridSweep
from aiperf.plugin.enums import SearchPlannerType


def _convert_recipe(
    recipe_name: str,
    *,
    streaming: bool,
    **sweeping_overrides: object,
):
    user = CLIConfig(
        model_names=["test-model"],
        streaming=streaming,
        **CLIConfig(
            concurrency=1,
            request_count=10,
        ).model_dump(exclude_unset=True),
        search_recipe=recipe_name,
        **sweeping_overrides,
    )
    return convert_cli_to_aiperf(user)


@pytest.mark.parametrize(
    ("recipe_name", "streaming", "loadgen_overrides"),
    [
        ("max-throughput-ttft-sla", True, {"ttft_sla_ms": 200.0}),
        ("max-throughput-itl-sla", True, {"itl_sla_ms": 50.0}),
        (
            "max-goodput-under-slo",
            True,
            {"ttft_sla_ms": 200.0, "tpot_sla_ms": 50.0, "e2e_sla_ms": 2000.0},
        ),
        ("max-concurrency-under-sla", False, {"e2e_sla_ms": 2000.0}),
    ],
)  # fmt: skip
def test_v1_converter_expands_adaptive_search_recipes(
    recipe_name: str, streaming: bool, loadgen_overrides: dict[str, object]
) -> None:
    config = _convert_recipe(recipe_name, streaming=streaming, **loadgen_overrides)

    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.recipe_name == recipe_name
    assert config.multi_run is not None


@pytest.mark.parametrize(
    ("recipe_name", "streaming", "loadgen_overrides"),
    [
        ("concurrency-ramp", False, {}),
        ("prefill-ttft-curve", True, {}),
        ("decode-itl-curve", True, {}),
        (
            "max-concurrency-under-sla",
            False,
            {"e2e_sla_ms": 2000.0, "search_style": "grid"},
        ),
    ],
)  # fmt: skip
def test_v1_converter_expands_grid_recipes_with_sweep_metadata(
    recipe_name: str, streaming: bool, loadgen_overrides: dict[str, object]
) -> None:
    config = _convert_recipe(recipe_name, streaming=streaming, **loadgen_overrides)

    assert isinstance(config.sweep, GridSweep)
    assert config.sweep.parameters
    assert config.sweep.post_process is not None
    assert config.multi_run is not None


def test_v1_converter_promotes_magic_list_concurrency_without_recipe() -> None:
    """Bug B regression: list-shaped --concurrency must promote to a sweep
    block even when no --search-recipe is set. The converter previously built
    a speculative BenchmarkConfig before magic-list promotion, which rejected
    the list against the scalar phase field.
    """
    user = CLIConfig(
        model_names=["test-model"],
        streaming=False,
        **CLIConfig(
            concurrency=[1, 2, 4],
            request_count=10,
        ).model_dump(exclude_unset=True),
    )

    config = convert_cli_to_aiperf(user)

    assert isinstance(config.sweep, GridSweep)
    assert config.sweep.parameters
    # The list value must land under the dotted phase path the sweep
    # expander consumes, not on the phase scalar.
    matched = [key for key in config.sweep.parameters if key.endswith(".concurrency")]
    assert matched, (
        f"expected a phases.<name>.concurrency variable, got {config.sweep.parameters!r}"
    )
    assert config.sweep.parameters[matched[0]] == [1, 2, 4]


def test_v1_converter_parameter_sweep_mode_with_adaptive_recipe_does_not_crash() -> (
    None
):
    """Issue #5: ``--parameter-sweep-mode`` + an adaptive-search recipe must
    not crash Pydantic.

    ``_apply_parameter_sweep_meta_to_sweep`` previously stamped
    ``iteration_order`` / ``same_seed`` unconditionally onto whatever sweep
    block was emitted, but ``AdaptiveSearchSweep`` inherits from
    ``_SweepBase`` (not ``_GridSweepBase``) and uses ``extra="forbid"``, so
    those keys are rejected. Gate the stamp to grid/scenarios sweeps.
    ``cooldown_seconds`` lives on ``_SweepBase`` and applies to all sweep
    types, so it remains stamped.
    """
    from aiperf.common.enums import SweepMode

    user = CLIConfig(
        model_names=["test-model"],
        streaming=True,
        **CLIConfig(
            concurrency=1,
            request_count=10,
        ).model_dump(exclude_unset=True),
        search_recipe="max-throughput-ttft-sla",
        ttft_sla_ms=200.0,
        parameter_sweep_mode=SweepMode.INDEPENDENT,
        parameter_sweep_same_seed=True,
        parameter_sweep_cooldown_seconds=5.0,
    )

    config = convert_cli_to_aiperf(user)

    assert isinstance(config.sweep, AdaptiveSearchSweep)
    # Grid-only meta keys must NOT be stamped onto an adaptive sweep --
    # AdaptiveSearchSweep declares neither field, and stamping would have
    # crashed Pydantic on ``extra="forbid"``.
    sweep_dump = config.sweep.model_dump()
    assert "iteration_order" not in sweep_dump
    assert "same_seed" not in sweep_dump
    # ``cooldown_seconds`` lives on ``_SweepBase`` and is valid here.
    assert config.sweep.cooldown_seconds == 5.0


def test_v1_converter_parameter_sweep_mode_still_stamps_grid_sweep() -> None:
    """Sanity counterpart to issue #5: on a grid sweep the stamp still lands."""
    from aiperf.common.enums import SweepMode

    user = CLIConfig(
        model_names=["test-model"],
        streaming=False,
        **CLIConfig(concurrency=[1, 2, 4], request_count=10).model_dump(
            exclude_unset=True
        ),
        parameter_sweep_mode=SweepMode.INDEPENDENT,
        parameter_sweep_same_seed=True,
        parameter_sweep_cooldown_seconds=5.0,
    )

    config = convert_cli_to_aiperf(user)

    assert isinstance(config.sweep, GridSweep)
    assert config.sweep.iteration_order == SweepMode.INDEPENDENT
    assert config.sweep.same_seed is True
    assert config.sweep.cooldown_seconds == 5.0


# =============================================================================
# Fix #1: --itl-sla-ms / --tpot-sla-ms alias coverage
# =============================================================================


@pytest.mark.parametrize(
    ("recipe_name", "streaming", "loadgen_overrides"),
    [
        param(
            "max-throughput-itl-sla",
            True,
            {"itl_sla_ms": 50.0},
            id="itl_recipe_with_itl_flag",
        ),
        param(
            "max-throughput-itl-sla",
            True,
            {"tpot_sla_ms": 50.0},
            id="itl_recipe_with_tpot_alias",
        ),
        param(
            "max-throughput-itl-sla",
            True,
            {"itl_sla_ms": 50.0, "tpot_sla_ms": 50.0},
            id="itl_recipe_with_both_same_value",
        ),
        param(
            "max-concurrency-under-sla",
            True,
            {"itl_sla_ms": 50.0},
            id="max_concurrency_with_itl_flag",
        ),
        param(
            "max-concurrency-under-sla",
            True,
            {"tpot_sla_ms": 50.0},
            id="max_concurrency_with_tpot_alias",
        ),
        param(
            "max-goodput-under-slo",
            True,
            {"ttft_sla_ms": 200.0, "itl_sla_ms": 50.0, "e2e_sla_ms": 2000.0},
            id="goodput_with_itl_alias",
        ),
        param(
            "max-goodput-under-slo",
            True,
            {"ttft_sla_ms": 200.0, "tpot_sla_ms": 50.0, "e2e_sla_ms": 2000.0},
            id="goodput_with_tpot",
        ),
    ],
)  # fmt: skip
def test_v1_converter_accepts_itl_and_tpot_as_aliases(
    recipe_name: str,
    streaming: bool,
    loadgen_overrides: dict[str, object],
) -> None:
    """The two flags map to the same inter-token-latency SLA; either
    (or both with the same value) must successfully expand the recipe."""
    config = _convert_recipe(recipe_name, streaming=streaming, **loadgen_overrides)
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.recipe_name == recipe_name


@pytest.mark.parametrize(
    "recipe_name",
    [
        "max-throughput-itl-sla",
        "max-concurrency-under-sla",
        "max-goodput-under-slo",
    ],
)
def test_v1_converter_rejects_conflicting_itl_and_tpot_values(
    recipe_name: str,
) -> None:
    """Both flags set with different values must raise the alias-conflict
    error so users don't silently get one threshold while specifying another."""
    extras: dict[str, object] = {"itl_sla_ms": 50.0, "tpot_sla_ms": 60.0}
    if recipe_name == "max-goodput-under-slo":
        extras["ttft_sla_ms"] = 200.0
        extras["e2e_sla_ms"] = 2000.0
    with pytest.raises(ValueError, match=r"--tpot-sla-ms and --itl-sla-ms"):
        _convert_recipe(recipe_name, streaming=True, **extras)


# =============================================================================
# Fix #2: tunable --search-* flags override recipe defaults
# =============================================================================


def test_v1_converter_search_max_iterations_overrides_recipe_default() -> None:
    """``--search-max-iterations N`` layers on top of a BO recipe's default
    iteration budget without rejecting the recipe."""
    config = _convert_recipe(
        "max-throughput-ttft-sla",
        streaming=True,
        ttft_sla_ms=200.0,
        search_max_iterations=100,
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.max_iterations == 100


def test_v1_converter_search_random_seed_overrides_recipe_default() -> None:
    """``--search-random-seed`` flows through onto the recipe's adaptive sweep."""
    config = _convert_recipe(
        "max-throughput-ttft-sla",
        streaming=True,
        ttft_sla_ms=200.0,
        search_random_seed=42,
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.random_seed == 42


def test_v1_converter_search_initial_points_overrides_recipe_default() -> None:
    """``--search-initial-points`` flows through onto the recipe's adaptive sweep."""
    config = _convert_recipe(
        "max-throughput-ttft-sla",
        streaming=True,
        ttft_sla_ms=200.0,
        search_initial_points=11,
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.n_initial_points == 11


def test_v1_converter_recipe_defining_flags_still_rejected() -> None:
    """``--search-stat`` (recipe-defining) still raises with the
    explicit-flags-mutex error even though the budget knobs are now allowed."""
    with pytest.raises(TypeError, match=r"mutually exclusive"):
        _convert_recipe(
            "max-throughput-ttft-sla",
            streaming=True,
            ttft_sla_ms=200.0,
            search_stat="p99",
        )


def test_v1_converter_tunable_flag_against_grid_recipe_raises() -> None:
    """Tunable budget flags are BO-only; combining one with a grid recipe must
    raise a clear error rather than silently no-op."""
    with pytest.raises(TypeError, match=r"grid sweep"):
        _convert_recipe(
            "concurrency-ramp",
            streaming=False,
            search_max_iterations=100,
        )


# =============================================================================
# Fix #3: --search-style optuna for max-concurrency-under-sla
# =============================================================================


def test_v1_converter_max_concurrency_under_sla_optuna_style() -> None:
    """``--search-style optuna`` produces an AdaptiveSearchSweep dispatched
    to the Optuna planner so users can opt into TPE/GP/BoTorch samplers
    without dropping the recipe."""
    config = _convert_recipe(
        "max-concurrency-under-sla",
        streaming=True,
        ttft_sla_ms=200.0,
        search_style="optuna",
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    assert config.sweep.planner == SearchPlannerType.OPTUNA
    assert config.sweep.sla_filters


# =============================================================================
# Fix #1 (#1): orphaned override flags now flow through to recipes
# =============================================================================


def test_v1_converter_concurrency_max_overrides_decode_itl_curve_grid() -> None:
    """``--concurrency-max`` reaches DecodeITLCurve via ctx.sweep_overrides."""
    config = _convert_recipe(
        "decode-itl-curve",
        streaming=True,
        concurrency_max=512,
    )
    assert isinstance(config.sweep, GridSweep)
    matched = [k for k in config.sweep.parameters if k.endswith(".concurrency")]
    assert matched, f"expected concurrency variable, got {config.sweep.parameters!r}"
    values = config.sweep.parameters[matched[0]]
    # Default lo=1, default steps=6, override hi=512: last value must be 512.
    assert values[-1] == 512
    assert values[0] == 1


def test_v1_converter_osl_overrides_decode_itl_curve_grid() -> None:
    """``--osl-min`` / ``--osl-steps`` shape the decode-itl-curve OSL grid."""
    config = _convert_recipe(
        "decode-itl-curve",
        streaming=True,
        osl_min=128,
        osl_max=2048,
        osl_steps=8,
    )
    assert isinstance(config.sweep, GridSweep)
    osl_keys = [k for k in config.sweep.parameters if k.endswith(".osl")]
    assert osl_keys, f"expected osl variable, got {config.sweep.parameters!r}"
    values = config.sweep.parameters[osl_keys[0]]
    assert values[0] == 128
    assert values[-1] == 2048
    assert len(values) == 8


def test_v1_converter_isl_steps_overrides_prefill_ttft_curve() -> None:
    """``--isl-steps 12`` produces 12 grid values on the prefill-ttft-curve ISL axis."""
    config = _convert_recipe(
        "prefill-ttft-curve",
        streaming=True,
        isl_steps=12,
    )
    assert isinstance(config.sweep, GridSweep)
    isl_keys = [k for k in config.sweep.parameters if k.endswith(".isl")]
    assert isl_keys, f"expected isl variable, got {config.sweep.parameters!r}"
    assert len(config.sweep.parameters[isl_keys[0]]) == 12


def test_v1_converter_concurrency_steps_overrides_concurrency_ramp() -> None:
    """``--concurrency-steps 5`` shapes the concurrency-ramp grid."""
    config = _convert_recipe(
        "concurrency-ramp",
        streaming=False,
        concurrency_min=2,
        concurrency_max=128,
        concurrency_steps=5,
    )
    assert isinstance(config.sweep, GridSweep)
    matched = [k for k in config.sweep.parameters if k.endswith(".concurrency")]
    assert matched, f"expected concurrency variable, got {config.sweep.parameters!r}"
    values = config.sweep.parameters[matched[0]]
    assert len(values) == 5
    assert values[0] == 2
    assert values[-1] == 128


# =============================================================================
# Fix #16: ConcurrencyRamp post-process metric/stat overridable
# =============================================================================


def test_v1_converter_degradation_metric_and_stat_reach_concurrency_ramp() -> None:
    """``--degradation-metric-tag`` / ``--degradation-stat`` flow into the
    ConcurrencyRamp recipe's degradation_knee_detect PostProcessSpec params."""
    config = _convert_recipe(
        "concurrency-ramp",
        streaming=True,
        degradation_metric_tag="time_to_first_token",
        degradation_stat="p95",
    )
    assert isinstance(config.sweep, GridSweep)
    assert config.sweep.post_process is not None
    assert config.sweep.post_process.handler == "degradation_knee_detect"
    assert config.sweep.post_process.params["metric_tag"] == "time_to_first_token"
    assert config.sweep.post_process.params["stat"] == "p95"


def test_v1_converter_concurrency_ramp_default_post_process_metric_and_stat() -> None:
    """Without overrides, ConcurrencyRamp keeps the historical defaults
    (``request_latency`` / ``p99``); guards #16 against silent default drift."""
    config = _convert_recipe("concurrency-ramp", streaming=False)
    assert isinstance(config.sweep, GridSweep)
    assert config.sweep.post_process is not None
    assert config.sweep.post_process.params["metric_tag"] == "request_latency"
    assert config.sweep.post_process.params["stat"] == "p99"


# =============================================================================
# Fix #2 (#15): BO recipes honor concurrency_min/max overrides
# =============================================================================


@pytest.mark.parametrize(
    ("recipe_name", "loadgen_overrides"),
    [
        param(
            "max-throughput-ttft-sla",
            {"ttft_sla_ms": 200.0},
            id="ttft_sla",
        ),
        param(
            "max-throughput-itl-sla",
            {"itl_sla_ms": 50.0},
            id="itl_sla",
        ),
        param(
            "max-goodput-under-slo",
            {
                "ttft_sla_ms": 200.0,
                "tpot_sla_ms": 50.0,
                "e2e_sla_ms": 2000.0,
            },
            id="goodput_slo",
        ),
    ],
)  # fmt: skip
def test_v1_converter_bo_recipes_honor_concurrency_bounds(
    recipe_name: str, loadgen_overrides: dict[str, object]
) -> None:
    """BO recipes read concurrency_min/concurrency_max from sweep_overrides
    instead of hardcoding [1, 1000]."""
    config = _convert_recipe(
        recipe_name,
        streaming=True,
        concurrency_min=8,
        concurrency_max=512,
        **loadgen_overrides,
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    dim = config.sweep.search_space[0]
    assert dim.lo == 8
    assert dim.hi == 512


@pytest.mark.parametrize(
    "search_style",
    ["smooth_isotonic", "monotonic", "bo", "optuna"],
)
def test_v1_converter_max_concurrency_under_sla_styles_honor_bounds(
    search_style: str,
) -> None:
    """Every non-grid style branch on max-concurrency-under-sla must respect
    concurrency_min/concurrency_max overrides (#15)."""
    config = _convert_recipe(
        "max-concurrency-under-sla",
        streaming=True,
        ttft_sla_ms=200.0,
        search_style=search_style,
        concurrency_min=8,
        concurrency_max=512,
    )
    assert isinstance(config.sweep, AdaptiveSearchSweep)
    dim = config.sweep.search_space[0]
    assert dim.lo == 8
    assert dim.hi == 512


def test_v1_converter_max_concurrency_under_sla_grid_honors_bounds() -> None:
    """The grid branch on max-concurrency-under-sla must also respect overrides."""
    config = _convert_recipe(
        "max-concurrency-under-sla",
        streaming=False,
        e2e_sla_ms=2000.0,
        search_style="grid",
        concurrency_min=4,
        concurrency_max=256,
        concurrency_steps=8,  # currently unused by this branch but harmless
    )
    assert isinstance(config.sweep, GridSweep)
    matched = [k for k in config.sweep.parameters if k.endswith(".concurrency")]
    assert matched, f"expected concurrency variable, got {config.sweep.parameters!r}"
    values = config.sweep.parameters[matched[0]]
    assert values[0] == 4
    assert values[-1] == 256


@pytest.mark.parametrize(
    ("recipe_name", "extra"),
    [
        param("max-throughput-ttft-sla", {"ttft_sla_ms": 200.0}, id="ttft_sla"),
        param("max-throughput-itl-sla", {"itl_sla_ms": 50.0}, id="itl_sla"),
        param(
            "max-goodput-under-slo",
            {"ttft_sla_ms": 200.0, "tpot_sla_ms": 50.0, "e2e_sla_ms": 2000.0},
            id="goodput_slo",
        ),
        param(
            "max-concurrency-under-sla",
            {"ttft_sla_ms": 200.0},
            id="max_concurrency_smooth_isotonic",
        ),
        param(
            "decode-itl-curve",
            {},
            id="decode_itl_curve_grid",
        ),
        param(
            "concurrency-ramp",
            {},
            id="concurrency_ramp_grid",
        ),
    ],
)  # fmt: skip
def test_v1_converter_inverted_concurrency_bounds_raise(
    recipe_name: str, extra: dict[str, object]
) -> None:
    """Inverted --concurrency-min/--concurrency-max must raise a clear error
    instead of silently producing a degenerate search space (#15)."""
    streaming = recipe_name != "concurrency-ramp"
    with pytest.raises(ValueError, match=r"concurrency-min.*must be <"):
        _convert_recipe(
            recipe_name,
            streaming=streaming,
            concurrency_min=100,
            concurrency_max=50,
            **extra,
        )
