# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared hypothesis strategies for AIPerf property tests.

The strategies here intentionally produce both well-formed and adversarial
values (NaN, infinities, very large/small floats, empty/long strings,
control characters, negative ints) to drive the "validation never raises
an unhandled exception" property in :mod:`test_pydantic_field_fuzz`.

All strategies are deterministic under hypothesis' default seed; tests use
``settings(max_examples=...)`` and ``derandomize=True`` to keep CI flakes
bounded.
"""

from __future__ import annotations

import math
import string

from hypothesis import strategies as st

# ----------------------------------------------------------------------------
# Primitive strategies
# ----------------------------------------------------------------------------


def adversarial_floats() -> st.SearchStrategy[float]:
    """Floats including NaN, +/-inf, very-large, very-small, zero, negatives.

    Hypothesis' default ``st.floats()`` with ``allow_nan=True`` and
    ``allow_infinity=True`` covers every case we care about for fuzzing
    Pydantic numeric fields. We OR-in a tight set of subnormals/edge
    constants so the small-example shrinking gravitates toward them.
    """
    return st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.sampled_from(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                math.nan,
                math.inf,
                -math.inf,
                1e-300,
                -1e-300,
                1e300,
                -1e300,
                2.2250738585072014e-308,  # smallest normal
                5e-324,  # smallest subnormal
            ]
        ),
    )


def finite_floats() -> st.SearchStrategy[float]:
    """Finite floats only (no NaN/inf), useful for cross-field constraints."""
    return st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10,
    )


def adversarial_ints() -> st.SearchStrategy[int]:
    """Integers including negatives, zero, small, and very-large absolute values."""
    return st.one_of(
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        st.sampled_from([-1, 0, 1, 2, 2**31, -(2**31), 2**62, -(2**62)]),
    )


def adversarial_strings() -> st.SearchStrategy[str]:
    """Strings including empty, whitespace, control chars, very-long, and ascii."""
    return st.one_of(
        st.text(max_size=64),
        st.text(alphabet=string.printable, max_size=128),
        st.sampled_from(
            [
                "",
                " ",
                "\t",
                "\n",
                "\x00",
                "..",
                ".",
                "a",
                "a." * 32,
                "x" * 256,
                "phases.profiling.concurrency",
                "phases.profiling.rate",
            ]
        ),
    )


def dotted_paths() -> st.SearchStrategy[str]:
    """Plausible dotted paths (well-formed AND malformed).

    Covers the validator's reject-cases (empty, leading/trailing dot,
    consecutive dots, envelope-level prefixes) AND the success path
    (valid body-rooted paths).
    """
    valid_segment = st.text(
        alphabet=string.ascii_lowercase + string.digits + "_",
        min_size=1,
        max_size=12,
    )
    well_formed = st.lists(valid_segment, min_size=1, max_size=4).map(".".join)
    return st.one_of(
        well_formed,
        st.sampled_from(
            [
                "",
                ".",
                "..",
                "..foo",
                "foo.",
                "foo..bar",
                "sweep.x",
                "multi_run.num_runs",
                "random_seed",
                "benchmark.phases",
                "phases.profiling.concurrency",
                "phases.warmup.rate",
            ]
        ),
    )


# ----------------------------------------------------------------------------
# Pydantic-model input strategies
#
# Each strategy returns ``dict``s suitable for ``Model.model_validate(...)``.
# They mix well-formed and adversarial fields so the property
# "validation never raises an unhandled exception" exercises both branches.
# ----------------------------------------------------------------------------


def sampling_dimension_inputs() -> st.SearchStrategy[dict]:
    """Inputs for SamplingDimension (and SearchSpaceDimension overlap)."""
    return st.fixed_dictionaries(
        {
            "path": dotted_paths(),
        },
        optional={
            "lo": st.one_of(st.none(), adversarial_floats()),
            "hi": st.one_of(st.none(), adversarial_floats()),
            "scale": st.sampled_from(["linear", "log", "bogus"]),
            "kind": st.sampled_from(["int", "real", "bogus"]),
            "choices": st.one_of(
                st.none(),
                st.lists(
                    st.one_of(
                        st.integers(),
                        adversarial_floats(),
                        st.text(max_size=8),
                        st.none(),
                    ),
                    max_size=8,
                ),
            ),
        },
    )


def search_space_dimension_inputs() -> st.SearchStrategy[dict]:
    """Inputs for SearchSpaceDimension (BO)."""
    return st.fixed_dictionaries(
        {
            "path": dotted_paths(),
            "lo": adversarial_floats(),
            "hi": adversarial_floats(),
        },
        optional={
            "kind": st.sampled_from(["int", "real", "bogus"]),
        },
    )


def sla_filter_inputs() -> st.SearchStrategy[dict]:
    """Dict inputs that exercise SLAFilter validation branches.

    Generates valid and bogus operators/stats plus adversarial numeric thresholds
    so property tests cover both accepted filters and clean validation failures.
    """
    return st.fixed_dictionaries(
        {
            "metric_tag": adversarial_strings(),
            "op": st.sampled_from(["lt", "le", "gt", "ge", "bogus"]),
            "threshold": adversarial_floats(),
        },
        optional={
            "stat": st.sampled_from(["avg", "p50", "p90", "p95", "p99", "bogus"]),
        },
    )


def adaptive_objective_inputs() -> st.SearchStrategy[dict]:
    """Inputs for Objective."""
    return st.fixed_dictionaries(
        {
            "metric": adversarial_strings(),
            "direction": st.sampled_from(
                ["maximize", "minimize", "MAXIMIZE", "MINIMIZE"]
            ),
        },
        optional={
            "stat": st.sampled_from(["avg", "p50", "p90", "p95", "p99", "bogus"]),
        },
    )


def grid_sweep_inputs() -> st.SearchStrategy[dict]:
    """Inputs for GridSweep."""
    return st.fixed_dictionaries(
        {
            "type": st.sampled_from(["grid", "wrong"]),
            "parameters": st.dictionaries(
                keys=dotted_paths(),
                values=st.lists(
                    st.one_of(
                        adversarial_ints(), adversarial_floats(), adversarial_strings()
                    ),
                    max_size=4,
                ),
                max_size=4,
            ),
        },
        optional={
            "cooldown_seconds": adversarial_floats(),
            "same_seed": st.booleans(),
            "iteration_order": st.sampled_from(["repeated", "independent", "bogus"]),
        },
    )


def scenario_sweep_inputs() -> st.SearchStrategy[dict]:
    """Inputs for ScenarioSweep."""
    return st.fixed_dictionaries(
        {
            "type": st.sampled_from(["scenarios", "wrong"]),
            "runs": st.lists(
                st.dictionaries(
                    keys=adversarial_strings(),
                    values=st.one_of(
                        adversarial_ints(),
                        adversarial_floats(),
                        adversarial_strings(),
                    ),
                    max_size=4,
                ),
                max_size=4,
            ),
        },
        optional={
            "cooldown_seconds": adversarial_floats(),
        },
    )


def sobol_sweep_inputs() -> st.SearchStrategy[dict]:
    """Inputs for SobolSweep / LatinHypercubeSweep (samples + dims)."""
    return st.fixed_dictionaries(
        {
            "type": st.sampled_from(["sobol", "latin_hypercube", "wrong"]),
            "samples": adversarial_ints(),
            "dimensions": st.lists(sampling_dimension_inputs(), max_size=4),
        },
        optional={
            "seed": st.one_of(st.none(), adversarial_ints()),
            "label_format": st.sampled_from(["index", "kv", "bogus"]),
            "scramble": st.booleans(),
            "optimization": st.sampled_from([None, "random-cd", "lloyd", "bogus"]),
        },
    )


def adaptive_search_sweep_inputs() -> st.SearchStrategy[dict]:
    """Inputs for AdaptiveSearchSweep."""
    return st.fixed_dictionaries(
        {
            "type": st.sampled_from(["adaptive_search", "wrong"]),
            "max_iterations": adversarial_ints(),
            "search_space": st.lists(search_space_dimension_inputs(), max_size=4),
            "objectives": st.lists(adaptive_objective_inputs(), min_size=1, max_size=1),
        },
        optional={
            "n_initial_points": adversarial_ints(),
            "plateau_window": adversarial_ints(),
            "plateau_threshold": adversarial_floats(),
            "improvement_patience": adversarial_ints(),
            "random_seed": st.one_of(st.none(), adversarial_ints()),
            "constraint_mode": st.sampled_from(["penalty", "eic", "bogus"]),
            "planner": st.sampled_from(
                ["bayesian", "optuna", "monotonic_sla", "smooth_isotonic", "bogus"]
            ),
        },
    )


def multi_run_inputs() -> st.SearchStrategy[dict]:
    """Inputs for MultiRunConfig."""
    return st.fixed_dictionaries(
        {},
        optional={
            "num_runs": adversarial_ints(),
            "cooldown_seconds": adversarial_floats(),
            "confidence_level": adversarial_floats(),
            "set_consistent_seed": st.booleans(),
            "disable_warmup_after_first": st.booleans(),
            "convergence": st.one_of(
                st.none(),
                st.fixed_dictionaries(
                    {
                        "metric": adversarial_strings(),
                    },
                    optional={
                        "stat": st.sampled_from(["avg", "p50", "p90", "p95", "p99"]),
                        "mode": st.sampled_from(
                            ["ci_width", "cv", "distribution", "bogus"]
                        ),
                        "threshold": adversarial_floats(),
                        "min_runs": adversarial_ints(),
                    },
                ),
            ),
        },
    )


def fixed_distribution_inputs() -> st.SearchStrategy:
    """Scalar OR dict input that should resolve to FixedDistribution."""
    return st.one_of(
        adversarial_ints(),
        adversarial_floats(),
        st.fixed_dictionaries(
            {"value": adversarial_floats()},
            optional={
                "min": st.one_of(st.none(), adversarial_floats()),
                "max": st.one_of(st.none(), adversarial_floats()),
            },
        ),
    )


def normal_distribution_inputs() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {"mean": adversarial_floats()},
        optional={
            "stddev": adversarial_floats(),
            "min": st.one_of(st.none(), adversarial_floats()),
            "max": st.one_of(st.none(), adversarial_floats()),
        },
    )


def lognormal_distribution_inputs() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries(
        {
            "mean": adversarial_floats(),
            "median": adversarial_floats(),
        },
        optional={
            "min": st.one_of(st.none(), adversarial_floats()),
            "max": st.one_of(st.none(), adversarial_floats()),
        },
    )


def multimodal_distribution_inputs() -> st.SearchStrategy[dict]:
    peak = st.fixed_dictionaries(
        {"mean": adversarial_floats(), "stddev": adversarial_floats()},
        optional={"weight": adversarial_floats()},
    )
    return st.fixed_dictionaries(
        {"peaks": st.lists(peak, max_size=4)},
        optional={
            "min": st.one_of(st.none(), adversarial_floats()),
            "max": st.one_of(st.none(), adversarial_floats()),
        },
    )


def empirical_distribution_inputs() -> st.SearchStrategy[dict]:
    point = st.fixed_dictionaries(
        {"value": adversarial_floats()},
        optional={"weight": adversarial_floats()},
    )
    return st.fixed_dictionaries(
        {"points": st.lists(point, max_size=4)},
        optional={
            "min": st.one_of(st.none(), adversarial_floats()),
            "max": st.one_of(st.none(), adversarial_floats()),
        },
    )


def endpoint_inputs() -> st.SearchStrategy[dict]:
    """Inputs for v1 EndpointConfig."""
    return st.fixed_dictionaries(
        {},
        optional={
            "model_names": st.one_of(
                adversarial_strings(),
                st.lists(adversarial_strings(), max_size=4),
            ),
            "type": st.sampled_from(
                ["chat", "completions", "embeddings", "rankings", "bogus"]
            ),
            "streaming": st.booleans(),
            "urls": st.lists(adversarial_strings(), max_size=3),
            "timeout_seconds": adversarial_floats(),
            "wait_for_model_timeout": adversarial_floats(),
            "wait_for_model_interval": adversarial_floats(),
            "api_key": st.one_of(st.none(), adversarial_strings()),
        },
    )
