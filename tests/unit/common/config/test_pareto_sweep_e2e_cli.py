# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CLI -> envelope test for the pareto-sweep recipe.

Asserts that:
  1. CLIConfig with --search-recipe pareto-sweep + --isl-osl-pairs +
     --concurrency 1,2,4 successfully converts to a top-level AIPerfConfig
  2. The resulting ``sweep`` block is a ``ScenarioSweep`` with
     pairs x concurrency rows
  3. Each scenario carries the expected (isl, osl, concurrency) shape

Integration with the mock server is out of scope; that is covered by the
existing ScenarioSweep integration tests.
"""

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.sweep import ScenarioSweep


def _build_user(pairs: str, concurrency: list[int] | int | None) -> CLIConfig:
    """Construct the v1 CLIConfig the CLI would produce for the pareto-sweep flow."""
    loadgen = (
        CLIConfig(concurrency=concurrency, request_count=10)
        if concurrency is not None
        else CLIConfig(request_count=10)
    )
    return CLIConfig(
        model_names=["test-model"],
        streaming=True,
        **loadgen.model_dump(exclude_unset=True),
        search_recipe="pareto-sweep",
        isl_osl_pairs=pairs,
    )


def test_pareto_sweep_cli_to_scenarios_envelope() -> None:
    user = _build_user(pairs="128/128,256/256", concurrency=[1, 2, 4])
    config = convert_cli_to_aiperf(user)
    assert isinstance(config.sweep, ScenarioSweep)
    assert len(config.sweep.runs) == 2 * 3  # pairs x concurrencies
    names = {r["name"] for r in config.sweep.runs}
    assert "shape_128_128_c1" in names
    assert "shape_256_256_c4" in names


def test_pareto_sweep_cli_default_concurrency_when_omitted() -> None:
    user = _build_user(pairs="128/128,256/256", concurrency=None)
    config = convert_cli_to_aiperf(user)
    assert isinstance(config.sweep, ScenarioSweep)
    assert (
        len(config.sweep.runs) == 2 * 5
    )  # pairs x default-concurrency (1,4,16,64,256)


def test_pareto_sweep_cli_scalar_concurrency_passes_through() -> None:
    user = _build_user(pairs="128/128,256/256", concurrency=64)
    config = convert_cli_to_aiperf(user)
    assert isinstance(config.sweep, ScenarioSweep)
    runs = config.sweep.runs
    assert len(runs) == 2
    for r in runs:
        phases = r["benchmark"]["phases"]
        assert phases[0]["concurrency"] == 64
