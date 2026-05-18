# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.unit.search_recipes.conftest import make_ctx


def test_pareto_sweep_consumes_concurrency_magic_list() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    assert "concurrency" in ParetoSweep.consumed_magic_lists


def test_pareto_sweep_default_concurrency_when_omitted() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/128,256/256")
    out = ParetoSweep().expand(ctx)
    assert out.scenarios is not None
    # 2 pairs x 5 default concurrency values = 10 cells
    assert len(out.scenarios) == 2 * 5


def test_pareto_sweep_scalar_concurrency() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/128,256/256,512/512", concurrency=64)
    out = ParetoSweep().expand(ctx)
    assert out.scenarios is not None
    assert len(out.scenarios) == 3
    for s in out.scenarios:
        phases = s["benchmark"]["phases"]
        assert phases[0]["concurrency"] == 64


def test_pareto_sweep_list_concurrency() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/128,256/256", concurrency=[1, 4, 16])
    out = ParetoSweep().expand(ctx)
    assert out.scenarios is not None
    assert len(out.scenarios) == 2 * 3
    names = [s["name"] for s in out.scenarios]
    assert "shape_128_128_c1" in names
    assert "shape_256_256_c16" in names


def test_pareto_sweep_each_scenario_carries_isl_osl_and_concurrency() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/64", concurrency=[1, 8])
    out = ParetoSweep().expand(ctx)
    assert out.scenarios is not None
    s = out.scenarios[0]
    prompts = s["benchmark"]["datasets"][0]["prompts"]
    assert prompts == {"isl": 128, "osl": 64}
    phases = s["benchmark"]["phases"]
    assert phases[0]["concurrency"] == 1


def test_pareto_sweep_emits_post_process() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/128,256/256", concurrency=[1, 4])
    out = ParetoSweep().expand(ctx)
    assert out.post_process is not None
    assert out.post_process.handler == "pareto_sweep_export"
    assert out.post_process.output_filename == "pareto_sweep.json"


def test_pareto_sweep_rejects_non_streaming() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(streaming=False, isl_osl_pairs="128/128,256/256")
    with pytest.raises(ValueError, match="streaming"):
        ParetoSweep().expand(ctx)


def test_pareto_sweep_rejects_missing_pairs() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx()
    with pytest.raises(ValueError, match="--isl-osl-pairs"):
        ParetoSweep().expand(ctx)


def test_pareto_sweep_rejects_single_cell() -> None:
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    ctx = make_ctx(isl_osl_pairs="128/128", concurrency=64)
    with pytest.raises(ValueError, match="single point"):
        ParetoSweep().expand(ctx)
