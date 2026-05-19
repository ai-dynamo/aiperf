# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MaxGoodputUnderSLO built-in recipe (DistServe formulation).

Closes the canonical-formulation gap to GH issue #883: maximizes goodput
under simultaneous TTFT + TPOT + E2E per-request SLOs with a configurable
attainment fraction (default 0.95). Targets the post-merge
``AdaptiveSearchSweep`` shape with nested ``Objective``.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.search_recipes.builtins import MaxGoodputUnderSLO
from tests.unit.search_recipes.conftest import make_ctx


def _full_sla_targets(**overrides: float) -> dict[str, float]:
    base = {"ttft_sla_ms": 500.0, "tpot_sla_ms": 15.0, "e2e_sla_ms": 2000.0}
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "missing_key,missing_flag",
    [
        param("ttft_sla_ms", "--ttft-sla-ms", id="missing-ttft"),
        param("tpot_sla_ms", "--tpot-sla-ms", id="missing-tpot"),
        param("e2e_sla_ms", "--e2e-sla-ms", id="missing-e2e"),
    ],
)  # fmt: skip
def test_max_goodput_under_slo_missing_required_sla_flag_raises(
    missing_key: str, missing_flag: str
) -> None:
    targets = _full_sla_targets()
    targets.pop(missing_key)
    with pytest.raises(ValueError) as exc:
        MaxGoodputUnderSLO().expand(make_ctx(sla_targets=targets))
    msg = str(exc.value)
    assert missing_flag in msg
    assert "max-goodput-under-slo" in msg


def test_max_goodput_under_slo_full_sla_targets_returns_adaptive_search() -> None:
    out = MaxGoodputUnderSLO().expand(make_ctx(sla_targets=_full_sla_targets()))
    assert out.adaptive_search is not None
    assert out.sweep_parameters is None
    assert out.adaptive_search.objectives[0].metric == "goodput"
    assert out.adaptive_search.objectives[0].direction == OptimizationDirection.MAXIMIZE
    assert out.adaptive_search.objectives[0].stat == "avg"


def test_max_goodput_under_slo_default_attainment_fraction_is_95_percent() -> None:
    out = MaxGoodputUnderSLO().expand(make_ctx(sla_targets=_full_sla_targets()))
    fraction_filters = [
        f for f in out.sla_filters if f.metric_tag == "good_request_fraction"
    ]
    assert len(fraction_filters) == 1
    assert fraction_filters[0].threshold == pytest.approx(0.95)
    assert fraction_filters[0].op == "ge"


def test_max_goodput_under_slo_custom_attainment_fraction_overrides_default() -> None:
    targets = _full_sla_targets()
    targets["slo_attainment_fraction"] = 0.99
    out = MaxGoodputUnderSLO().expand(make_ctx(sla_targets=targets))
    fraction_filters = [
        f for f in out.sla_filters if f.metric_tag == "good_request_fraction"
    ]
    assert fraction_filters[0].threshold == pytest.approx(0.99)


def test_max_goodput_under_slo_threads_per_request_slo_thresholds() -> None:
    """The three SLA flags become per-request `slos` keyed by metric tag."""
    out = MaxGoodputUnderSLO().expand(make_ctx(sla_targets=_full_sla_targets()))
    assert out.slos is not None
    assert out.slos["time_to_first_token"] == pytest.approx(500.0)
    assert out.slos["inter_token_latency"] == pytest.approx(15.0)
    assert out.slos["request_latency"] == pytest.approx(2000.0)


def test_max_goodput_under_slo_concurrency_search_space_is_one_to_thousand_int() -> (
    None
):
    out = MaxGoodputUnderSLO().expand(make_ctx(sla_targets=_full_sla_targets()))
    assert out.adaptive_search is not None
    dims = out.adaptive_search.search_space
    assert len(dims) == 1
    assert dims[0].path == "phases.profiling.concurrency"
    assert dims[0].lo == 1
    assert dims[0].hi == 1000
    assert dims[0].kind == "int"


def test_max_goodput_under_slo_streaming_explicitly_disabled_raises() -> None:
    with pytest.raises(ValueError, match="streaming"):
        MaxGoodputUnderSLO().expand(
            make_ctx(streaming=False, sla_targets=_full_sla_targets())
        )


def test_max_goodput_under_slo_resolves_through_plugin_registry() -> None:
    cls = plugins.get_class(PluginType.SEARCH_RECIPE, "max-goodput-under-slo")
    assert cls is MaxGoodputUnderSLO


def test_max_goodput_under_slo_name_and_description_are_classvars() -> None:
    assert MaxGoodputUnderSLO.name == "max-goodput-under-slo"
    assert "goodput" in MaxGoodputUnderSLO.description.lower()
