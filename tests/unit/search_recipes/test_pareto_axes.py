# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ParetoAxesSpec."""

import pytest
from pydantic import ValidationError

from aiperf.search_recipes._pareto_axes import ParetoAxesSpec


def test_pareto_axes_spec_minimum_fields_pareto_sweep_shape():
    spec = ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
        series_keys=("isl", "osl"),
    )
    assert spec.x_minimize is True
    assert spec.y_maximize is True
    assert spec.series_keys == ("isl", "osl")


def test_pareto_axes_spec_single_series_default():
    spec = ParetoAxesSpec(
        x_metric="time_to_first_token",
        x_stat="p99",
        y_metric="output_token_throughput",
        y_stat="avg",
    )
    assert spec.series_keys == ()


def test_pareto_axes_spec_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ParetoAxesSpec(
            x_metric="x",
            x_stat="p95",
            y_metric="y",
            y_stat="avg",
            unexpected_field="boom",  # type: ignore[call-arg]
        )


def test_pareto_sweep_recipe_declares_axes():
    from aiperf.search_recipes._pareto_sweep import ParetoSweep

    axes = ParetoSweep.pareto_axes
    assert axes is not None
    assert axes.x_metric == "time_to_first_token"
    assert axes.x_stat == "p95"
    assert axes.y_metric == "output_token_throughput"
    assert axes.y_stat == "avg"
    assert axes.series_keys == ("isl", "osl")


def test_max_throughput_ttft_sla_declares_axes():
    from aiperf.search_recipes.builtins import MaxThroughputUnderTTFTSLA

    axes = MaxThroughputUnderTTFTSLA.pareto_axes
    assert axes is not None
    assert axes.x_metric == "time_to_first_token"
    assert axes.x_stat == "p99"
    assert axes.y_metric == "output_token_throughput"
    assert axes.y_stat == "avg"
    assert axes.series_keys == ()


def test_max_throughput_itl_sla_declares_axes():
    from aiperf.search_recipes.builtins import MaxThroughputUnderITLSLA

    axes = MaxThroughputUnderITLSLA.pareto_axes
    assert axes is not None
    assert axes.x_metric == "inter_token_latency"
    assert axes.x_stat == "p99"


def test_concurrency_ramp_declares_axes():
    from aiperf.search_recipes.builtins import ConcurrencyRamp

    axes = ConcurrencyRamp.pareto_axes
    assert axes is not None
    assert axes.x_metric == "request_latency"
    assert axes.y_metric == "output_token_throughput"


def test_max_goodput_under_slo_declares_axes():
    from aiperf.search_recipes._max_goodput_under_slo import MaxGoodputUnderSLO

    axes = MaxGoodputUnderSLO.pareto_axes
    assert axes is not None
    assert axes.y_metric == "goodput"


def test_max_concurrency_under_sla_declares_axes():
    from aiperf.search_recipes._max_concurrency_under_sla import MaxConcurrencyUnderSLA

    axes = MaxConcurrencyUnderSLA.pareto_axes
    assert axes is not None
    assert axes.y_metric == "concurrency"  # parameter-as-axis case


def test_curve_fit_recipes_dont_declare_axes():
    """Curve-fit recipes are out of scope for the Pareto plot path."""
    from aiperf.search_recipes.builtins import DecodeITLCurve, PrefillTTFTCurve

    assert PrefillTTFTCurve.pareto_axes is None
    assert DecodeITLCurve.pareto_axes is None
