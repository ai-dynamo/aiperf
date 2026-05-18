# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``SearchRecipe.auto_plot_default`` ClassVar.

The recipes whose output is a curve worth visualizing immediately
(``concurrency-ramp``, ``prefill-ttft-curve``, ``decode-itl-curve``) opt in by
setting ``auto_plot_default = True``. All other recipes leave it unset; the
read site falls back to ``False`` via ``getattr(recipe, "auto_plot_default",
False)``. These tests pin both halves of that contract so a recipe can't
silently change its post-run plotting behavior.
"""

from __future__ import annotations

from aiperf.search_recipes.builtins import (
    ConcurrencyRamp,
    DecodeITLCurve,
    MaxThroughputUnderTTFTSLA,
    PrefillTTFTCurve,
)


def test_concurrency_ramp_auto_plot_default_is_true():
    assert ConcurrencyRamp.auto_plot_default is True


def test_prefill_ttft_curve_auto_plot_default_is_true():
    assert PrefillTTFTCurve.auto_plot_default is True


def test_decode_itl_curve_auto_plot_default_is_true():
    assert DecodeITLCurve.auto_plot_default is True


def test_max_throughput_ttft_sla_does_not_define_auto_plot_default():
    # Non-curve recipes (search for an optimum, not a curve) MUST NOT define
    # the attribute -- the read site relies on the getattr fallback to False.
    assert "auto_plot_default" not in vars(MaxThroughputUnderTTFTSLA)


def test_getattr_fallback_returns_false_for_recipe_without_attr():
    # Mirrors the read site at the v1->v2 converter / cli_runner boundary:
    # ``getattr(recipe, "auto_plot_default", False)`` must yield False for any
    # recipe that hasn't opted in, including external plugin recipes.
    assert getattr(MaxThroughputUnderTTFTSLA, "auto_plot_default", False) is False
