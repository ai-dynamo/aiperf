# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: a YAML percentile ISL config produces a dataset whose sampled
ISLs hit the configured p50/p99/mean — the headline user scenario."""

import statistics

import pytest

# Reuse the composer construction helper (native BenchmarkConfig path that can
# express typed distributions) from the wiring regression suite.
from tests.unit.dataset.composer.test_distribution_wiring import _make_composer


def test_percentile_isl_end_to_end_hits_p50_p99_mean():
    composer = _make_composer(
        isl={"p50": 5000, "p99": 40000, "mean": 6000},
        osl=128,
        entries=6000,
    )
    composer.create_dataset()
    isls = sorted(p[0] for p in composer._turn_sequence_cache.values())
    n = len(isls)
    assert isls[n // 2] == pytest.approx(5000, rel=0.08)
    assert isls[int(n * 0.99)] == pytest.approx(40000, rel=0.12)
    assert statistics.fmean(isls) == pytest.approx(6000, rel=0.06)
