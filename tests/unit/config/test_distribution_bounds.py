# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""min/max bounds on Distribution base class.

Bounds clamp samples post-draw. Every distribution type supports them
via base-class wrapping; subclasses provide _sample_raw, base provides
sample with clamp.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.common import random_generator as rng
from aiperf.config.distributions import (
    NormalDistribution,
    SamplingDistribution,
)

_DIST = TypeAdapter(SamplingDistribution)


def _get_rng() -> rng.RandomGenerator:
    return rng.derive("test.distribution_bounds")


def _draw_many(dist, n: int = 200) -> list[float]:
    gen = _get_rng()
    return [dist.sample(gen) for _ in range(n)]


@pytest.mark.parametrize(
    "payload, lo, hi",
    [
        ({"mean": 256, "stddev": 80, "min": 100, "max": 400}, 100, 400),
        ({"value": 50, "min": 100, "max": 400}, 100, 400),  # value below min -> clamped up
        ({"value": 500, "min": 100, "max": 400}, 100, 400),  # value above max -> clamped down
        ({"mean": 256, "median": 200, "min": 50, "max": 500}, 50, 500),
        (
            {
                "peaks": [{"mean": 64, "stddev": 10}, {"mean": 512, "stddev": 50}],
                "min": 100,
                "max": 400,
            },
            100,
            400,
        ),
        (
            {
                "points": [{"value": 50, "weight": 50}, {"value": 600, "weight": 50}],
                "min": 100,
                "max": 400,
            },
            100,
            400,
        ),
    ],
    ids=["normal", "fixed-below", "fixed-above", "lognormal", "multimodal", "empirical"],
)  # fmt: skip
def test_sample_respects_bounds(payload, lo, hi):
    dist = _DIST.validate_python(payload)
    samples = _draw_many(dist)
    assert all(lo <= s <= hi for s in samples), (
        f"out-of-bounds sample(s) found in {samples!r}; expected all in [{lo}, {hi}]"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"mean": 256, "stddev": 80, "min": 400, "max": 100},  # min > max
        {"mean": 256, "stddev": 80, "min": "low"},  # bad type
        {"mean": 256, "stddev": 80, "max": float("inf")},  # non-finite
        {"mean": 256, "stddev": 80, "min": float("nan")},  # nan
    ],
    ids=["min-gt-max", "min-bad-type", "max-inf", "min-nan"],
)
def test_invalid_bounds_rejected(payload):
    with pytest.raises((ValidationError, ValueError)):
        _DIST.validate_python(payload)


def test_only_min_bound_works():
    dist = _DIST.validate_python({"mean": 100, "stddev": 30, "min": 80})
    samples = _draw_many(dist)
    assert all(s >= 80 for s in samples)


def test_only_max_bound_works():
    dist = _DIST.validate_python({"mean": 100, "stddev": 30, "max": 130})
    samples = _draw_many(dist)
    assert all(s <= 130 for s in samples)


def test_no_bounds_unchanged_behavior():
    """Without min/max, samples are not clamped. Sanity guard against accidental defaults."""
    dist = _DIST.validate_python({"mean": 100, "stddev": 30})
    samples = _draw_many(dist)
    # If this was being clamped to e.g. [0, 100], we'd see no values >100. Verify a
    # spread including values >100 (probabilistically near-certain over 200 draws of
    # N(100, 30)).
    assert any(s > 100 for s in samples)


def test_sample_int_respects_bounds():
    dist = _DIST.validate_python({"mean": 256, "stddev": 80, "min": 100, "max": 400})
    gen = _get_rng()
    for _ in range(50):
        v = dist.sample_int(gen)
        assert 100 <= v <= 400


def test_explicit_type_with_bounds_works():
    """Explicit type: still works alongside bounds."""
    dist = _DIST.validate_python(
        {"type": "normal", "mean": 256, "stddev": 80, "min": 100, "max": 400}
    )
    assert isinstance(dist, NormalDistribution)
    samples = _draw_many(dist)
    assert all(100 <= s <= 400 for s in samples)
