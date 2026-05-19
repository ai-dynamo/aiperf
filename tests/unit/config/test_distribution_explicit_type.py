# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict-but-tolerant `type:` key on Distribution YAML inputs.

When `type:` is present, the dispatch must use it AND the dict's other keys
must validate against that specific subclass. No aliasing — `type: mixture`
or `type: clamped` fails. When `type:` is absent, structural inference
(existing behavior) takes over.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from aiperf.config.distributions import (
    EmpiricalDistribution,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    SamplingDistribution,
)

_DIST_ADAPTER = TypeAdapter(SamplingDistribution)


@pytest.mark.parametrize(
    "payload, expected_cls",
    [
        # Explicit type matches structure
        ({"type": "fixed", "value": 256}, FixedDistribution),
        ({"type": "normal", "mean": 512, "stddev": 100}, NormalDistribution),
        ({"type": "lognormal", "mean": 512, "median": 400}, LogNormalDistribution),
        (
            {
                "type": "multimodal",
                "peaks": [
                    {"mean": 64, "stddev": 10},
                    {"mean": 512, "stddev": 50},
                ],
            },
            MultimodalDistribution,
        ),
        (
            {
                "type": "empirical",
                "points": [
                    {"value": 128, "weight": 40},
                    {"value": 256, "weight": 60},
                ],
            },
            EmpiricalDistribution,
        ),
        # No type — structural inference (existing behavior, sanity check)
        ({"mean": 512, "stddev": 100}, NormalDistribution),
        ({"mean": 512, "median": 400}, LogNormalDistribution),
        (256, FixedDistribution),
        (256.5, FixedDistribution),
        ({"value": 256}, FixedDistribution),
    ],
    ids=[
        "explicit-fixed",
        "explicit-normal",
        "explicit-lognormal",
        "explicit-multimodal",
        "explicit-empirical",
        "structural-normal",
        "structural-lognormal",
        "structural-fixed-int",
        "structural-fixed-float",
        "structural-fixed-value",
    ],
)  # fmt: skip
def test_explicit_type_matches_structure(payload, expected_cls):
    result = _DIST_ADAPTER.validate_python(payload)
    assert isinstance(result, expected_cls)


@pytest.mark.parametrize(
    "payload",
    [
        # Explicit type contradicts structure (fields don't match the chosen subclass)
        {"type": "normal", "median": 400},  # median is lognormal-only
        {"type": "lognormal", "mean": 512, "stddev": 100},  # stddev not on lognormal
        {"type": "fixed", "mean": 10},  # mean is normal-only; fixed needs value
        {"type": "fixed", "stddev": 10},  # stddev is normal-only
        {"type": "normal", "peaks": []},  # peaks is multimodal-only
        {"type": "empirical", "value": 128},  # value is fixed-only
        # Unknown explicit type — no aliasing
        {"type": "mixture", "peaks": []},  # mixture is NOT an alias for multimodal
        {"type": "bimodal", "peaks": []},  # bimodal is NOT an alias for multimodal
        {"type": "clamped", "min": 0, "max": 100},  # clamped does not exist
        {"type": "gaussian", "mean": 512, "stddev": 100},  # gaussian not an alias
    ],
    ids=[
        "normal-with-median",
        "lognormal-with-stddev",
        "fixed-with-mean",
        "fixed-with-stddev",
        "normal-with-peaks",
        "empirical-with-value",
        "unknown-mixture",
        "unknown-bimodal",
        "unknown-clamped",
        "unknown-gaussian",
    ],
)  # fmt: skip
def test_explicit_type_mismatch_or_unknown_fails(payload):
    # Discriminator-level ValueError (e.g. unknown `type:`) propagates raw;
    # subclass-level errors (e.g. extra_forbidden) wrap as ValidationError.
    # Existing tests in test_distributions.py follow the same convention.
    with pytest.raises((ValidationError, ValueError)):
        _DIST_ADAPTER.validate_python(payload)


def test_type_key_not_persisted_after_validation():
    """The type: key is stripped before subclass validation; result has no `type` attr."""
    result = _DIST_ADAPTER.validate_python(
        {"type": "normal", "mean": 512, "stddev": 100}
    )
    assert isinstance(result, NormalDistribution)
    assert not hasattr(result, "type")
    # And dump shouldn't contain it either
    assert "type" not in result.model_dump()
