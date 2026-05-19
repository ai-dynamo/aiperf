# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.config.sweep.expand import _expand_scenario_sweep


def test_scenario_values_key_becomes_variation_values() -> None:
    """When a scenario run carries 'values', SweepVariation.values mirrors it."""
    base = {"benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]}}
    runs = [
        {
            "name": "shape_128_128_c1",
            "values": {"isl": 128, "osl": 128, "concurrency": 1},
            "benchmark": {
                "datasets": [
                    {"name": "profiling", "prompts": {"isl": 128, "osl": 128}}
                ],
                "phases": [{"name": "profiling", "concurrency": 1}],
            },
        },
    ]
    expanded = _expand_scenario_sweep(base, runs)
    assert len(expanded) == 1
    _variant, variation = expanded[0]
    assert variation.values == {"isl": 128, "osl": 128, "concurrency": 1}
    assert variation.label == "shape_128_128_c1"


def test_scenario_without_values_falls_back_to_legacy() -> None:
    """Scenarios without 'values' preserve current behavior (values=scenario_data)."""
    base = {"benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]}}
    runs = [
        {
            "name": "v0",
            "benchmark": {"phases": [{"name": "profiling", "concurrency": 4}]},
        },
    ]
    expanded = _expand_scenario_sweep(base, runs)
    _, variation = expanded[0]
    # Legacy: nested dict survives in values (hashing-unsafe but unchanged behavior)
    assert "benchmark" in variation.values


def test_scenario_values_must_be_flat_hashable() -> None:
    """Reject values where any value is unhashable (e.g. dict, list)."""
    import pytest

    base = {"benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]}}
    runs = [
        {
            "name": "bad",
            "values": {"shape": {"isl": 128}},  # nested dict -- rejected
            "benchmark": {"phases": [{"name": "profiling", "concurrency": 1}]},
        },
    ]
    with pytest.raises(ValueError, match="values"):
        _expand_scenario_sweep(base, runs)
