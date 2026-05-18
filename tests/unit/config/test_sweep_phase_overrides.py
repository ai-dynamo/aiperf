# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name-targeted phase override semantics for list-shaped phases.

After the dict→list `phases` migration (plan 2026-04-26), sweep dot-paths
like ``phases.profiling.rate`` and scenario phase overrides like
``phases: [{name: profiling, rate: 50}]`` must resolve by name into the
list, not by index, so user mental model and YAML stability are preserved.

This file locks the contract for the sweep merger; the implementation
lives in ``aiperf.config.sweep``.
"""

from __future__ import annotations

from aiperf.config.sweep import (
    _set_nested_value,
    expand_sweep,
)
from aiperf.config.sweep.expand import (
    _deep_merge,
    detect_sweep_fields,
)

# ----------------------------------------------------------------------
# _set_nested_value: phases.<name>.<field> targets list entry by name
# ----------------------------------------------------------------------


class TestSetNestedValueByPhaseName:
    """Grid-sweep dot-path resolver matches list entries by ``name``."""

    def test_sets_field_on_named_phase_in_list(self) -> None:
        data = {
            "phases": [
                {"name": "warmup", "concurrency": 8},
                {"name": "profiling", "rate": 10.0},
            ]
        }
        _set_nested_value(data, "phases.profiling.rate", 50.0)

        assert data["phases"][0] == {"name": "warmup", "concurrency": 8}
        assert data["phases"][1] == {"name": "profiling", "rate": 50.0}

    def test_sets_field_on_first_named_phase(self) -> None:
        data = {
            "phases": [
                {"name": "warmup", "concurrency": 8},
                {"name": "profiling", "rate": 10.0},
            ]
        }
        _set_nested_value(data, "phases.warmup.concurrency", 32)

        assert data["phases"][0]["concurrency"] == 32
        assert data["phases"][1]["rate"] == 10.0  # untouched

    def test_creates_field_on_named_phase_when_missing(self) -> None:
        data = {"phases": [{"name": "profiling", "type": "concurrency"}]}
        _set_nested_value(data, "phases.profiling.requests", 100)

        assert data["phases"][0]["requests"] == 100


# ----------------------------------------------------------------------
# detect_sweep_fields: list-of-named-dicts produces phases.<name>.<field>
# ----------------------------------------------------------------------


class TestDetectSweepFieldsOnPhasesList:
    """Magic list detection traverses list-shaped phases by name."""

    def test_finds_magic_list_inside_named_phase(self) -> None:
        data = {
            "phases": [
                {"name": "profiling", "concurrency": [8, 16, 32]},
            ]
        }
        fields = detect_sweep_fields(data)

        assert "phases.profiling.concurrency" in fields
        assert fields["phases.profiling.concurrency"] == [8, 16, 32]

    def test_finds_magic_list_in_each_named_phase(self) -> None:
        data = {
            "phases": [
                {"name": "warmup", "concurrency": 8},
                {"name": "profiling", "rate": [10.0, 30.0, 50.0]},
            ]
        }
        fields = detect_sweep_fields(data)

        assert "phases.profiling.rate" in fields
        assert fields["phases.profiling.rate"] == [10.0, 30.0, 50.0]


# ----------------------------------------------------------------------
# Grid sweep end-to-end: dot-path with phase name resolves into list
# ----------------------------------------------------------------------


class TestGridSweepWithPhaseNamePath:
    """``phases.<name>.<field>`` in grid variables targets the list entry."""

    def _data(self) -> dict:
        return {
            "benchmark": {
                "phases": [
                    {
                        "name": "warmup",
                        "type": "concurrency",
                        "requests": 50,
                        "concurrency": 8,
                    },
                    {
                        "name": "profiling",
                        "type": "poisson",
                        "rate": 20.0,
                        "concurrency": 64,
                    },
                ],
            },
            "sweep": {
                "type": "grid",
                "parameters": {
                    "phases.profiling.rate": [10.0, 30.0, 50.0],
                },
            },
        }

    def test_grid_sweep_targets_named_phase_by_name(self) -> None:
        result = expand_sweep(self._data())

        assert len(result) == 3
        rates = [
            next(p for p in cfg["benchmark"]["phases"] if p["name"] == "profiling")[
                "rate"
            ]
            for cfg, _ in result
        ]
        assert rates == [10.0, 30.0, 50.0]

    def test_grid_sweep_leaves_untouched_phases_alone(self) -> None:
        result = expand_sweep(self._data())

        for cfg, _ in result:
            warmup = next(
                p for p in cfg["benchmark"]["phases"] if p["name"] == "warmup"
            )
            assert warmup["concurrency"] == 8
            assert warmup["requests"] == 50


# ----------------------------------------------------------------------
# Scenario sweep: list-of-named-overrides merges by name
# ----------------------------------------------------------------------


class TestScenarioSweepPhaseOverrideByName:
    """Scenario `phases:` overrides are a list of partial named entries."""

    def _data(self) -> dict:
        return {
            "benchmark": {
                "phases": [
                    {
                        "name": "warmup",
                        "type": "concurrency",
                        "requests": 50,
                        "concurrency": 8,
                    },
                    {
                        "name": "profiling",
                        "type": "poisson",
                        "rate": 20.0,
                        "concurrency": 64,
                    },
                ],
            },
            "sweep": {
                "type": "scenarios",
                "runs": [
                    {
                        "name": "chatbot",
                        "benchmark": {
                            "phases": [
                                {
                                    "name": "profiling",
                                    "rate": 50.0,
                                    "concurrency": 128,
                                },
                            ],
                        },
                    },
                    {
                        "name": "summarization",
                        "benchmark": {
                            "phases": [
                                {
                                    "name": "profiling",
                                    "rate": 5.0,
                                    "concurrency": 16,
                                },
                            ],
                        },
                    },
                ],
            },
        }

    def test_scenario_phase_override_merges_by_name(self) -> None:
        result = expand_sweep(self._data())

        assert len(result) == 2

        chatbot_phases = result[0][0]["benchmark"]["phases"]
        assert isinstance(chatbot_phases, list)
        prof = next(p for p in chatbot_phases if p["name"] == "profiling")
        assert prof["rate"] == 50.0
        assert prof["concurrency"] == 128
        assert prof["type"] == "poisson"  # inherited from base

    def test_scenario_phase_override_leaves_other_phases_unchanged(self) -> None:
        result = expand_sweep(self._data())

        for cfg, _ in result:
            warmup = next(
                p for p in cfg["benchmark"]["phases"] if p["name"] == "warmup"
            )
            assert warmup == {
                "name": "warmup",
                "type": "concurrency",
                "requests": 50,
                "concurrency": 8,
            }

    def test_scenario_appends_phase_not_in_base(self) -> None:
        """Override may introduce a new phase that wasn't in the base list."""
        data = {
            "benchmark": {
                "phases": [
                    {"name": "profiling", "type": "concurrency", "concurrency": 8}
                ],
            },
            "sweep": {
                "type": "scenarios",
                "runs": [
                    {
                        "name": "with_warmup",
                        "benchmark": {
                            "phases": [
                                {
                                    "name": "warmup",
                                    "type": "concurrency",
                                    "requests": 10,
                                    "concurrency": 1,
                                }
                            ],
                        },
                    }
                ],
            },
        }
        result = expand_sweep(data)

        assert len(result) == 1
        names = [p["name"] for p in result[0][0]["benchmark"]["phases"]]
        # Existing phases stay, override-only entries are appended at the end.
        assert "profiling" in names
        assert "warmup" in names


# ----------------------------------------------------------------------
# _deep_merge directly: list-of-named-dicts merges by name
# ----------------------------------------------------------------------


class TestDeepMergePhasesByName:
    """``_deep_merge`` matches list entries by ``name`` for phases."""

    def test_merge_overrides_only_named_entry(self) -> None:
        base = {
            "phases": [
                {"name": "warmup", "concurrency": 8},
                {"name": "profiling", "rate": 20.0, "concurrency": 64},
            ]
        }
        override = {"phases": [{"name": "profiling", "rate": 50.0}]}
        _deep_merge(base, override)

        assert base["phases"][0] == {"name": "warmup", "concurrency": 8}
        assert base["phases"][1] == {
            "name": "profiling",
            "rate": 50.0,
            "concurrency": 64,
        }

    def test_merge_appends_missing_named_entry(self) -> None:
        base = {"phases": [{"name": "profiling", "rate": 20.0}]}
        override = {"phases": [{"name": "warmup", "concurrency": 8}]}
        _deep_merge(base, override)

        names = [p["name"] for p in base["phases"]]
        assert names == ["profiling", "warmup"]
