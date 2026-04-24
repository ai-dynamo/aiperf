# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep configuration models for parameter exploration.

Supports two sweep strategies:
- Grid: Cartesian product of all variable values
- Scenarios: Hand-picked configurations deep-merged with base
"""

from __future__ import annotations

import copy
import itertools
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Discriminator, Field

from aiperf.config._base import BaseConfig

__all__ = [
    "GridSweep",
    "ScenarioSweep",
    "SweepConfig",
    "SweepVariation",
    "expand_sweep",
]


class GridSweep(BaseConfig):
    """Grid sweep - all combinations of parameters (Cartesian product)."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    type: Literal["grid"] = Field(
        default="grid", description="Sweep type discriminator."
    )
    variables: dict[str, list[Any]] = Field(
        ...,
        description="Variables to sweep: dot-notation path -> list of values.",
        min_length=1,
    )


class ScenarioSweep(BaseConfig):
    """Scenario sweep - hand-picked configurations deep-merged with base."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    type: Literal["scenarios"] = Field(
        default="scenarios", description="Sweep type discriminator."
    )
    runs: list[dict[str, Any]] = Field(
        ...,
        description="List of scenario dicts to deep-merge with base config.",
        min_length=1,
    )


SweepConfig = Annotated[GridSweep | ScenarioSweep, Discriminator("type")]


class SweepVariation(BaseConfig):
    """Metadata for a single sweep variation."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="Zero-based variation index.")
    label: str = Field(description="Human-readable label for this variation.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter values that differ from base config.",
    )


# ---------------------------------------------------------------------------
# Expansion functions
# ---------------------------------------------------------------------------

MAGIC_LIST_FIELDS = frozenset(
    {"level", "concurrency", "rate", "count", "requests", "time", "mean"}
)


def expand_sweep(data: dict[str, Any]) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Expand sweep configuration into (variation_dict, metadata) pairs.

    Returns:
        List of (config_dict, SweepVariation) tuples.
        If no sweep detected, returns a single-element list with the base config.
    """
    variations: list[tuple[dict[str, Any], SweepVariation]] = []

    if "sweep" in data and data["sweep"] is not None:
        sweep_config = data["sweep"]
        if isinstance(sweep_config, dict):
            sweep_type = sweep_config.get("type", "grid")

            if sweep_type == "grid":
                variables = sweep_config.get("variables", {})
                variations = _expand_grid_sweep(data, variables)
            elif sweep_type == "scenarios":
                runs = sweep_config.get("runs", [])
                variations = _expand_scenario_sweep(data, runs)

    if not variations:
        magic_sweeps = detect_sweep_fields(data)
        if magic_sweeps:
            variations = _expand_magic_lists(data, magic_sweeps)

    if not variations:
        base = {k: v for k, v in data.items() if k != "sweep"}
        return [(base, SweepVariation(index=0, label="base", values={}))]

    return variations


def detect_sweep_fields(data: dict[str, Any]) -> dict[str, list[Any]]:
    """Detect numeric list fields that qualify as magic list sweeps."""
    sweep_fields: dict[str, list[Any]] = {}

    def traverse(obj: Any, current_path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if (
                    isinstance(value, list)
                    and key in MAGIC_LIST_FIELDS
                    and all(isinstance(v, (int, float)) for v in value)
                ):
                    sweep_fields[new_path] = value
                else:
                    traverse(value, new_path)

    traverse(data)
    return sweep_fields


# ---------------------------------------------------------------------------
# Private expansion helpers
# ---------------------------------------------------------------------------


def _expand_grid_sweep(
    base_data: dict[str, Any], variables: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    field_names = list(variables.keys())
    value_lists = [variables[f] for f in field_names]
    combinations = list(itertools.product(*value_lists))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(base_data)
        values = {}
        for field_path, value in zip(field_names, combo, strict=False):
            _set_nested_value(variant, field_path, value)
            values[field_path] = value
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = ", ".join(f"{k}={v}" for k, v in values.items())
        results.append((variant, SweepVariation(index=idx, label=label, values=values)))
    return results


def _expand_scenario_sweep(
    base_data: dict[str, Any], runs: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    results = []
    for idx, scenario in enumerate(runs):
        variant = copy.deepcopy(base_data)
        scenario_data = {k: v for k, v in scenario.items() if k != "name"}
        _deep_merge(variant, scenario_data)
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = scenario.get("name", f"scenario_{idx}")
        results.append(
            (variant, SweepVariation(index=idx, label=label, values=scenario_data))
        )
    return results


def _expand_magic_lists(
    data: dict[str, Any], sweep_fields: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    field_names = list(sweep_fields.keys())
    value_lists = [sweep_fields[f] for f in field_names]
    combinations = list(itertools.product(*value_lists))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(data)
        values = {}
        for field_path, value in zip(field_names, combo, strict=False):
            _set_nested_value(variant, field_path, value)
            values[field_path] = value
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = ", ".join(f"{k}={v}" for k, v in values.items())
        results.append((variant, SweepVariation(index=idx, label=label, values=values)))
    return results


def _set_nested_value(data: dict, path: str, value: Any) -> None:
    """Set a nested value using dot-notation path."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base (modifies base in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
