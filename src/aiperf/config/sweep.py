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
    """Detect numeric list fields that qualify as magic list sweeps.

    Lists of name-bearing dicts (e.g. ``phases: [{name: profiling, ...}]``)
    are traversed using the ``name`` value as the path segment, so a
    nested magic list at ``phases[i].rate`` surfaces as
    ``phases.<name>.rate``. List entries without a string ``name`` are
    skipped — magic-list detection is a name-targeted feature.
    """
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
        elif isinstance(obj, list) and _is_named_dict_list(obj):
            for item in obj:
                name = item.get("name")
                if isinstance(name, str):
                    traverse(item, f"{current_path}.{name}" if current_path else name)

    traverse(data)
    return sweep_fields


def _is_named_dict_list(obj: list[Any]) -> bool:
    """True if every entry of ``obj`` is a dict carrying a string ``name``."""
    return bool(obj) and all(
        isinstance(item, dict) and isinstance(item.get("name"), str) for item in obj
    )


# ---------------------------------------------------------------------------
# Private expansion helpers
# ---------------------------------------------------------------------------


def _expand_grid_sweep(
    base_data: dict[str, Any], variables: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    # Sort field names alphabetically so variation order is stable across
    # writes / reads of the CR. The K8s apiserver alphabetizes object-typed
    # map keys at storage (CRD `additionalProperties` schemas), so a Python
    # dict's insertion order on submit does not survive a re-read. Without
    # this sort, child names shift between submit and resume — defeating
    # idempotent reconcile. See `gotcha_k8s_crd_object_map_keys_alphabetized`.
    field_names = sorted(variables.keys())
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
    """Set a nested value using dot-notation path.

    Path segments traverse dicts by key; for list-of-named-dicts (e.g.
    ``phases: [{name: profiling, ...}]``) the segment is matched against
    each entry's ``name`` field, so ``phases.profiling.rate`` resolves to
    the list entry whose name is ``profiling``.

    If a named-list segment does not match any existing entry, raises
    ``ValueError`` rather than silently appending a phantom entry. Typos
    like ``phases.profilling.rate`` (extra 'l') would otherwise create a
    new phase missing required fields and surface as a confusing downstream
    error.
    """
    keys = path.split(".")
    current: Any = data
    for key in keys[:-1]:
        if isinstance(current, list) and _is_named_dict_list(current):
            match = _find_named(current, key)
            if match is None:
                names = [item.get("name") for item in current]
                raise ValueError(
                    f"sweep path {path!r}: no entry named {key!r} found "
                    f"(existing: {names}). Add the entry first or fix the typo."
                )
            current = match
            continue
        if key not in current:
            current[key] = {}
        current = current[key]
    last = keys[-1]
    if isinstance(current, list) and _is_named_dict_list(current):
        match = _find_named(current, last)
        if match is None:
            names = [item.get("name") for item in current]
            raise ValueError(
                f"sweep path {path!r}: no entry named {last!r} found "
                f"(existing: {names}). Add the entry first or fix the typo."
            )
        match[last] = value
    else:
        current[last] = value


def _find_named(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Return the entry in ``items`` whose ``name`` matches, or None."""
    for item in items:
        if item.get("name") == name:
            return item
    return None


def _find_or_append_named(items: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Return the entry in ``items`` whose ``name`` matches; append if absent.

    Used for scenario-sweep deep-merge where new named entries are an
    intentional way to extend the base config. Grid/magic sweeps use
    `_find_named` (via `_set_nested_value`) so typos error loudly.
    """
    existing = _find_named(items, name)
    if existing is not None:
        return existing
    new_item: dict[str, Any] = {"name": name}
    items.append(new_item)
    return new_item


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base (modifies base in-place).

    Lists of name-bearing dicts merge by ``name`` rather than being
    replaced — entries with matching ``name`` are recursively merged,
    new-name entries are appended, and base entries not mentioned in the
    override are inherited unchanged. This is the semantics used by
    scenario-sweep ``phases:`` overrides.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif (
            key in base
            and isinstance(base[key], list)
            and isinstance(value, list)
            and _is_named_dict_list(base[key])
            and _is_named_dict_list(value)
        ):
            _merge_named_dict_lists(base[key], value)
        else:
            base[key] = value


def _merge_named_dict_lists(
    base_items: list[dict[str, Any]], override_items: list[dict[str, Any]]
) -> None:
    """Merge two lists of named dicts in-place, matching by ``name``."""
    for override_item in override_items:
        name = override_item["name"]
        existing = next((b for b in base_items if b.get("name") == name), None)
        if existing is None:
            base_items.append(copy.deepcopy(override_item))
        else:
            _deep_merge(existing, override_item)
