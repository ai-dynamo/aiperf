# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-config expansion helpers.

Public re-export: ``aiperf.config.sweep.expand_sweep``.
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.sweep.config import SweepVariation

MAGIC_LIST_FIELDS = frozenset(
    {
        # Phase-field names (used by detect_sweep_fields and the converter
        # promote pass that walks phases[*]).
        "level",
        "concurrency",
        "prefill_concurrency",
        "rate",
        "count",
        "requests",
        "duration",
        "sessions",
        "users",
        "time",
        "mean",
        # CLI-attribute names. Used only by the CLI-side rejection logic
        # in `_reject_recipe_plus_magic_lists` and
        # `_reject_variants_plus_magic_lists`, which iterate
        # `cli.model_fields_set` and check `name in MAGIC_LIST_FIELDS`.
        # Phase-walking scans never see these as keys, so listing them
        # here is a no-op on those code paths.
        "request_rate",
        "request_count",
        "benchmark_duration",
        "conversation_num",
        "num_users",
        "prompt_input_tokens_mean",
        "prompt_input_tokens_stddev",
        "prompt_output_tokens_mean",
        "prompt_output_tokens_stddev",
        "conversation_turn_mean",
    }
)

# Field-presence keys that discriminate distribution kinds in
# ``aiperf.config.distributions`` (Fixed/Normal/LogNormal/Multimodal/Empirical).
# Swapping any of these between override and base implies a distribution-kind
# change, which deep-merge can't express -- merging would produce a frankendict
# with conflicting discriminator keys (e.g. {mean, stddev, median}).
_DIST_DISCRIMINATOR_KEYS = frozenset({"peaks", "points", "value", "median", "stddev"})


def expand_sweep(data: dict[str, Any]) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Expand sweep configuration into (variation_dict, metadata) pairs.

    Returns:
        List of (config_dict, SweepVariation) tuples.
        If no sweep detected, returns a single-element list with the base config.
    """
    from aiperf.config.sweep import SweepVariation

    variations = _expand_explicit_sweep(data)

    if not variations:
        magic_sweeps = detect_sweep_fields(data.get("benchmark") or {})
        if magic_sweeps:
            variations = _expand_magic_lists(data, magic_sweeps)

    if not variations:
        base = {k: v for k, v in data.items() if k != "sweep"}
        return [(base, SweepVariation(index=0, label="base", values={}))]

    return variations


def _expand_explicit_sweep(
    data: dict[str, Any],
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Dispatch the explicit ``sweep:`` block by ``type``.

    Returns an empty list when there is no explicit sweep block (caller falls
    back to magic-list detection / single base variation).
    """
    sweep_config = data.get("sweep")
    if not isinstance(sweep_config, dict):
        return []

    sweep_type = sweep_config.get("type", "grid")
    if sweep_type == "grid":
        return _expand_grid_sweep(data, sweep_config.get("parameters", {}))
    if sweep_type == "zip":
        return _expand_zip_sweep(data, sweep_config.get("parameters", {}))
    if sweep_type == "scenarios":
        return _expand_scenario_sweep(data, sweep_config.get("runs", []))
    if sweep_type in ("sobol", "latin_hypercube"):
        return _expand_qmc_sweep_from_block(data, sweep_config, sweep_type)
    return []


def _expand_qmc_sweep_from_block(
    data: dict[str, Any],
    sweep_config: dict[str, Any],
    sweep_type: str,
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Hydrate the QMC sweep block into ``expand_qmc_sweep`` keyword args."""
    from aiperf.config.sweep import SamplingDimension
    from aiperf.config.sweep.expand_qmc import expand_qmc_sweep

    dimensions = [SamplingDimension(**d) for d in sweep_config.get("dimensions", [])]
    return expand_qmc_sweep(
        data,
        sweep_type=sweep_type,
        samples=sweep_config["samples"],
        seed=sweep_config.get("seed"),
        dimensions=dimensions,
        options={
            "scramble": sweep_config.get("scramble", True),
            "optimization": sweep_config.get("optimization"),
        },
        label_format=sweep_config.get("label_format", "index"),
    )


def detect_sweep_fields(data: dict[str, Any]) -> dict[str, list[Any]]:
    """Detect numeric list fields under ``phases`` that qualify as magic
    list sweeps.

    Magic-list detection is intentionally **scoped to phase-rooted paths**.
    The convention is "phase-only sweep shorthand": a list at
    ``phases[i].rate`` is the YAML equivalent of ``--rate 10,20,30``. A
    magic-named key at any other path (e.g. ``datasets.X.prompts.isl.mean
    = [100, 200]``) is almost always a user error and silently
    auto-sweeping produces variations that fail downstream validation.
    Use the explicit ``sweep:`` block for non-phase paths.
    """
    sweep_fields: dict[str, list[Any]] = {}

    def _collect(phase: dict[str, Any], prefix: str) -> None:
        for key, value in phase.items():
            if (
                isinstance(value, list)
                and key in MAGIC_LIST_FIELDS
                and all(isinstance(v, (int, float)) for v in value)
            ):
                sweep_fields[f"{prefix}.{key}"] = value

    phases = data.get("phases")
    if isinstance(phases, dict):
        _collect(phases, "phases")
    elif isinstance(phases, list):
        for item in phases:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                _collect(item, f"phases.{item['name']}")
    return sweep_fields


def _is_named_dict_list(obj: list[Any]) -> bool:
    """True if every entry of ``obj`` is a dict carrying a string ``name``."""
    return bool(obj) and all(
        isinstance(item, dict) and isinstance(item.get("name"), str) for item in obj
    )


# ---------------------------------------------------------------------------
# Private expansion helpers
# ---------------------------------------------------------------------------


def _classify_sweep_parameters(
    parameters: dict[str, list[Any]], *, mode: str
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Validate sweep parameter keys/values and split into body- vs envelope-rooted.

    Shared by ``_expand_grid_sweep`` and ``_expand_zip_sweep``: both apply
    the same per-key alias resolution, structural checks (string key,
    non-empty list value, dotted-path validity, no canonical+alias
    collision), and partition by first segment (``variables.*`` is
    envelope-rooted; everything else is body-rooted under ``benchmark.``).

    ``mode`` is just the leading word in error messages (``"grid"`` or
    ``"zip"``) — the validation rules are identical.
    """
    from aiperf.config.loader.dotted_path import _validate_dotted_path

    body_paths: dict[str, list[Any]] = {}
    envelope_paths: dict[str, list[Any]] = {}
    for path, values in parameters.items():
        if not isinstance(path, str):
            raise ValueError(
                f"{mode} sweep parameter key must be a string, "
                f"got {type(path).__name__}: {path!r}"
            )
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(
                f"{mode} sweep parameter {path!r}: value list must be non-empty."
            )
        try:
            resolved = _validate_dotted_path(path)
        except ValueError as e:
            raise ValueError(f"{mode} sweep parameter: {e}") from e
        if resolved in body_paths or resolved in envelope_paths:
            raise ValueError(
                f"{mode} sweep parameter {path!r} resolves to {resolved!r}, "
                f"which is already a parameter. Pick one spelling (the bare "
                f"alias or the full dotted path) and remove the other."
            )
        if resolved.split(".", 1)[0] == "variables":
            envelope_paths[resolved] = values
        else:
            body_paths[resolved] = values
    return body_paths, envelope_paths


def _expand_grid_sweep(
    base_data: dict[str, Any], parameters: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Cartesian-product expansion. Path keys are body-rooted under ``benchmark``.

    Bare paths target fields inside the ``benchmark:`` block, e.g.
    ``phases.profiling.concurrency`` resolves to
    ``benchmark.phases.profiling.concurrency``. The redundant
    ``benchmark.`` prefix is rejected.

    The single envelope-level escape is ``variables.<name>``, which writes
    into the root ``variables:`` Jinja block per variation before Jinja
    re-renders the variant. Use this when one Jinja var templates into
    multiple body fields and you want them swept together (paired sweep).
    Other top-level prefixes (``sweep.``, ``multi_run.``, ``random_seed.``)
    are rejected as non-sweepable.
    """
    from aiperf.config.sweep import SweepVariation

    body_paths, envelope_paths = _classify_sweep_parameters(parameters, mode="grid")

    # Sort field names alphabetically so variation order is stable across
    # writes / reads of the CR. The K8s apiserver alphabetizes object-typed
    # map keys at storage (CRD `additionalProperties` schemas), so a Python
    # dict's insertion order on submit does not survive a re-read. Without
    # this sort, child names shift between submit and resume — defeating
    # idempotent reconcile. See `gotcha_k8s_crd_object_map_keys_alphabetized`.
    field_names = sorted({**body_paths, **envelope_paths}.keys())
    value_lists = [
        (body_paths.get(f) if f in body_paths else envelope_paths[f])
        for f in field_names
    ]
    combinations = list(itertools.product(*value_lists))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(base_data)
        body = variant.setdefault("benchmark", {})
        values: dict[str, Any] = {}
        for field_path, value in zip(field_names, combo, strict=False):
            if field_path in envelope_paths:
                # variables.<name> -> envelope-level Jinja block (re-rendered per variation)
                _set_nested_value(variant, field_path, value)
            else:
                _set_nested_value(body, field_path, value)
            values[field_path] = value
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = ", ".join(f"{k}={v}" for k, v in values.items())
        results.append((variant, SweepVariation(index=idx, label=label, values=values)))
    return results


def _expand_zip_sweep(
    base_data: dict[str, Any], parameters: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Element-wise (lockstep) expansion of paired parameter lists.

    Same path semantics as ``_expand_grid_sweep`` (bare paths target
    ``benchmark.``, ``variables.<name>`` writes the envelope-level Jinja
    block) but uses ``zip(strict=True)`` instead of ``itertools.product``.
    All parameter lists must have identical length; mismatched lengths
    raise ``ValueError`` here as well as at Pydantic validation time
    (defense-in-depth — direct callers that bypass the model still error).

    Field names are sorted alphabetically before zipping so variation
    order is stable across writes / reads of the CR. The K8s apiserver
    alphabetizes object-typed map keys at storage (CRD
    `additionalProperties` schemas), so a Python dict's insertion order
    on submit does not survive a re-read. Without this sort, child names
    shift between submit and resume — defeating idempotent reconcile.
    See `gotcha_k8s_crd_object_map_keys_alphabetized`.
    """
    from aiperf.config.sweep import SweepVariation

    body_paths, envelope_paths = _classify_sweep_parameters(parameters, mode="zip")

    lengths = {k: len(v) for k, v in parameters.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"zip sweep parameters must all have equal length; got {lengths!r}."
        )

    field_names = sorted({**body_paths, **envelope_paths}.keys())
    value_lists = [
        (body_paths.get(f) if f in body_paths else envelope_paths[f])
        for f in field_names
    ]
    combinations = list(zip(*value_lists, strict=True))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(base_data)
        body = variant.setdefault("benchmark", {})
        values: dict[str, Any] = {}
        for field_path, value in zip(field_names, combo, strict=False):
            if field_path in envelope_paths:
                _set_nested_value(variant, field_path, value)
            else:
                _set_nested_value(body, field_path, value)
            values[field_path] = value
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = ", ".join(f"{k}={v}" for k, v in values.items())
        results.append((variant, SweepVariation(index=idx, label=label, values=values)))
    return results


def _normalize_scenario_dataset_form(
    scenario: dict[str, Any], base: dict[str, Any], idx: int
) -> None:
    """Rewrite scenario `benchmark.dataset:` (singular) into
    `benchmark.datasets: [...]` so it deep-merges cleanly against the
    always-plural base.
    """
    from aiperf.config.loader.normalizers import DATASET_VS_DATASETS_MSG

    bench = scenario.get("benchmark")
    if not isinstance(bench, dict):
        return
    if "dataset" not in bench:
        return
    if "datasets" in bench:
        raise ValueError(f"sweep run [{idx}]: " + DATASET_VS_DATASETS_MSG)

    original = bench["dataset"]
    if not isinstance(original, dict):
        raise ValueError(
            f"sweep run [{idx}]: 'benchmark.dataset:' must be a mapping; "
            f"got {type(original).__name__}."
        )

    base_bench = base.get("benchmark", {})
    base_datasets = base_bench.get("datasets") or []
    # Base may also use the singular `dataset:` shorthand — treat it as a
    # one-element list with the auto-named "default" entry that the
    # benchmark normalizer will eventually produce.
    if not base_datasets and isinstance(base_bench.get("dataset"), dict):
        base_datasets = [{"name": "default", **base_bench["dataset"]}]
    explicit_name = original.get("name") if isinstance(original, dict) else None
    if explicit_name is not None:
        resolved_name = explicit_name
    elif len(base_datasets) == 1 and isinstance(base_datasets[0], dict):
        resolved_name = base_datasets[0].get("name")
        if resolved_name is None:
            raise ValueError(
                f"sweep run [{idx}]: base dataset has no 'name' to inherit; "
                f"add 'name:' to the scenario's dataset."
            )
    else:
        names = [d.get("name") for d in base_datasets if isinstance(d, dict)]
        raise ValueError(
            f"sweep run [{idx}]: scenario uses singular 'benchmark.dataset:' "
            f"against a base with multiple datasets ({names!r}); add 'name:' "
            f"to disambiguate."
        )

    bench.pop("dataset")
    bench["datasets"] = [
        {"name": resolved_name, **{k: v for k, v in original.items() if k != "name"}}
    ]
    # If the BASE used singular `dataset:` shorthand, promote it to plural
    # `datasets:` form too — otherwise the post-merge variant has BOTH keys
    # and `BenchmarkConfig`'s mutual-exclusivity check rejects it.
    base_bench_mut = base.get("benchmark")
    if (
        isinstance(base_bench_mut, dict)
        and "dataset" in base_bench_mut
        and "datasets" not in base_bench_mut
    ):
        base_singular = base_bench_mut.pop("dataset")
        if isinstance(base_singular, dict):
            base_bench_mut["datasets"] = [
                {
                    "name": resolved_name,
                    **{k: v for k, v in base_singular.items() if k != "name"},
                }
            ]


_ALLOWED_SCENARIO_RUN_KEYS = {"name", "variables", "benchmark", "values"}


def _expand_scenario_sweep(
    base_data: dict[str, Any], runs: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    """Expand scenario sweep. Each run is a partial envelope.

    Allowed run keys: ``name``, ``variables``, ``benchmark``, ``values``.
    ``values`` is an optional flat ``dict[str, Hashable]`` that overrides
    ``SweepVariation.values`` when set; without it, values defaults to the
    full deep-merge subtree (the default). Recipes that need clean
    hashable groupings (e.g. pareto-sweep crossing paired ISL/OSL with a
    concurrency list) set ``values`` explicitly.
    """
    from aiperf.config.sweep import SweepVariation

    results = []
    for idx, scenario in enumerate(runs):
        unknown = set(scenario.keys()) - _ALLOWED_SCENARIO_RUN_KEYS
        if unknown:
            raise ValueError(
                f"sweep run [{idx}]: unknown field(s) {sorted(unknown)!r}; "
                f"allowed: {sorted(_ALLOWED_SCENARIO_RUN_KEYS)}. (If you migrated "
                f"from the flat shape, wrap body fields under "
                f"`benchmark:` inside the run.)"
            )
        explicit_values = scenario.get("values")
        if explicit_values is not None:
            if not isinstance(explicit_values, dict):
                raise ValueError(
                    f"sweep run [{idx}].values must be a dict, got "
                    f"{type(explicit_values).__name__}"
                )
            for k, v in explicit_values.items():
                try:
                    hash(v)
                except TypeError as e:
                    raise ValueError(
                        f"sweep run [{idx}].values[{k!r}] must be a "
                        f"hashable scalar; got {type(v).__name__}"
                    ) from e
        variant = copy.deepcopy(base_data)
        scenario_data = {
            k: v for k, v in scenario.items() if k not in {"name", "values"}
        }
        _normalize_scenario_dataset_form(scenario_data, variant, idx)
        _deep_merge(variant, scenario_data)
        variant = {k: v for k, v in variant.items() if k != "sweep"}
        label = scenario.get("name", f"scenario_{idx}")
        values = dict(explicit_values) if explicit_values is not None else scenario_data
        results.append((variant, SweepVariation(index=idx, label=label, values=values)))
    return results


def _expand_magic_lists(
    data: dict[str, Any], sweep_fields: dict[str, list[Any]]
) -> list[tuple[dict[str, Any], SweepVariation]]:
    from aiperf.config.sweep import SweepVariation

    field_names = list(sweep_fields.keys())
    value_lists = [sweep_fields[f] for f in field_names]
    combinations = list(itertools.product(*value_lists))

    results = []
    for idx, combo in enumerate(combinations):
        variant = copy.deepcopy(data)
        body = variant.setdefault("benchmark", {})
        values = {}
        for field_path, value in zip(field_names, combo, strict=False):
            _set_nested_value(body, field_path, value)
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
    the list entry whose name is ``profiling``. Missing-name segments
    raise ValueError rather than silently appending a phantom entry.

    Special case for ``phases.profiling.<X>``: when no phase named
    ``profiling`` exists, fall back to the unique non-warmup phase (if
    exactly one exists). YAML's ``phases: {type: ..}`` shorthand emits
    a phase named ``default``, but search recipes hard-code the
    ``profiling`` segment. See ``_find_phase_or_recipe_alias``.
    """
    keys = path.split(".")
    current: Any = data
    for i, key in enumerate(keys[:-1]):
        if isinstance(current, list) and _is_named_dict_list(current):
            match = _find_phase_or_recipe_alias(
                current, key, parent_key=keys[i - 1] if i > 0 else ""
            )
            if match is None:
                names = [item.get("name") for item in current]
                raise ValueError(
                    f"sweep path {path!r}: no entry named {key!r} found "
                    f"(existing: {names}). Add the entry first or fix the typo."
                )
            current = match
            continue
        if not isinstance(current, dict):
            _raise_traversal_type_error(path, keys, i, current, descend=True)
        if key not in current:
            current[key] = {}
        current = current[key]
    last = keys[-1]
    if isinstance(current, list) and _is_named_dict_list(current):
        match = _find_phase_or_recipe_alias(
            current, last, parent_key=keys[-2] if len(keys) >= 2 else ""
        )
        if match is None:
            names = [item.get("name") for item in current]
            raise ValueError(
                f"sweep path {path!r}: no entry named {last!r} found "
                f"(existing: {names}). Add the entry first or fix the typo."
            )
        match[last] = value
    elif isinstance(current, dict):
        current[last] = value
    else:
        _raise_traversal_type_error(path, keys, len(keys) - 1, current, descend=False)


def _raise_traversal_type_error(
    path: str, keys: list[str], i: int, current: Any, *, descend: bool
) -> None:
    """Raise a clear ValueError when ``_set_nested_value`` hits a non-dict
    mid-traversal. ``descend`` chooses the wording: True = mid-loop, False
    = final assignment.
    """
    prev = ".".join(keys[:i]) or "<root>"
    verb = "descend into" if descend else "assign into"
    raise ValueError(
        f"sweep path {path!r}: cannot {verb} {type(current).__name__} "
        f"at segment {prev!r}; expected a dict or list-of-named-dicts."
    )


def _find_phase_or_recipe_alias(
    items: list[dict[str, Any]], name: str, *, parent_key: str
) -> dict[str, Any] | None:
    """Resolve `_find_named` with a `phases.profiling` recipe-friendly fallback.

    When ``name`` is ``profiling`` and the parent segment is ``phases`` and
    no phase by that name exists, return the unique non-warmup phase if one
    exists; otherwise return None (caller raises with the existing-names
    list, same as a plain miss).
    """
    direct = _find_named(items, name)
    if direct is not None:
        return direct
    if name != "profiling" or parent_key != "phases":
        return None
    candidates = [item for item in items if item.get("name") != "warmup"]
    if len(candidates) == 1:
        return candidates[0]
    return None


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

    Distribution-shaped dicts (any side containing a discriminator key in
    ``_DIST_DISCRIMINATOR_KEYS``) are REPLACED rather than merged when the
    override changes the discriminator set. This lets scenarios cleanly
    swap Normal <-> LogNormal <-> Multimodal <-> Empirical without producing
    frankendicts that fail Pydantic discrimination on the merged side.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            if _is_distribution_swap(base[key], value):
                base[key] = value
            else:
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


def _is_distribution_swap(base: dict, override: dict) -> bool:
    """True when override changes the implied distribution kind vs base.

    Both sides must carry at least one discriminator key, and their
    discriminator sets must differ. Common-modifier-only overrides (e.g.
    just adding ``min:``/``max:``) merge as usual.
    """
    base_disc = set(base) & _DIST_DISCRIMINATOR_KEYS
    over_disc = set(override) & _DIST_DISCRIMINATOR_KEYS
    if not base_disc or not over_disc:
        return False
    return base_disc != over_disc


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
