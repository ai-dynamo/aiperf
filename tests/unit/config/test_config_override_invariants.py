# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mechanical invariants over every dataset flag, not a hand-picked sample.

Two properties are checked for all of ``INPUT_FIELDS`` at once, so a field
added later is covered without anyone remembering to write a test for it:

**No spontaneous keys.** In override mode ``build_dataset`` must emit only
values the user actually supplied. The detector runs the builder twice with
two *different* values for the same field: any emitted key whose value is
identical across both runs cannot be derived from the input, so it is a
materialized default. Merged onto a YAML dataset, such a key silently
overwrites a value the user never mentioned -- how ``images.batch_size=1``
and ``turns.stddev=0`` would have leaked before this was guarded.

**No dead routing.** Every field the registry claims is routed under
``--config`` must actually change the resolved config. ``ROUTED_UNDER_CONFIG``
computes the input section as a complement, so a newly-added field is
classified as routed automatically; this test is what stops that convenience
from quietly re-opening the bug for a field ``build_dataset`` does not carry.
"""

from __future__ import annotations

import enum
import types
import typing
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import DatasetType
from aiperf.config.flags import CLIConfig
from aiperf.config.flags._config_flag_routing import (
    DATASET_OVERRIDE_FIELDS,
    MAGIC_LIST_ONLY_UNDER_CONFIG,
    ROUTED_UNDER_CONFIG,
)
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError

# Every field the registry claims to route, which is what the guarantee is
# about -- not just the dataset ones.
_ROUTED_FIELDS_UNDER_TEST = ROUTED_UNDER_CONFIG


def cli(**kwargs: object) -> CLIConfig:
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


def _with_companion(field: str, value: Any) -> dict[str, Any]:
    """Flag kwargs for ``field``, plus the companion its schema requires.

    ``images.width`` and friends validate as a normal distribution, so a
    stddev with no mean is rejected outright. Driving the pair keeps these
    fields covered instead of erroring into a skip.
    """
    kwargs: dict[str, Any] = {field: value}
    if field.endswith("_stddev"):
        mean_field = field[: -len("_stddev")] + "_mean"
        if mean_field in CLIConfig.model_fields:
            kwargs[mean_field] = 8
    return kwargs


def _unwrap(annotation: Any) -> Any:
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
            continue
        if origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if not args:
                return None
            # Magic-list flags are declared `list[int] | int | None`. Prefer
            # the scalar member: it is the form this path routes, and taking
            # args[0] would report the field as undrivable.
            scalars = [a for a in args if typing.get_origin(a) is not list]
            annotation = scalars[0] if scalars else args[0]
            continue
        return annotation


# Values for fields whose annotation cannot produce them: ``typing.Any``
# (the magic-list flags), or types whose contents must parse. Without these
# the fields would report as undrivable and go untested.
FIELD_PROBE_VALUES: dict[str, list[Any]] = {
    "request_count": [2, 3],
    "concurrency": [2, 3],
    "num_users": [2, 3],
    "prefill_concurrency": [2, 3],
    "request_rate": [2.0, 3.0],
    "benchmark_duration": [2.0, 3.0],
    "headers": [["x-probe-a:1"], ["x-probe-b:2"]],
    "extra_inputs": [["probe_a:1"], ["probe_b:2"]],
    "mlflow_tags": [["probe_a:1"], ["probe_b:2"]],
    "server_metrics": [
        ["http://localhost:9090/metrics"],
        ["http://localhost:9091/metrics"],
    ],
    "gpu_telemetry": [
        ["http://localhost:9400/metrics"],
        ["http://localhost:9401/metrics"],
    ],
    "isl_osl_pairs": [["128,16"], ["256,32"]],
    "server_metrics_formats": [["json"], ["csv"]],
    "sweep_variants": [["concurrency=2"], ["concurrency=4"]],
}


def _candidate_values(field: str) -> list[Any]:
    """Valid values to drive ``field`` with, or [] if none can be generated.

    Returning [] (rather than guessing) keeps the test honest: a field it
    cannot drive is reported as skipped instead of silently passing.
    """
    if field in FIELD_PROBE_VALUES:
        return FIELD_PROBE_VALUES[field]
    info = CLIConfig.model_fields.get(field)
    if info is None:
        return []
    annotation = _unwrap(info.annotation)
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return list(annotation)
    if annotation is bool:
        return [True, False]
    if annotation is int:
        return [2, 3]
    if annotation is float:
        return [2.0, 3.0]
    if annotation is str:
        return ["aiperf-probe-a", "aiperf-probe-b"]
    if annotation is Path:
        return [Path("/tmp/aiperf-probe-a"), Path("/tmp/aiperf-probe-b")]
    # list[str] and friends: a one-element list is enough to vary the value.
    # Fields whose contents must parse (--header k:v, --search-sla ...) reject
    # these and raise, which the invariant accepts as loud.
    if typing.get_origin(annotation) is list:
        args = typing.get_args(annotation)
        if args and args[0] is str:
            return [["aiperf-probe-a"], ["aiperf-probe-b"]]
    return []


def _value_pair(field: str) -> tuple[Any, Any] | None:
    """Two distinct valid values for ``field``, or None if unavailable."""
    values = _candidate_values(field)
    return (values[0], values[1]) if len(values) >= 2 else None


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``{dotted.path: leaf}``."""
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key, sub in value.items():
        out.update(_leaves(sub, f"{prefix}.{key}" if prefix else str(key)))
    return out


# Fields the value generator cannot drive from their annotation (lists,
# free-form strings, parsed DSL strings). Listing them explicitly rather than
# skipping on the fly means a newly-added field of an unsupported type fails
# this suite instead of quietly reporting "skipped" -- a skip is invisible in
# CI, which is how a coverage hole hides.
UNDRIVABLE_FIELDS: frozenset[str] = frozenset(
    {
        # list-shaped
        "audio_depths",
        "audio_sample_rates",
        "conversation_num",
        "conversation_turn_mean",
        "dataset_filters",
        "prompt_input_tokens_mean",
        "prompt_input_tokens_stddev",
        "prompt_output_tokens_mean",
        "prompt_output_tokens_stddev",
        "video_audio_depth",
        # free-form / parsed strings with no safe auto-generated value
        "hf_dataset_subset",
        "prompt_sequence_distribution",
        "video_codec",
    }
)


def _require_drivable(field: str) -> None:
    """Fail rather than skip when a field cannot be driven and is not listed."""
    assert field in UNDRIVABLE_FIELDS, (
        f"{field} cannot be driven from its annotation, so it is untested. "
        f"Extend _candidate_values to cover its type, or add it to "
        f"UNDRIVABLE_FIELDS with a reason."
    )


# Fields whose emitted shape legitimately carries a constant companion key.
# Each entry needs a reason; an unexplained entry is a bug being suppressed.
_CONSTANT_KEY_ALLOWLIST: dict[str, set[str]] = {
    # --cache-bust emits {"target": <value>}; the wrapper key is structural.
    "cache_bust": set(),
}


@pytest.mark.parametrize("field", sorted(DATASET_OVERRIDE_FIELDS))
def test_override_emits_no_key_the_user_did_not_set(field: str) -> None:
    """An emitted key that ignores the input value is a materialized default."""
    pair = _value_pair(field)
    if pair is None:
        _require_drivable(field)
        pytest.skip(f"{field} is listed in UNDRIVABLE_FIELDS")
    first, second = pair

    try:
        low = build_dataset(cli(**{field: first}), declared_type=DatasetType.SYNTHETIC)
        high = build_dataset(
            cli(**{field: second}), declared_type=DatasetType.SYNTHETIC
        )
    except (ValueError, ConfigurationError):
        pytest.skip(f"{field} is not valid against a synthetic dataset alone")

    low_leaves, high_leaves = _leaves(low), _leaves(high)
    constant = {
        path
        for path in low_leaves.keys() & high_leaves.keys()
        if low_leaves[path] == high_leaves[path]
    }
    constant -= _CONSTANT_KEY_ALLOWLIST.get(field, set())

    assert not constant, (
        f"build_dataset emitted {sorted(constant)} for --{field.replace('_', '-')} "
        f"with a value independent of the flag, so it is a default rather than "
        f"something the user asked for. Merged onto a YAML dataset it would "
        f"silently overwrite the config file's value. Suppress it in override "
        f"mode, as _apply_implicit_media_batch is."
    )


@pytest.fixture
def trace_yaml(tmp_path: Path) -> Path:
    """A file/trace dataset, for the flags that only apply to one."""
    pool = tmp_path / "trace.jsonl"
    pool.write_text('{"text": "hi", "hash_ids": [1]}\n')
    cfg = tmp_path / "trace.yaml"
    cfg.write_text(
        f"""\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: file
    format: mooncake_trace
    path: {pool}
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


@pytest.fixture
def rich_yaml(tmp_path: Path) -> Path:
    """A synthetic dataset carrying values an override could clobber."""
    cfg = tmp_path / "rich.yaml"
    cfg.write_text(
        """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    random_seed: 11
    entries: 16
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
    )
    return cfg


@pytest.mark.parametrize("field", sorted(DATASET_OVERRIDE_FIELDS))
def test_routed_dataset_field_actually_changes_the_resolved_config(
    field: str, rich_yaml: Path, trace_yaml: Path
) -> None:
    """A field claimed as routed must have an observable effect.

    Checked against both a synthetic and a file dataset: many flags apply to
    only one (synthesis_* is file/public-only, prefix prompts are
    synthetic-only), so requiring an effect on a single type would flag
    correct routing as broken.
    """
    if field in MAGIC_LIST_ONLY_UNDER_CONFIG:
        pytest.skip(f"{field} routes as a sweep parameter in its list form")
    candidates = _candidate_values(field)
    if not candidates:
        _require_drivable(field)
        pytest.skip(f"{field} is listed in UNDRIVABLE_FIELDS")

    # Try every candidate: one may coincide with the model's own default
    # (DatasetSamplingStrategy.SEQUENTIAL does), which would look like "no
    # effect" without meaning the flag is dropped.
    resolutions = []
    errors: list[str] = []
    changed = None
    for config_yaml in (rich_yaml, trace_yaml):
        baseline = resolve_config(cli(), config_yaml).model_dump(mode="json")
        for value in candidates:
            try:
                resolved = resolve_config(
                    cli(**_with_companion(field, value)), config_yaml
                ).model_dump(mode="json")
            except (ValueError, ConfigurationError) as exc:
                errors.append(str(exc))
                continue
            resolutions.append(resolved)
            if resolved != baseline:
                changed = resolved
    if not resolutions:
        # "has no effect on a dataset of type X" is OUR error for a field
        # nothing routes -- the bug, not a domain guard. Anything else (weka
        # -only, baseten-only, needs a companion flag) is a real constraint
        # and the flag is loud rather than dropped, so skipping is right.
        assert not all("have no effect" in e for e in errors), (
            f"--{field.replace('_', '-')} is classified as routed but nothing "
            f"routes it: resolving raises {errors[0]!r}. Route it, or list it "
            f"in UNROUTED_UNDER_CONFIG so the error names the flag properly."
        )
        pytest.skip(f"{field} is not valid against a synthetic dataset alone")
    assert changed is not None, (
        f"--{field.replace('_', '-')} is classified as routed under --config but "
        f"setting it changed nothing in the resolved config -- it is being "
        f"silently dropped. Route it, or list it in UNROUTED_UNDER_CONFIG so "
        f"users get an error."
    )


# ---------------------------------------------------------------------------
# The whole-CLIConfig guarantee
# ---------------------------------------------------------------------------

# Enums with a single member: no second value exists to prove an effect with.
SINGLE_VALUED_FIELDS: frozenset[str] = frozenset(
    {
        # URLSelectionStrategy has exactly one member.
        "url_selection_strategy",
        # Every mode but the default fails validation on its own, and setting
        # a field to the value it already has cannot change anything -- so no
        # observation here can distinguish routed from dropped.
        "wait_for_model_mode",
    }
)


def _companions_for(field: str) -> dict[str, Any]:
    """Flags that must accompany ``field`` for it to mean anything."""
    from aiperf.config.flags._config_flag_routing import COMPANION_ROUTED

    companions: dict[str, Any] = {}
    for companion in COMPANION_ROUTED.get(field, ()):  # type: ignore[call-overload]
        if companion == "model_names":
            companions["model_names"] = ["companion-model"]
    if field.endswith("_stddev"):
        mean_field = field[: -len("_stddev")] + "_mean"
        if mean_field in CLIConfig.model_fields:
            companions[mean_field] = 8
    return companions


@pytest.mark.parametrize("field", sorted(_ROUTED_FIELDS_UNDER_TEST))
def test_routed_field_never_silently_no_ops(
    field: str, rich_yaml: Path, trace_yaml: Path
) -> None:
    """Every routed flag must change the config or raise -- never both quiet.

    This is the guarantee in one assertion, and it covers all of
    ROUTED_UNDER_CONFIG rather than the dataset subset. Sections whose
    builders consume the whole section (OUTPUT/TOKENIZER/ACCURACY/SWEEPING)
    are marked routed wholesale, so a newly-added member of one of them is
    classified as routed automatically; without this test nothing would ever
    check whether that claim is true, and the flag would be silently dropped
    exactly as before.

    Raising is acceptable: a flag rejected for needing a companion, a
    different dataset type, or a streaming endpoint is loud, and loud is the
    property we care about. The forbidden outcome is resolving successfully
    while ignoring what the user asked for.
    """
    if field in MAGIC_LIST_ONLY_UNDER_CONFIG:
        pytest.skip(f"{field} routes as a sweep parameter in its list form")
    if field in SINGLE_VALUED_FIELDS:
        pytest.skip(f"{field} has a single valid value; no effect to observe")
    candidates = _candidate_values(field)
    if not candidates:
        _require_drivable(field)
        pytest.skip(f"{field} is listed in UNDRIVABLE_FIELDS")

    companions = _companions_for(field)
    # Assert per YAML, not across the union: a field that works on a trace
    # dataset and silently no-ops on a synthetic one is a silent drop for
    # every user of a synthetic config, and accumulating `changed` across
    # both fixtures hid exactly that.
    for config_yaml in (rich_yaml, trace_yaml):
        try:
            baseline = resolve_config(cli(**companions), config_yaml).model_dump(
                mode="json"
            )
        except Exception:
            continue
        changed = silent_noop = False
        for value in candidates:
            try:
                resolved = resolve_config(
                    cli(**{field: value}, **companions), config_yaml
                ).model_dump(mode="json")
            except Exception:
                continue  # loud: acceptable
            if resolved != baseline:
                changed = True
            else:
                silent_noop = True

        assert changed or not silent_noop, (
            f"--{field.replace('_', '-')} is classified as routed under "
            f"--config, but with {config_yaml.name} it resolves successfully "
            f"and changes nothing -- it is silently ignored. Route it, or move "
            f"it to UNROUTED_UNDER_CONFIG so the user gets an error naming it."
        )


# One routable flag per dataset type, used as an anchor so the flag under
# test is never the only thing driving the override.
_ANCHOR_FLAGS: dict[str, dict[str, Any]] = {
    "rich": {"random_seed": 4242},
    "trace": {"random_seed": 4242},
}


@pytest.mark.parametrize("field", sorted(DATASET_OVERRIDE_FIELDS))
def test_dataset_flag_is_not_excused_by_a_neighbouring_flag(
    field: str, rich_yaml: Path, trace_yaml: Path
) -> None:
    """A flag must not become silently droppable by being passed with others.

    The single-flag parametrization above cannot see this: the guard used to
    fire only when the whole override came back empty, so any inert flag
    riding along with a routable one was discarded with no diagnostic. Every
    multi-flag command line touching the dataset escaped the guarantee.
    """
    if field in MAGIC_LIST_ONLY_UNDER_CONFIG or field in SINGLE_VALUED_FIELDS:
        pytest.skip(f"{field} is covered by the single-flag invariant only")
    candidates = _candidate_values(field)
    if not candidates:
        _require_drivable(field)
        pytest.skip(f"{field} is listed in UNDRIVABLE_FIELDS")

    companions = _companions_for(field)
    for label, config_yaml in (("rich", rich_yaml), ("trace", trace_yaml)):
        anchor = _ANCHOR_FLAGS[label]
        if field in anchor:
            continue
        try:
            baseline = resolve_config(
                cli(**anchor, **companions), config_yaml
            ).model_dump(mode="json")
        except Exception:
            continue
        changed = silent_noop = False
        # Every candidate, not just one: a bool's False and an enum's first
        # member frequently ARE the resolved default, and setting a field to
        # the value it already holds cannot change anything.
        for value in candidates:
            try:
                resolved = resolve_config(
                    cli(**{field: value}, **anchor, **companions), config_yaml
                ).model_dump(mode="json")
            except Exception:
                continue  # loud: acceptable
            if resolved != baseline:
                changed = True
            else:
                silent_noop = True
        assert changed or not silent_noop, (
            f"--{field.replace('_', '-')} resolves cleanly and changes nothing "
            f"when passed alongside {sorted(anchor)} against {config_yaml.name}. "
            f"Paired with another flag it is silently dropped, even though it "
            f"may error correctly on its own."
        )
