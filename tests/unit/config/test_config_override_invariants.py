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
)
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.resolver import resolve_config
from aiperf.config.loader.errors import ConfigurationError


def cli(**kwargs: object) -> CLIConfig:
    return CLIConfig(**CLIConfig(**kwargs).model_dump(exclude_unset=True))  # type: ignore[arg-type]


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
            annotation = args[0]
            continue
        return annotation


def _candidate_values(field: str) -> list[Any]:
    """Valid values to drive ``field`` with, or [] if none can be generated.

    Returning [] (rather than guessing) keeps the test honest: a field it
    cannot drive is reported as skipped instead of silently passing.
    """
    info = CLIConfig.model_fields.get(field)
    if info is None:
        return []
    annotation = _unwrap(info.annotation)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return list(annotation)
    if annotation is bool:
        return [True, False]
    if annotation is int:
        return [2, 3]
    if annotation is float:
        return [2.0, 3.0]
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
        pytest.skip(f"no two distinct auto-generated values for {field}")
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
    field: str, rich_yaml: Path
) -> None:
    """A field claimed as routed must have an observable effect."""
    if field in MAGIC_LIST_ONLY_UNDER_CONFIG:
        pytest.skip(f"{field} routes as a sweep parameter in its list form")
    candidates = _candidate_values(field)
    if not candidates:
        pytest.skip(f"no auto-generated value for {field}")

    baseline = resolve_config(cli(), rich_yaml).model_dump(mode="json")
    # Try every candidate: one of them may coincide with the model's own
    # default (DatasetSamplingStrategy.SEQUENTIAL does), which would show up
    # as "no effect" without meaning the flag is dropped.
    resolutions = []
    for value in candidates:
        try:
            resolutions.append(
                resolve_config(cli(**{field: value}), rich_yaml).model_dump(mode="json")
            )
        except (ValueError, ConfigurationError):
            continue
    if not resolutions:
        pytest.skip(f"{field} is not valid against a synthetic dataset alone")
    changed = next(
        (r for r in resolutions if r != baseline),
        baseline,
    )

    assert changed != baseline, (
        f"--{field.replace('_', '-')} is classified as routed under --config but "
        f"setting it changed nothing in the resolved config -- it is being "
        f"silently dropped. Route it, or list it in UNROUTED_UNDER_CONFIG so "
        f"users get an error."
    )
