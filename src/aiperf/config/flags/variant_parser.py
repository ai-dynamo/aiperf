# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse `--variant` CLI strings and discover the short-alias table.

Two public entry points:

- :func:`parse_variant`: parses a single `[name:] key=value, ...` string.
- :func:`build_alias_table`: walks the ``CLIConfig`` model tree once and
  builds a `{short_alias: dotted_field_path}` map. Cached on first call.

Used by ``aiperf.config.flags.converter._apply_variants_scenario_sweep`` to
translate `--variant` occurrences into a ``ScenarioSweep`` block.
"""

from __future__ import annotations

import types
import typing
from typing import Any, get_args, get_origin

from cyclopts import Parameter
from pydantic.fields import FieldInfo

from aiperf.config.base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter


def parse_variant(s: str) -> tuple[str | None, dict[str, Any]]:
    """Parse `'[name:] key=value, key=value'` into `(name, kvpairs)`.

    Returns `(None, kvpairs)` when no `name:` prefix is present. Coerces
    each value to int, then float, then strips quotes and returns str.
    Booleans `true`/`false` are preserved as Python bools.

    Raises:
        ValueError: on malformed input (no `=`, empty body, duplicate key).
    """
    raw = s.strip()
    if not raw:
        raise ValueError("variant string is empty")

    name: str | None = None
    body = raw
    head, sep, tail = raw.partition(":")
    if sep and "=" not in head:
        candidate = head.strip()
        if not candidate:
            raise ValueError(f"variant {s!r}: empty name before ':'")
        name = candidate
        body = tail.strip()

    if not body:
        raise ValueError(f"variant {s!r}: empty body")

    pairs: dict[str, Any] = {}
    for chunk in body.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"variant {s!r}: token {chunk!r} is missing '=' (expected key=value)"
            )
        key, _, value = chunk.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"variant {s!r}: empty key in token {chunk!r}")
        if key in pairs:
            raise ValueError(f"variant {s!r}: duplicate key {key!r}")
        pairs[key] = _coerce(value)

    if not pairs:
        raise ValueError(f"variant {s!r}: no key=value pairs")
    return name, pairs


def _coerce(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


_alias_table_cache: dict[str, str] | None = None


def build_alias_table() -> dict[str, str]:
    """Walk CLIConfig and return `{cli_alias: dotted_field_path}`.

    Each `CLIParameter.name` flag (without the leading `--`) becomes a key.
    Both the kebab-case form (`request-rate`) and the snake_case form
    (`request_rate`) register so users can spell either way in `--variant`.
    """
    global _alias_table_cache
    if _alias_table_cache is not None:
        return _alias_table_cache

    from aiperf.config.flags.cli_config import CLIConfig

    table: dict[str, str] = {}
    _walk_model(CLIConfig, "", table)
    _alias_table_cache = table
    return table


def _walk_model(model_cls: type[BaseConfig], prefix: str, out: dict[str, str]) -> None:
    for field_name, info in model_cls.model_fields.items():
        path = f"{prefix}.{field_name}" if prefix else field_name
        nested = _nested_baseconfig_type(info)
        if nested is not None:
            _walk_model(nested, path, out)
            continue
        for alias in _aliases_for(info):
            out.setdefault(alias, path)


def _nested_baseconfig_type(info: FieldInfo) -> type[BaseConfig] | None:
    annotation = info.annotation
    if annotation is None:
        return None
    return _extract_baseconfig(annotation)


def _extract_baseconfig(tp: Any) -> type[BaseConfig] | None:
    if isinstance(tp, type) and issubclass(tp, BaseConfig):
        return tp
    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        for arg in get_args(tp):
            if arg is type(None):
                continue
            inner = _extract_baseconfig(arg)
            if inner is not None:
                return inner
    return None


def _aliases_for(info: FieldInfo) -> list[str]:
    aliases: list[str] = []
    for flag in _flag_names(info):
        if not flag.startswith("--"):
            continue
        body = flag[2:]
        if not body:
            continue
        aliases.append(body)
        if "-" in body:
            aliases.append(body.replace("-", "_"))
    return aliases


def _flag_names(info: FieldInfo) -> list[str]:
    for meta in info.metadata:
        if isinstance(meta, (CLIParameter, Parameter)):
            name = meta.name
            if name is None:
                continue
            if isinstance(name, str):
                return [name]
            return list(name)
    return []


def reset_alias_table_cache_for_tests() -> None:
    """Drop the cached alias table. Tests may need this when monkeypatching."""
    global _alias_table_cache
    _alias_table_cache = None
