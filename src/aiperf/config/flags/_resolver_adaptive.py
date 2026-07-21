# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive-scale CLI overlay helpers for YAML resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.config.flags._adaptive_control_cli import (
    parse_adaptive_scale_control,
    reject_mixed_adaptive_control_cli,
)

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


BASIC_ADAPTIVE_CLI_FIELDS = frozenset(
    {
        "adaptive_scale",
        "adaptive_sustain_duration",
        "adaptive_assessment_period",
        "adaptive_scale_control",
        "adaptive_control_variable",
        "adaptive_control_min",
        "adaptive_control_max",
        "adaptive_scale_sla",
    }
)


def apply_basic_adaptive_scale_overrides(
    target: dict[str, Any], cli: CLIConfig
) -> None:
    """Overlay the small adaptive-scale CLI surface onto a YAML phase."""
    fields_set = cli.model_fields_set & BASIC_ADAPTIVE_CLI_FIELDS
    if not fields_set:
        return
    _reject_search_sla_for_adaptive(cli)

    adaptive_block = _current_adaptive_block(target)
    _apply_adaptive_enabled(target, cli, fields_set, adaptive_block)
    _apply_adaptive_durations(target, cli, fields_set)
    reject_mixed_adaptive_control_cli(cli.model_fields_set)
    _apply_compact_control(target, cli, fields_set)
    _apply_explicit_control(target, cli, fields_set)
    _apply_adaptive_sla(target, cli, fields_set)


def _reject_search_sla_for_adaptive(cli: CLIConfig) -> None:
    if (
        "search_sla" in cli.model_fields_set
        and "adaptive_scale_sla" not in cli.model_fields_set
    ):
        raise ValueError(
            "--adaptive-scale uses --adaptive-scale-sla; --search-sla is reserved "
            "for adaptive-search/grid runs"
        )


def _current_adaptive_block(target: dict[str, Any]) -> dict[str, Any] | None:
    existing = target.get("adaptive_scale")
    return existing if isinstance(existing, dict) else None


def _adaptive_scale_enabled(target: dict[str, Any], cli: CLIConfig) -> bool:
    if "adaptive_scale" in cli.model_fields_set:
        return bool(cli.adaptive_scale)
    existing = target.get("adaptive_scale")
    if isinstance(existing, dict):
        return bool(existing.get("enabled", True))
    return bool(existing)


def _ensure_adaptive_block(target: dict[str, Any]) -> dict[str, Any]:
    adaptive_block = _current_adaptive_block(target)
    if adaptive_block is None:
        adaptive_block = {}
        target["adaptive_scale"] = adaptive_block
    return adaptive_block


def _apply_adaptive_enabled(
    target: dict[str, Any],
    cli: CLIConfig,
    fields_set: set[str],
    adaptive_block: dict[str, Any] | None,
) -> None:
    if "adaptive_scale" not in fields_set:
        return
    if adaptive_block is not None:
        adaptive_block["enabled"] = bool(cli.adaptive_scale)
    else:
        target["adaptive_scale"] = bool(cli.adaptive_scale)


def _apply_adaptive_durations(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    duration_fields = {
        "adaptive_sustain_duration",
        "adaptive_assessment_period",
    }
    if not fields_set.intersection(duration_fields):
        return
    adaptive_block = _ensure_adaptive_block(target)
    if cli.adaptive_sustain_duration is not None:
        adaptive_block["sustain_duration"] = cli.adaptive_sustain_duration
    if cli.adaptive_assessment_period is not None:
        adaptive_block["assessment_period"] = cli.adaptive_assessment_period


def _apply_compact_control(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if "adaptive_scale_control" not in fields_set or not cli.adaptive_scale_control:
        return
    adaptive_block = _ensure_adaptive_block(target)
    control = adaptive_block.setdefault("control", {})
    parsed_control = parse_adaptive_scale_control(cli.adaptive_scale_control)
    control["variable"] = parsed_control["adaptive_control_variable"]
    control["min"] = parsed_control["adaptive_control_min"]
    control["max"] = parsed_control["adaptive_control_max"]


def _apply_explicit_control(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    explicit_fields = {
        "adaptive_control_variable",
        "adaptive_control_min",
        "adaptive_control_max",
    }
    if not fields_set.intersection(explicit_fields):
        return
    control = _ensure_adaptive_block(target).setdefault("control", {})
    if "adaptive_control_variable" in fields_set and cli.adaptive_control_variable:
        control["variable"] = cli.adaptive_control_variable
    if "adaptive_control_min" in fields_set and cli.adaptive_control_min is not None:
        control["min"] = cli.adaptive_control_min
    if "adaptive_control_max" in fields_set and cli.adaptive_control_max is not None:
        control["max"] = cli.adaptive_control_max


def _apply_adaptive_sla(
    target: dict[str, Any], cli: CLIConfig, fields_set: set[str]
) -> None:
    if "adaptive_scale_sla" not in fields_set or not cli.adaptive_scale_sla:
        return
    from aiperf.orchestrator.search_planner.parsing import parse_sla_filter

    parsed_sla: list[dict[str, Any]] = []
    for value in cli.adaptive_scale_sla:
        try:
            parsed_sla.append(parse_sla_filter(value).model_dump(mode="json"))
        except TypeError as exc:
            message = str(exc).replace("--search-sla", "--adaptive-scale-sla")
            raise TypeError(message) from exc
    if not _adaptive_scale_enabled(target, cli):
        raise ValueError("--adaptive-scale-sla requires --adaptive-scale")
    target["sla"] = parsed_sla
    adaptive_block = _current_adaptive_block(target)
    if adaptive_block is not None:
        adaptive_block["sla"] = parsed_sla
