# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive-scale CLI overlay helpers for YAML resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


BASIC_ADAPTIVE_CLI_FIELDS = frozenset(
    {
        "adaptive_scale",
        "adaptive_sustain_duration",
        "adaptive_assessment_period",
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
    if (
        "search_sla" in cli.model_fields_set
        and "adaptive_scale_sla" not in cli.model_fields_set
    ):
        raise ValueError(
            "--adaptive-scale uses --adaptive-scale-sla; --search-sla is reserved "
            "for adaptive-search/grid runs"
        )

    existing = target.get("adaptive_scale")
    adaptive_block = existing if isinstance(existing, dict) else None

    if "adaptive_scale" in fields_set:
        if adaptive_block is not None:
            adaptive_block["enabled"] = bool(cli.adaptive_scale)
        else:
            target["adaptive_scale"] = bool(cli.adaptive_scale)

    if (
        "adaptive_sustain_duration" in fields_set
        and cli.adaptive_sustain_duration is not None
        and adaptive_block is not None
    ):
        adaptive_block["sustain_duration"] = cli.adaptive_sustain_duration

    if (
        "adaptive_assessment_period" in fields_set
        and cli.adaptive_assessment_period is not None
        and adaptive_block is not None
    ):
        adaptive_block["assessment_period"] = cli.adaptive_assessment_period

    if "adaptive_scale_sla" in fields_set and cli.adaptive_scale_sla:
        from aiperf.orchestrator.search_planner.parsing import parse_sla_filter

        parsed_sla: list[dict[str, Any]] = []
        for value in cli.adaptive_scale_sla:
            try:
                parsed_sla.append(parse_sla_filter(value).model_dump(mode="json"))
            except TypeError as exc:
                message = str(exc).replace("--search-sla", "--adaptive-scale-sla")
                raise TypeError(message) from exc
        target["sla"] = parsed_sla
