# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive scale YAML lowering helpers for phase config."""

from __future__ import annotations

from aiperf.config.sweep.adaptive import SLAFilter


def normalize_adaptive_sla(sla: dict[str, object]) -> list[SLAFilter]:
    """Lower compact metric/stat/op SLA YAML into SLAFilter objects."""
    filters: list[SLAFilter] = []
    for metric_tag, stats in sla.items():
        if not isinstance(stats, dict):
            raise ValueError("adaptive_scale.sla entries must map metric tags to stats")
        for stat, ops in stats.items():
            if not isinstance(ops, dict):
                raise ValueError(
                    "adaptive_scale.sla stats must map operators to thresholds"
                )
            for op, threshold in ops.items():
                filters.append(
                    SLAFilter(
                        metric_tag=metric_tag,
                        stat=stat,
                        op=op,
                        threshold=threshold,
                    )
                )
    return filters


_ADAPTIVE_SCALE_FIELD_MAP = {
    "control_variable": "adaptive_control_variable",
    "controlVariable": "adaptive_control_variable",
    "min_concurrency": "adaptive_scale_min_concurrency",
    "minConcurrency": "adaptive_scale_min_concurrency",
    "window": "adaptive_assessment_period",
    "assessment_period": "adaptive_assessment_period",
    "assessmentPeriod": "adaptive_assessment_period",
    "min_completed_requests": "adaptive_min_completed_requests",
    "minCompletedRequests": "adaptive_min_completed_requests",
    "sustain_duration": "adaptive_sustain_duration",
    "sustainDuration": "adaptive_sustain_duration",
}


_ADAPTIVE_SCALE_STRATEGY_FIELD_MAP = {
    "type": "adaptive_scale_strategy_type",
    "step_policy": "adaptive_scale_step_policy",
    "stepPolicy": "adaptive_scale_step_policy",
    "base_step": "adaptive_scale_base_step",
    "baseStep": "adaptive_scale_base_step",
    "max_step_multiplier": "adaptive_scale_max_step_multiplier",
    "maxStepMultiplier": "adaptive_scale_max_step_multiplier",
    "step_percent": "adaptive_scale_step_percent",
    "stepPercent": "adaptive_scale_step_percent",
}


def _copy_mapped_fields(
    lowered: dict[str, object],
    source_data: dict[str, object],
    field_map: dict[str, str],
) -> None:
    for source, target in field_map.items():
        if source in source_data:
            lowered[target] = source_data[source]


def lower_adaptive_scale_details(
    lowered: dict[str, object], block: dict[str, object]
) -> None:
    lowered["adaptive_scale"] = bool(block.get("enabled", True))
    _copy_mapped_fields(lowered, block, _ADAPTIVE_SCALE_FIELD_MAP)

    strategy = block.get("strategy", {})
    if isinstance(strategy, dict):
        _copy_mapped_fields(lowered, strategy, _ADAPTIVE_SCALE_STRATEGY_FIELD_MAP)

    sla = block.get("sla")
    if isinstance(sla, list):
        lowered["sla"] = sla
    elif isinstance(sla, dict):
        lowered["sla"] = normalize_adaptive_sla(sla)
