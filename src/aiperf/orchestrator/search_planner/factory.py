# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adaptive-search planner construction."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from aiperf.config import BenchmarkPlan
    from aiperf.orchestrator.search_planner.base import SearchPlanner

logger = logging.getLogger(__name__)


def build_search_planner(plan: BenchmarkPlan) -> SearchPlanner | None:
    """Build the outer-loop SearchPlanner for an adaptive-search plan.

    Returns None outside adaptive search. Planner selection is dispatched via
    the plugin registry, except that two or more SLO tiers select the shared
    multi-tier planner.

    Args:
        plan: Canonical benchmark plan produced by Config-v2 plan construction.

    Returns:
        A configured search planner, or None for non-adaptive plans.
    """
    from aiperf.config.sweep import AdaptiveSearchSweep

    if not isinstance(plan.sweep, AdaptiveSearchSweep):
        return None

    config = plan.sweep
    real_dims = [dim for dim in config.search_space if dim.kind == "real"]
    if real_dims:
        from aiperf.config.config import BenchmarkConfig
        from aiperf.config.sweep import _set_nested_value

        base = plan.configs[0] if plan.configs else None
        if base is not None:
            for dim in real_dims:
                lo = float(dim.lo)
                hi = float(dim.hi)
                mid = (lo + hi) / 2
                probe_val = math.nextafter(mid, hi)
                if probe_val <= lo or probe_val >= hi:
                    probe_val = math.nextafter(mid, lo)
                probe = base.model_dump(  # type: ignore[union-attr]
                    mode="python", exclude_none=True, context={"include_secrets": True}
                )
                _set_nested_value(probe, dim.path, probe_val)
                try:
                    BenchmarkConfig.model_validate(probe)
                except ValidationError as exc:
                    leaf = dim.path.rsplit(".", 1)[-1]
                    raise ValueError(
                        f"search dimension {dim.path!r} has kind='real' but targets "
                        f"int-typed field {leaf!r}; the planner would propose "
                        f"fractional values that the config rejects. Use "
                        f"kind='int', or target a float-typed field (e.g. 'rate')."
                    ) from exc
    if len(config.sla_tiers) >= 2:
        from aiperf.orchestrator.search_planner.multi_tier_planner import (
            MultiTierPlanner,
        )
        from aiperf.plugin.enums import SearchPlannerType

        if config.planner != SearchPlannerType.SMOOTH_ISOTONIC:
            logger.warning(
                "The search algorithm for --search-style %s is not used when "
                "--search-sla-tier is active; multi-tier uses its own "
                "bracket/bisection method. The style's precision and warmup "
                "settings still apply.",
                config.planner,
            )
        return MultiTierPlanner(plan.configs[0], config, config.sla_tiers)

    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    planner_class = plugins.get_class(PluginType.SEARCH_PLANNER, str(config.planner))
    return planner_class(plan.configs[0], config)
