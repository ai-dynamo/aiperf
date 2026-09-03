# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adaptive-search planner construction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
