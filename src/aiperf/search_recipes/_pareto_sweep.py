# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ParetoSweep`` recipe (closes GitHub issue #874).

Sweeps paired ISL/OSL workload shapes across a concurrency list and emits
a Pareto-frontier JSON artifact.
"""

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.search_recipes._base import (
    PostProcessSpec,
    SearchRecipe,
    SearchRecipeContext,
    SearchRecipeOutput,
    require_streaming,
)
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec
from aiperf.search_recipes._pareto_sweep_parser import parse_isl_osl_pairs

__all__ = ["ParetoSweep"]


class ParetoSweep(SearchRecipe):
    """Sweep paired ISL/OSL shapes across a concurrency list; emit a Pareto frontier.

    Each ``(isl, osl)`` pair from ``--isl-osl-pairs`` is crossed with each
    value in ``--concurrency`` (a magic-list flag this recipe consumes
    directly). The recipe pre-flattens to a ``ScenarioSweep`` so ISL and OSL
    stay paired (vs the Cartesian product a grid sweep would emit).

    Streaming is required because the Pareto y-axis is
    ``output_token_throughput`` (a streaming-only metric).

    Closes GitHub issue #874.

    Example:
        aiperf profile --search-recipe pareto-sweep \\
            --isl-osl-pairs 128/128,512/256,2048/512 \\
            --concurrency 1,4,16,64,256 \\
            --streaming
    """

    name: ClassVar[str] = "pareto-sweep"
    description: ClassVar[str] = (
        "Sweep paired ISL/OSL shapes across a concurrency list; emit a Pareto "
        "frontier (time_to_first_token p95 vs output_token_throughput avg)."
    )
    pareto_axes: ClassVar[ParetoAxesSpec | None] = ParetoAxesSpec(
        x_metric="time_to_first_token",
        x_stat="p95",
        x_minimize=True,
        y_metric="output_token_throughput",
        y_stat="avg",
        y_maximize=True,
        series_keys=("isl", "osl"),
    )
    consumed_magic_lists: ClassVar[frozenset[str]] = frozenset({"concurrency"})

    _DEFAULT_CONCURRENCY: ClassVar[tuple[int, ...]] = (1, 4, 16, 64, 256)
    _DATASET_NAME: ClassVar[str] = "main"
    _PHASE_NAME: ClassVar[str] = "profiling"

    def expand(self, ctx: SearchRecipeContext) -> SearchRecipeOutput:
        """Compile paired ISL/OSL inputs and concurrencies into scenario runs.

        Consumes ``isl_osl_pairs`` and ``concurrency`` from ``ctx.sweep_overrides``.
        The output is a pre-flattened ``ScenarioSweep`` so ISL and OSL remain paired,
        and includes Pareto export metadata for the aggregate artifact.
        """
        require_streaming(
            ctx.benchmark_config.endpoint,
            recipe_name=self.name,
            reason="output_token_throughput is a streaming-only metric",
        )

        raw_pairs = ctx.sweep_overrides.get("isl_osl_pairs")
        if not raw_pairs:
            raise ValueError(
                f"recipe {self.name!r} requires --isl-osl-pairs "
                "(e.g. '128/128,256/256')."
            )
        pairs = parse_isl_osl_pairs(str(raw_pairs))

        concurrencies = self._resolve_concurrencies(ctx.sweep_overrides)
        if len(pairs) * len(concurrencies) < 2:
            raise ValueError(
                f"recipe {self.name!r}: a Pareto sweep with a single point "
                "is meaningless. Pass at least 2 pairs OR at least 2 "
                "concurrency values."
            )

        scenarios = [
            self._build_scenario(isl, osl, conc)
            for (isl, osl) in pairs
            for conc in concurrencies
        ]
        return SearchRecipeOutput(
            scenarios=scenarios,
            post_process=PostProcessSpec(
                handler="pareto_sweep_export",
                params={
                    "x_metric": "time_to_first_token",
                    "x_stat": "p95",
                    "y_metric": "output_token_throughput",
                    "y_stat": "avg",
                    "isl_key": "isl",
                    "osl_key": "osl",
                    "concurrency_key": "concurrency",
                },
                output_filename="pareto_sweep.json",
            ),
        )

    def _resolve_concurrencies(self, overrides: dict[str, Any]) -> list[int]:
        raw = overrides.get("concurrency")
        if raw is None:
            return list(self._DEFAULT_CONCURRENCY)
        if isinstance(raw, list):
            return [int(v) for v in raw]
        return [int(raw)]

    def _build_scenario(self, isl: int, osl: int, conc: int) -> dict[str, Any]:
        return {
            "name": f"shape_{isl}_{osl}_c{conc}",
            "values": {"isl": isl, "osl": osl, "concurrency": conc},
            "benchmark": {
                "datasets": [
                    {"name": self._DATASET_NAME, "prompts": {"isl": isl, "osl": osl}},
                ],
                "phases": [
                    {"name": self._PHASE_NAME, "concurrency": conc},
                ],
            },
        }
