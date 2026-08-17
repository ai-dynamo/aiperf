# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-dispatch config resolver.

Runs AFTER :class:`~aiperf.config.resolution.resolvers.ScenarioResolver` so any
scenario-driven cache-bust auto-fill is already in place when this
step derives the wrap default from it. Reads ``run.cfg`` and populates two
graph-plane fields on ``run.resolved``:

- ``allow_dataset_wrap``: honors the explicit raw user value stashed on
  ``dataset.synthesis.allow_dataset_wrap`` (``--allow-dataset-wrap`` /
  ``--no-allow-dataset-wrap``); when unset, derives the default from the
  resolved cache-bust target (``cache_bust != NONE``).
- ``dataset_sampling_strategy``: surfaces the default dataset's ``sampling``
  strategy (the linear path lands per-dataset strategies on
  ``dataset_sampling_strategies``, but graph workloads skip file resolution).
- Warmup-isolation cache-bust targets are rejected because the graph payload
  path cannot yet keep their marker out of profiling payloads.

No-op for non-graph runs -- detection is the single memoized
``resolve_graph_workload`` accessor every consumer shares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger

if TYPE_CHECKING:
    from aiperf.config.dataset.config import DatasetConfig
    from aiperf.config.resolution.plan import BenchmarkRun

_logger = AIPerfLogger(__name__)


class GraphDispatchResolver:
    """Derive graph-plane dispatch defaults after the scenario lock is applied."""

    def resolve(self, run: BenchmarkRun) -> None:
        """Populate ``allow_dataset_wrap`` + ``dataset_sampling_strategy``.

        No-op unless the run is a graph workload. Reads the (possibly
        scenario-auto-filled) ``get_cache_bust_target()`` to derive the wrap default
        when the user left ``--allow-dataset-wrap`` unset.
        """
        from aiperf.dataset.graph.workload_detect import resolve_graph_workload

        if resolve_graph_workload(run) is None:
            return

        from aiperf.common.enums import CacheBustTarget

        dataset = run.cfg.get_default_dataset()
        cache_bust = run.cfg.get_cache_bust_target()
        if cache_bust in (
            CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
            CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
        ):
            raise ValueError(
                "cache_bust targets warmup_isolation_system and "
                "warmup_isolation_first_turn are not compatible with agent_graph "
                "timing mode because its payload materializer cannot keep the warmup "
                "marker out of profiling payloads. Use cache_bust=none or an RID-based "
                "target such as first_turn_prefix."
            )

        raw_wrap = self._raw_allow_dataset_wrap(dataset)
        if raw_wrap is not None:
            run.resolved.allow_dataset_wrap = bool(raw_wrap)
        else:
            run.resolved.allow_dataset_wrap = cache_bust != CacheBustTarget.NONE
        _logger.debug(
            lambda: f"Resolved allow_dataset_wrap={run.resolved.allow_dataset_wrap} "
            f"(explicit={raw_wrap is not None}, cache_bust={cache_bust})"
        )

        run.resolved.dataset_sampling_strategy = dataset.sampling

    @staticmethod
    def _raw_allow_dataset_wrap(dataset: DatasetConfig) -> bool | None:
        """Return the raw explicit ``synthesis.allow_dataset_wrap`` (None if unset).

        The CLI flag's raw value lands on the default dataset's synthesis block;
        an absent synthesis block or absent field both read as unset (None), so
        the caller falls back to the cache-bust-derived default.
        """
        synthesis = dataset.synthesis
        return synthesis.allow_dataset_wrap if synthesis is not None else None
