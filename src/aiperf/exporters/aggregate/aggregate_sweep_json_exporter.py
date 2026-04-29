# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for sweep aggregate results.

Serializes the dict returned by
:meth:`aiperf.orchestrator.aggregation.sweep.SweepAnalyzer.compute` —
schema preserved byte-compatible with PR #699.
"""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.exporters.aggregate.aggregate_base_exporter import AggregateBaseExporter


class AggregateSweepJsonExporter(AggregateBaseExporter):
    """Exports sweep aggregate results to JSON format.

    The output JSON contains the four sweep sections produced by
    :meth:`SweepAnalyzer.compute`:

    - ``metadata`` — sweep parameter definitions + ``num_combinations``.
    - ``per_combination_metrics`` — list of ``{parameters, metrics}`` per cell.
    - ``best_configurations`` — per-objective single-best
      (``best_throughput``, ``best_latency_p99``).
    - ``pareto_optimal`` — list of non-dominated parameter dicts.

    Constructor surface differs from siblings: takes the sweep dict
    directly rather than wrapping in :class:`AggregateResult`, because
    :class:`SweepAnalyzer` is already the authoritative producer.

    Example:
        >>> sweep_dict = SweepAnalyzer.compute(stats, sweep_parameters)  # doctest: +SKIP
        >>> exp = AggregateSweepJsonExporter(config, sweep_dict)  # doctest: +SKIP
        >>> await exp.export()  # doctest: +SKIP
    """

    def __init__(self, config, sweep_dict: dict[str, Any], **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._sweep_dict = sweep_dict

    def get_file_name(self) -> str:
        """Return ``"profile_export_aiperf_sweep.json"``."""
        return "profile_export_aiperf_sweep.json"

    def _generate_content(self) -> str:
        """Serialize the four sweep sections as indented JSON.

        Numpy scalars are unwrapped by ``OPT_SERIALIZE_NUMPY``; this is
        load-bearing because :class:`SweepAnalyzer` propagates numpy
        ``float64`` values from the underlying ``ConfidenceMetric``.
        """
        output = {
            "metadata": self._sweep_dict.get("metadata", {}),
            "per_combination_metrics": self._sweep_dict.get(
                "per_combination_metrics", []
            ),
            "best_configurations": self._sweep_dict.get("best_configurations", {}),
            "pareto_optimal": self._sweep_dict.get("pareto_optimal", []),
        }
        return orjson.dumps(
            output,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        ).decode("utf-8")
