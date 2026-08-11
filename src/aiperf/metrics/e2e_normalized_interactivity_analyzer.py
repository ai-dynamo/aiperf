# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E Normalized Interactivity analyzer: compute the
``e2e_normalized_interactivity_{p75,p90}`` family from the column store at
summarize-time.

The metrics themselves are defined in
:mod:`aiperf.metrics.types.e2e_normalized_interactivity_metrics` and defer their
derivation; this module owns the columnar computation and injection, mirroring
:mod:`aiperf.metrics.replay_sched_lag_analyzer` and
:mod:`aiperf.metrics.derived_latency`.

The value reproduces InferenceX's derivation exactly: the percentile is taken
over the per-request seconds-per-output-token ratio ``request_latency_s / OSL``
and then inverted, so ``p90`` is the effective token rate of the 90th-percentile
(slowest) request::

    p90 interactivity = 1 / p90(request_latency_s / OSL)

Request-weighted: every request with positive, finite ``request_latency``,
``output_sequence_length``, and -- when the columns are present --
``time_to_first_token`` and ``input_sequence_length`` contributes one sample,
matching InferenceX's filter (all four fields > 0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.finite import is_finite_value
from aiperf.common.models import MetricResult
from aiperf.metrics.types.e2e_normalized_interactivity_metrics import (
    E2ENormalizedInteractivityP75Metric,
    E2ENormalizedInteractivityP90Metric,
    E2ENormalizedInteractivityPercentileBase,
)
from aiperf.metrics.types.input_sequence_length_metric import InputSequenceLengthMetric
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.ttft_metric import TTFTMetric

if TYPE_CHECKING:
    from aiperf.metrics.column_store import ColumnStore


# The ``percentile`` each class declares is the single source of truth for both
# its identity and the ratio percentile the analyzer inverts.
_PERCENTILE_METRICS: tuple[type[E2ENormalizedInteractivityPercentileBase], ...] = (
    E2ENormalizedInteractivityP75Metric,
    E2ENormalizedInteractivityP90Metric,
)

# Columns that gate a request in/out of the sample, matching InferenceX's
# "all fields > 0" filter. request_latency and OSL are required (no-op without
# them); TTFT and ISL are applied only when the run produced them.
_FILTER_ONLY_TAGS = (TTFTMetric.tag, InputSequenceLengthMetric.tag)


def inject_e2e_normalized_interactivity_metrics(
    store: ColumnStore,
    results: dict[str, MetricResult],
    mask: NDArray[np.bool_] | None = None,
) -> None:
    """Inject the E2E Normalized Interactivity percentile family.

    No-op when ``request_latency`` or ``output_sequence_length`` are absent, or
    when no request passes the positive-and-finite filter. Pure side effect on
    ``results``.
    """
    tags = store.numeric_tags()
    if (
        RequestLatencyMetric.tag not in tags
        or OutputSequenceLengthMetric.tag not in tags
    ):
        return

    def _column(tag: str) -> NDArray[np.float64]:
        col = store.numeric(tag)
        return col[mask] if mask is not None else col

    latency_ns = _column(RequestLatencyMetric.tag)
    osl = _column(OutputSequenceLengthMetric.tag)

    keep = np.isfinite(latency_ns) & (latency_ns > 0) & np.isfinite(osl) & (osl > 0)
    for tag in _FILTER_ONLY_TAGS:
        if tag in tags:
            col = _column(tag)
            keep &= np.isfinite(col) & (col > 0)

    if not keep.any():
        return

    # Seconds per output token, per surviving request. Drop any non-finite
    # ratio (an extreme latency/OSL can overflow the division) so the
    # percentile and its reciprocal stay finite -- injected metric values must
    # be finite per the NaN/Inf discipline.
    ratio = (latency_ns[keep] / NANOS_PER_SECOND) / osl[keep]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return

    for cls in _PERCENTILE_METRICS:
        ratio_p = float(np.percentile(ratio, cls.percentile))
        if ratio_p <= 0:
            continue
        value = 1.0 / ratio_p
        # Guard the reciprocal too: a subnormal ratio_p can overflow to inf.
        if not is_finite_value(value):
            continue
        results[cls.tag] = MetricResult(
            tag=cls.tag,
            header=cls.header,
            unit=str(cls.unit),
            avg=value,
            count=int(ratio.size),
            console_group=cls.console_group,
        )
