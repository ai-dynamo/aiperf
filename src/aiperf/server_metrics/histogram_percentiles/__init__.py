# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram percentile models and computation functions.

This package provides percentile estimation for Prometheus histograms using
a polynomial histogram algorithm that:
- Learns per-bucket mean positions from single-bucket scrape intervals
- Learns per-bucket variance from multiple single-bucket intervals
- Uses exact sum constraint to improve observation placement
- Back-calculates +Inf bucket observations for accurate tail percentiles

Based on HistogramTools research (arXiv 2504.00001). Variance tracking enables
optimal observation generation strategies (F3 two-point mass, blended,
variance-aware) based on learned distribution characteristics within each bucket.

Typical accuracy on LLM inference workloads: ~20% average P99 error vs ~950%
for standard Prometheus linear interpolation.
"""

from aiperf.server_metrics.histogram_percentiles._bucket_utils import (
    _MAX_OBSERVATIONS,
    _cumulative_to_per_bucket,
    _estimate_bucket_sums,
    _estimate_inf_bucket_observations,
    _get_bucket_bounds,
    accumulate_bucket_statistics,
)
from aiperf.server_metrics.histogram_percentiles._estimate import (
    compute_estimated_percentiles,
)
from aiperf.server_metrics.histogram_percentiles._models import (
    BucketStatistics,
    EstimatedPercentiles,
)
from aiperf.server_metrics.histogram_percentiles._observation_generators import (
    _generate_blended_observations,
    _generate_f3_observations,
    _generate_variance_aware_observations,
)
from aiperf.server_metrics.histogram_percentiles._prometheus import (
    compute_prometheus_percentiles,
)
from aiperf.server_metrics.histogram_percentiles._sum_constraint import (
    _generate_observations_with_sum_constraint,
)

# Underscore-prefixed names are re-exported because existing tests import them
# directly from this package path; they remain implementation details.
__all__ = [
    "_MAX_OBSERVATIONS",
    "_cumulative_to_per_bucket",
    "_estimate_bucket_sums",
    "_estimate_inf_bucket_observations",
    "_generate_blended_observations",
    "_generate_f3_observations",
    "_generate_observations_with_sum_constraint",
    "_generate_variance_aware_observations",
    "_get_bucket_bounds",
    "BucketStatistics",
    "EstimatedPercentiles",
    "accumulate_bucket_statistics",
    "compute_estimated_percentiles",
    "compute_prometheus_percentiles",
]
