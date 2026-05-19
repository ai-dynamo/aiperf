# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import ConfigDict

from aiperf.config.base import BaseConfig

__all__ = ["MetricsConfig"]


class MetricsConfig(BaseConfig):
    """Configuration for benchmark metric aggregation behavior.

    Currently empty — list-valued record metrics aggregate via t-digest at
    first-touch (see :mod:`aiperf.metrics.list_metric_aggregation`); the
    sketch compression is tunable via ``AIPERF_METRICS_TDIGEST_COMPRESSION``.
    Reserved as the home for any future user-facing metric-aggregation knobs.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)
