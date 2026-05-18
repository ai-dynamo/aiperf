# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hypothesis configuration for AIPerf property tests.

Limits hypothesis runtime under CI by lowering the deadline tolerance and
suppressing the slow-data-generation health check (some of our strategies
build dicts of dicts, which trips ``data_too_large``).
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "aiperf_property",
    deadline=None,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)
settings.load_profile("aiperf_property")
