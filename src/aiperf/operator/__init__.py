# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes Operator package.

A kopf-based operator that watches AIPerfJob and AIPerfSweep custom resources
and orchestrates AIPerf benchmarks using JobSet.

Intentionally inert: import submodules directly (``from aiperf.operator import
environment``, ``aiperf.operator.main``, ...). Re-exporting ``main`` here made
every ``aiperf.operator.*`` import pull in kopf and the whole handler tree.

Configuration via environment variables:
    AIPERF_OPERATOR_MONITOR_INTERVAL: Progress monitoring interval in seconds (default: 10.0)
    AIPERF_OPERATOR_MONITOR_INITIAL_DELAY: Initial delay before monitoring starts (default: 5.0)
"""

from __future__ import annotations
