# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis tools that operate on completed aiperf run artifacts.

Includes vectorized sweepline algorithms (concurrency, throughput,
tokens-in-flight) used by the metrics accumulator, alongside CLI helper
scripts for profile-export analysis, memory calibration, and speed-bench
reporting.
"""
