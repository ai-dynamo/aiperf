# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis tools that operate on completed aiperf run artifacts.

Includes vectorized analysis algorithms for concurrency, throughput, and
ramp detection (sweepline, ramp_detection, stationarity, bootstrap,
energy_analyzer), alongside CLI helper scripts for profile-export
analysis, memory calibration, and speed-bench reporting.
"""
