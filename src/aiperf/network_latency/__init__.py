# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Network latency calibration package for AIPerf.

Probes the endpoint's TCP-handshake RTT throughout profiling, writes a
per-sample JSONL artifact, and delivers the mean RTT to the metric results
processor so it can emit ``network_adjusted_*`` metrics.
"""
