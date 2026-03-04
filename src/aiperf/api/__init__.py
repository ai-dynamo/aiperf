# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf API module.

Provides a unified HTTP + WebSocket API for:
- Prometheus metrics scraping (/metrics)
- JSON metrics API (/api/metrics)
- Benchmark config and progress (/api/config, /api/progress)
- Real-time ZMQ message streaming via WebSocket (/ws)
"""
