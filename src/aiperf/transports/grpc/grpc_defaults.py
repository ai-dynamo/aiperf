# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default gRPC channel options optimized for benchmarking."""

from __future__ import annotations

from typing import Any

DEFAULT_CHANNEL_OPTIONS: list[tuple[str, Any]] = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.max_send_message_length", 256 * 1024 * 1024),  # 256MB
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_pings_without_data", 0),
]
