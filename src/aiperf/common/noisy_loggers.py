# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suppress noisy HTTP loggers emitted by Kubernetes client libraries.

kubernetes_asyncio uses aiohttp internally and emits access-log-style
lines for every request. httpx (a legacy kr8s dependency during the
migration) emits similar lines. Both are silenced to WARNING.
"""

import logging

_NOISY_LOGGERS = (
    "aiohttp.access",
    "aiohttp.client",
    "kubernetes_asyncio.client.rest",
    "httpx",  # legacy — removed with kr8s
)


def suppress_noisy_http_loggers() -> None:
    """Raise noisy HTTP-client loggers to WARNING."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
