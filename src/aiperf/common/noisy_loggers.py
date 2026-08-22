# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suppress noisy HTTP client loggers."""

import logging

_NOISY_LOGGERS = (
    "aiohttp.access",
    "aiohttp.client",
)


def suppress_noisy_http_loggers() -> None:
    """Raise noisy HTTP-client loggers to WARNING."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
