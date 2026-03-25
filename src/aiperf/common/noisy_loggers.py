# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for clamping noisy third-party loggers.

kr8s uses httpx internally, and httpx emits an INFO log line for every request.
Those request logs add a lot of noise to AIPerf's Kubernetes workflows.
"""

from __future__ import annotations

import logging

_NOISY_HTTP_LOGGERS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
}


def suppress_noisy_http_loggers() -> None:
    """Raise noisy HTTP client loggers to WARNING unless already stricter."""
    for logger_name, minimum_level in _NOISY_HTTP_LOGGERS.items():
        logger = logging.getLogger(logger_name)
        if logger.level == logging.NOTSET or logger.level < minimum_level:
            logger.setLevel(minimum_level)
