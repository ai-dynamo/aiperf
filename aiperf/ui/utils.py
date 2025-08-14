# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


_time_suffixes = ["d", "h", "m"]
_time_factors = [60 * 60 * 24, 60 * 60, 60]


def format_elapsed_time(seconds: float | None, none_str: str = "--") -> str:
    """Format elapsed time in seconds to human-readable format."""
    if seconds is None:
        return none_str

    _logger.debug(f"format_elapsed_time: seconds: {seconds}")
    if seconds < 60:
        return f"{seconds:.1f}s"

    parts = []
    _logger.debug(
        f"format_elapsed_time: _time_suffixes: {_time_suffixes}, _time_factors: {_time_factors}"
    )
    for suffix, factor in zip(_time_suffixes, _time_factors, strict=True):
        _logger.debug(f"format_elapsed_time: seconds: {seconds}, factor: {factor}")
        if seconds >= factor:
            amount = int(seconds // factor)
            if amount > 0:
                parts.append(f"{amount}{suffix}")
            seconds -= amount * factor

    if seconds > 0:
        parts.append(f"{seconds:.0f}s")

    return " ".join(parts)


def format_eta(seconds: float | None, none_str: str = "--") -> str:
    """Format an ETA in seconds to human-readable format."""
    if seconds is None:
        return none_str

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds < 1:
            return f"{minutes}m"
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = minutes // 60
    minutes = minutes % 60

    if hours < 24:
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h {minutes}m"

    days = hours // 24
    hours = hours % 24

    if hours == 0:
        return f"{days}d"
    return f"{days}d {hours}h"


_byte_suffixes = ["KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
_byte_factors = [1024 ** (i + 1) for i in range(len(_byte_suffixes))]


def format_bytes(bytes: int | None, none_str: str = "--") -> str:
    """Format bytes to human-readable format."""
    if bytes is None:
        return none_str
    if bytes < 1000:
        return f"{bytes} B"

    for factor, suffix in zip(_byte_factors, _byte_suffixes, strict=True):
        if bytes / factor < 100:
            return f"{bytes / factor:.1f} {suffix}"
        if bytes / factor < 1000:
            return f"{bytes / factor:.0f} {suffix}"

    raise ValueError(f"Bytes value is too large to format: {bytes}")
