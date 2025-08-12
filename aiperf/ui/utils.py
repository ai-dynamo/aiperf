# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def format_duration(seconds: float | None, none_str: str = "--") -> str:
    """Format duration in seconds to human-readable format."""
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


def format_bytes(bytes: int | None, none_str: str = "--") -> str:
    """Format bytes to human-readable format."""
    if bytes is None:
        return none_str
    if bytes < 1000:
        return f"{bytes} B"

    _suffixes = ["KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    for i, suffix in enumerate(_suffixes):
        factor = 1024 ** (i + 1)
        if bytes / factor < 100:
            return f"{bytes / factor:.1f} {suffix}"
        if bytes / factor < 1000:
            return f"{bytes / factor:.0f} {suffix}"

    raise ValueError(f"Bytes value is too large to format: {bytes}")
