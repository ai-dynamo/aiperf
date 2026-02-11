# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlparse


def normalize_endpoint_display(url: str) -> str:
    """Normalize endpoint URL for display by removing scheme and trimming /metrics suffix.

    Args:
        url: The full URL to normalize (e.g., "https://host:9400/api/metrics")

    Returns:
        Normalized display string with netloc and trimmed path (e.g., "host:9400/api")
    """
    parsed = urlparse(url)
    path = parsed.path.removesuffix("/metrics")

    display = parsed.netloc
    if path:
        display += path

    return display
