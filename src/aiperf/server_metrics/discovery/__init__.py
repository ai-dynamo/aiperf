# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-metrics endpoint discovery implementations.

This package root is intentionally dependency-free. The Kubernetes gate lives
here rather than in ``discovery.kubernetes`` so callers can decide whether to
discover without importing ``kubernetes_asyncio``, which costs ~130 ms and 700+
modules in every process that merely asks the question.
"""

import os

__all__ = ["is_running_in_kubernetes"]


def is_running_in_kubernetes() -> bool:
    """Return whether Kubernetes injected its service environment."""
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
