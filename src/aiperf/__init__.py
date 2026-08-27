# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf - AI Benchmarking Tool."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiperf")
except PackageNotFoundError:
    __version__ = "unknown"

# _build_info is generated at wheel-build time by the CI pipeline. Source
# installs and dev checkouts won't have it — fall back to "unknown".
try:
    from aiperf._build_info import COMMIT_SHA as __commit_sha__
except ImportError:
    __commit_sha__ = "unknown"
