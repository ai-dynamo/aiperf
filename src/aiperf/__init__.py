# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf - AI Benchmarking Tool."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiperf")
except PackageNotFoundError:
    __version__ = "unknown"
