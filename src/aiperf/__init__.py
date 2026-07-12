# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf - AI Benchmarking Tool."""


def __getattr__(name: str):
    """Resolve package metadata only for consumers that actually request it."""
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("aiperf")
        except PackageNotFoundError:
            value = "unknown"
    elif name == "__commit_sha__":
        try:
            from aiperf._build_info import COMMIT_SHA as value
        except ImportError:
            value = "unknown"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
