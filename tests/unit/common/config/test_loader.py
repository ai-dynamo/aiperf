# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the current YAML-only configuration loader API."""

from __future__ import annotations

import importlib


def test_obsolete_config_loader_api_is_absent() -> None:
    """Configuration loading is exposed only through ``aiperf.config.loader``."""
    try:
        importlib.import_module("aiperf.common.config.loader")
    except ModuleNotFoundError:
        return
    raise AssertionError(
        "aiperf.common.config.loader unexpectedly exists; use aiperf.config.loader"
    )
