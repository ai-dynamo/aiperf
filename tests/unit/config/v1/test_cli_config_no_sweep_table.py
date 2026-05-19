# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the --no-sweep-table flag on CLIConfig."""

from __future__ import annotations

from aiperf.config.flags import CLIConfig


def test_no_sweep_table_default_false() -> None:
    cfg = CLIConfig()
    assert cfg.no_sweep_table is False


def test_no_sweep_table_can_be_set() -> None:
    cfg = CLIConfig(no_sweep_table=True)
    assert cfg.no_sweep_table is True
