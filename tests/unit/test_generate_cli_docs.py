# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for recursive CLI documentation extraction."""

from aiperf.cli import app
from tools.generate_cli_docs import (
    _resolve_lazy_commands,
    extract_commands,
    extract_params,
)


def test_defaulted_parent_and_lazy_child_are_both_documented() -> None:
    """A command family may expose both a default action and nested commands."""
    _resolve_lazy_commands(app)

    commands = dict(extract_commands(app))

    assert "dynamo" in commands
    assert "dynamo trace-report" in commands
    params = extract_params(app, "dynamo trace-report")
    assert any(
        param.long_opts == "--limit" for group in params.values() for param in group
    )
