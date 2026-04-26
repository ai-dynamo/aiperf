# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signature test verifying profile CLI takes v1 UserConfig+ServiceConfig."""

import inspect

from aiperf.cli_commands.profile import profile


def test_profile_cli_takes_user_and_service_config() -> None:
    sig = inspect.signature(profile)
    annots = {p.name: p.annotation for p in sig.parameters.values()}
    assert "user_config" in annots, (
        f"profile() must take user_config, got: {list(annots)}"
    )
    assert "service_config" in annots, (
        f"profile() must take service_config, got: {list(annots)}"
    )
