# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Conftest for plugin unit tests.

Overrides the session-scoped load_aiperf_modules fixture to prevent loading
all modules, which allows testing extensible_enums.py independently.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def load_aiperf_modules():
    """Override to skip module loading for plugin tests."""
    yield
