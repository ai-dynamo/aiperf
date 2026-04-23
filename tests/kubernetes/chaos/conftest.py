# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for chaos scenarios.

``chaos_injector`` is the single entry point; build new scenarios by
composing the injector's methods with the package-level
``operator_ready``, ``kubectl`` fixtures already provided by
``tests/kubernetes/conftest.py``.
"""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.helpers.kubectl import KubectlClient


@pytest.fixture
def chaos_injector(kubectl: KubectlClient) -> ChaosInjector:
    """Provide a ``ChaosInjector`` bound to the package-scoped cluster."""
    return ChaosInjector(kubectl=kubectl)
