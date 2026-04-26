# tests/integration/test_aiperfsweep_e2e.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test: real kind cluster, mock-server endpoint, 4-cell × 2-trial sweep.

Asserts:
- AIPerfSweep status.phase reaches Succeeded (or PartiallyFailed when any cell fails).
- aggregate JSON exists under <ns>/<sweep>/<epoch>/aggregate/.
- manifest.json lists all 8 child runs with correct labels and statuses.
"""

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
async def test_aiperfsweep_4x2_completes_with_aggregate():
    """Submit a 4-variation × 2-trial AIPerfSweep against a mock server.

    The full wiring requires kind-cluster + mock-server fixtures from
    tests/integration/conftest.py. Implement in-cluster when those are
    plumbed; until then this test is a placeholder.
    """
    pytest.skip("kind+mock-server fixture wiring is environment-specific")
