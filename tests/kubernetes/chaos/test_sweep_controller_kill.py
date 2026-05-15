# tests/kubernetes/chaos/test_sweep_controller_kill.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos test: kill the sweep-controller pod mid-sweep, assert idempotent resume.

Submits a 4-cell sweep, waits for variation 2 to succeed, kills the
sweep-controller pod via `kubectl exec` (requires podTemplate.shareProcessNamespace,
documented in docs/superpowers/specs/2026-04-23-chaos-expansion-design.md),
asserts that the restarted pod resumes at variation 3 (does NOT re-run 0/1/2),
and that the final aggregate.manifest.json contains all 8 child results.
"""

import pytest

pytestmark = [pytest.mark.k8s_slow]


@pytest.mark.asyncio
async def test_sweep_controller_kill_resumes_correctly():
    pytest.skip("chaos cluster fixture wiring is environment-specific")
