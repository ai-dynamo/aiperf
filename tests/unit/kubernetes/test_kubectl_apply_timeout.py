# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout coverage for KubectlClient.apply()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests.kubernetes.helpers.kubectl import KubectlClient


class _HangingProcess:
    """Subprocess test double whose communicate() never returns."""

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.waited = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def wait(self) -> None:
        self.waited = True


@pytest.mark.asyncio
async def test_apply_hanging_process_raises_timeout_error() -> None:
    """apply() must not hang forever when the kubectl subprocess never exits."""
    client = KubectlClient()
    process = _HangingProcess()

    with (
        patch(
            "tests.kubernetes.helpers.kubectl.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ),
        pytest.raises(TimeoutError),
    ):
        await asyncio.wait_for(
            client.apply("apiVersion: v1\nkind: Namespace", timeout=0.05),
            timeout=5,
        )

    assert process.killed
    assert process.waited
