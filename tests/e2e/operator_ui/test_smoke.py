# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary smoke test — deleted once per-page tests land."""

import httpx
import pytest


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="session")
async def test_live_operator_app_starts(live_operator_app):
    """The session fixture binds a real uvicorn and /healthz returns 200."""
    async with httpx.AsyncClient(trust_env=False) as client:
        resp = await client.get(f"{live_operator_app.base_url}/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.e2e
@pytest.mark.asyncio(loop_scope="session")
async def test_live_operator_app_serves_index(live_operator_app):
    """Root URL returns the SPA index.html."""
    async with httpx.AsyncClient(trust_env=False) as client:
        resp = await client.get(f"{live_operator_app.base_url}/")
    assert resp.status_code == 200
    assert "<div id=\"app\"></div>" in resp.text
