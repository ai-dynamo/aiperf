# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.aiperf_mock_server.app import app
from tests.aiperf_mock_server.control_state import control_state
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_mock_server_control_routes_mutate_state() -> None:
    control_state.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/start_profile")).status_code == 200
        assert (await client.post("/stop_profile")).status_code == 200
        assert (await client.post("/reset_prefix_cache")).status_code == 200
        assert (await client.post("/flush_cache")).status_code == 200
    assert control_state.profiler_starts == 1
    assert control_state.profiler_stops == 1
    assert control_state.reset_count == 2


@pytest.mark.asyncio
async def test_mock_server_flush_cache_is_sglang_compatible_alias() -> None:
    control_state.reset()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/flush_cache")).status_code == 200
        assert (await client.post("/flush_cache?timeout=30")).status_code == 200
    assert control_state.reset_count == 2
    assert control_state.events == ["reset", "reset"]
