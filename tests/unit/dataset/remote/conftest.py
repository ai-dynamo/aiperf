# tests/unit/dataset/remote/conftest.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from aiohttp import web

from aiperf.common.environment import Environment


@pytest.fixture(autouse=True)
def _remote_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "remote-cache"
    monkeypatch.setattr(Environment.DATASET, "REMOTE_CACHE_DIR", root)
    monkeypatch.setattr(Environment.DATASET, "REMOTE_OFFLINE", False)
    # tests/unit/conftest.py forces HF offline for the session; these tests exercise online paths.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


@pytest.fixture
async def http_server() -> AsyncIterator[Callable[[web.Application], Awaitable[str]]]:
    runners: list[web.AppRunner] = []

    async def start(app: web.Application) -> str:
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        runners.append(runner)
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"http://127.0.0.1:{port}"

    yield start
    for runner in runners:
        await runner.cleanup()
