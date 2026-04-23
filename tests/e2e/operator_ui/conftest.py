# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the operator web UI e2e suite.

Runs a real uvicorn server bound to 127.0.0.1:<random> once per session,
hosting the real ``create_app()`` FastAPI instance with a session-scoped
``results_dir``. Per-test fixtures mutate the contents of that dir and
monkeypatch the k8s helpers — no respawn.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI

from aiperf.operator.results_server import create_app


def _free_port() -> int:
    """Bind to port 0 and return the kernel-assigned port.

    There's a TOCTOU race between binding here and re-binding in uvicorn,
    but in practice it's safe on localhost and avoids uvicorn's lack of
    a "port 0 then tell me what you got" API in older versions.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LiveApp:
    base_url: str
    app: FastAPI
    results_dir: Path


@asynccontextmanager
async def _running_server(
    app: FastAPI, port: int
) -> AsyncIterator[None]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for startup
    for _ in range(200):  # 10s max at 50ms
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        server.should_exit = True
        await task
        raise RuntimeError("uvicorn failed to start within 10s")
    try:
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def live_operator_app(tmp_path_factory) -> AsyncIterator[LiveApp]:
    """Real uvicorn + real ``create_app()`` bound to a random port.

    The ``results_dir`` is session-scoped; per-test fixtures rewrite its
    contents. The jobs router's ``ApiClient`` stays ``None`` (tests that
    need it monkeypatch the six ``aiperf.kubernetes.client`` helpers).
    """
    results_dir = tmp_path_factory.mktemp("e2e_results")
    app = create_app(results_dir=results_dir)
    port = _free_port()
    async with _running_server(app, port):
        yield LiveApp(
            base_url=f"http://127.0.0.1:{port}",
            app=app,
            results_dir=results_dir,
        )
