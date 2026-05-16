# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration round-trip: prewarm a hermetic HF_HOME, serve the
bundle through the FastAPI ``TokenizerRouter``, download via
``download_tokenizer``, and verify ``AutoTokenizer.from_pretrained(local_path)``
produces token IDs identical to the warmer's tokenizer.

This mirrors the production path (api container's ``_prewarm_tokenizers``
populating a shared ``HF_HOME`` emptyDir, then serving via the router from
that cache).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer

from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
async def running_api(unused_tcp_port: int, tmp_path: Path, monkeypatch):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    # Prewarm the hermetic cache (mirrors api_service._prewarm_tokenizers).
    AutoTokenizer.from_pretrained("gpt2")

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    config = uvicorn.Config(
        app, host="127.0.0.1", port=unused_tcp_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}"
    finally:
        server.should_exit = True
        await task


async def test_round_trip_gpt2(running_api, tmp_path: Path, monkeypatch) -> None:
    base_url = running_api
    expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")

    local_path = await download_tokenizer(
        api_base_url=base_url,
        name="gpt2",
        dest_root=tmp_path / "dl",
        max_retries=3,
        logger=logging.getLogger("test"),
    )

    # Force HF offline for the local-path load -- proves the bundle is
    # self-contained and no Hub call leaks in.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
    assert actual == expected
