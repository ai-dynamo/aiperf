# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process round-trip: prove the router serves bundles from a shared
HF_HOME populated by a different Python process — mirrors the production
topology where the api container's interpreter is separate from the
controller-plane container that prewarms.

Spawns the FastAPI app via ``multiprocessing.get_context("spawn")`` so the
child cannot inherit the parent's already-imported ``transformers`` modules
or any module-level globals.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
import time
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve_router(hf_home: str, port: int) -> None:
    """Subprocess entry: serve the tokenizer router with HF_HOME injected.

    Module-level so spawn can pickle it.
    """
    import os
    import sys

    os.environ["HF_HOME"] = hf_home
    print(
        f"[child] HF_HOME={os.environ.get('HF_HOME')} "
        f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
        f"hub_dir_exists={os.path.isdir(os.path.join(hf_home, 'hub'))}",
        file=sys.stderr,
        flush=True,
    )

    import uvicorn
    from fastapi import FastAPI

    from aiperf.api.routers.tokenizer import build_tokenizer_router

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait_for_tcp(host: str, port: int, deadline_s: float = 30.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(
        f"router subprocess did not bind {host}:{port} within {deadline_s}s"
    )


async def test_router_serves_from_subprocess_populated_hf_home(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    hf_home.mkdir()

    # Parent process primes the hermetic cache by loading gpt2 with HF_HOME
    # pointed at the shared dir. The child process will read from the same
    # on-disk tree without sharing any Python state.
    import os

    os.environ["HF_HOME"] = str(hf_home)
    AutoTokenizer.from_pretrained("gpt2")

    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_serve_router, args=(str(hf_home), port), daemon=True)
    proc.start()
    try:
        await asyncio.to_thread(_wait_for_tcp, "127.0.0.1", port)

        local_path = await download_tokenizer(
            api_base_url=f"http://127.0.0.1:{port}",
            name="gpt2",
            dest_root=tmp_path / "dl",
            max_retries=3,
            logger=logging.getLogger("test"),
        )
        # Round-trip: load the downloaded snapshot offline, verify token IDs.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
        expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")
        assert actual == expected
    finally:
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
