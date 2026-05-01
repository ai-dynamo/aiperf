# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process round-trip: prove the router serves bundles from a shared
HF_HOME populated by a different Python process - mirrors the production
topology where the api container's interpreter is separate from the
controller-plane container that prewarms.

The prewarm and the router each run in fresh ``multiprocessing.get_context("spawn")``
children with HF_HOME injected at process-start (so ``huggingface_hub.constants``
resolves to the hermetic dir). The parent process never touches the cache dir
itself - this guarantees the router cannot accidentally rely on parent state.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
import time
from pathlib import Path

import pytest

from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _prewarm(hf_home: str) -> None:
    """Subprocess entry: download gpt2 into the hermetic HF_HOME.

    Module-level so spawn can pickle it. Sets HF_HOME and explicitly clears
    HF_HUB_OFFLINE so the prewarm can reach the network. The package-scope
    ``hf_offline_mode`` fixture sets HF_HUB_OFFLINE=1 in the parent env, which
    spawn inherits - we drop it here because prewarm requires Hub egress.
    """
    import os

    os.environ["HF_HOME"] = hf_home
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from transformers import AutoTokenizer as _AutoTokenizer

    _AutoTokenizer.from_pretrained("gpt2")


def _serve_router(hf_home: str, port: int) -> None:
    """Subprocess entry: serve the tokenizer router with HF_HOME injected.

    Module-level so spawn can pickle it. HF_HOME is set BEFORE huggingface_hub
    is imported - constants module captures the path at import time, so a
    post-import env mutation would silently use the wrong cache dir.
    """
    import os

    os.environ["HF_HOME"] = hf_home

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


async def test_router_serves_from_subprocess_populated_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    ctx = multiprocessing.get_context("spawn")

    # First spawn: prewarm the hermetic cache. Production parity: the
    # controller-plane container populates HF_HOME on disk before the api
    # container starts serving.
    prewarm = ctx.Process(target=_prewarm, args=(str(hf_home),))
    prewarm.start()
    await asyncio.to_thread(prewarm.join, 120.0)
    assert prewarm.exitcode == 0, f"prewarm failed (exitcode={prewarm.exitcode})"
    assert (hf_home / "hub" / "models--gpt2").is_dir(), (
        "prewarm did not populate hf_home"
    )

    # Second spawn: the router. Different Python interpreter, no shared
    # in-process state with the prewarm or the test process.
    port = _free_port()
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
        # Round-trip: load the downloaded snapshot offline, verify token IDs
        # match the warmer's output. HF_HOME points at the hermetic dir so the
        # parent's cache doesn't shadow the bundle under test.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer

        actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
        expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")
        assert actual == expected
    finally:
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
