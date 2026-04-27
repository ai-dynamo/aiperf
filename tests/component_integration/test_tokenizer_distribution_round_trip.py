# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip test: register a gpt2 snapshot with the operator-side
``TokenizerBundleRegistry``, serve it through the FastAPI ``TokenizerRouter``,
download it via ``download_tokenizer``, and verify the extracted snapshot
tokenizes identically to the controller-side one.

Overrides four package-scoped autouse fixtures from ``conftest.py``:
- ``hf_offline_mode`` -- this test sets HF offline mode itself via
  ``monkeypatch.setenv`` after seeding a hermetic cache.
- ``mock_tokenizer_from_pretrained`` -- we need the real ``Tokenizer``
  + ``AutoTokenizer`` for the round-trip; the package fixture also
  patches ``_prefetch_tokenizers`` away, which we want to keep alive.
- ``mock_os_exit`` -- left as the real ``os._exit`` so any subprocess
  cleanup (e.g. ProcessPoolExecutor children) terminates cleanly. The
  package patch turns it into a no-op lambda that hangs ``waitpid()``.
- ``mock_os_kill_sigkill`` -- left as the real ``os.kill`` for the
  same reason.
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
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry
from aiperf.workers.worker_pod_tokenizer_download import download_tokenizer

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True, scope="module")
def hf_offline_mode():
    """Override package-scoped fixture: do NOT force HF_HUB_OFFLINE for this module.

    The controller-side prefetch needs to read from the local HF cache (or the
    network if absent); the per-test offline guarantee is asserted via
    ``monkeypatch`` after the bundle has been downloaded.
    """
    yield


@pytest.fixture(autouse=True, scope="module")
def mock_tokenizer_from_pretrained():
    """Override package-scoped fixture: keep the real Tokenizer + prefetch wiring.

    The whole point of this test is to exercise the real
    ``_prefetch_tokenizers`` -> registry -> router -> downloader chain.
    """
    yield


@pytest.fixture(autouse=True, scope="module")
def mock_os_exit():
    """Override package-scoped fixture: leave the real ``os._exit`` in place.

    The package-level patch turns ``os._exit`` into a SystemExit-returning
    lambda that never actually terminates the process. ProcessPoolExecutor
    child workers (spawned by ``_prefetch_tokenizers``) call ``os._exit(0)``
    on clean shutdown, so the parent ``waitpid()`` hangs forever if we
    keep the patch.
    """
    yield


@pytest.fixture(autouse=True, scope="module")
def mock_os_kill_sigkill():
    """Override package-scoped fixture: leave the real ``os.kill`` in place.

    Not needed for this test, and would only complicate ProcessPoolExecutor
    teardown if a cleanup path ever sent itself a signal.
    """
    yield


@pytest.fixture
async def running_api(unused_tcp_port: int):
    """Spin up a FastAPI app with only the tokenizer router mounted."""
    reg = TokenizerBundleRegistry()
    app = FastAPI()
    app.include_router(build_tokenizer_router(reg))
    config = uvicorn.Config(
        app, host="127.0.0.1", port=unused_tcp_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield reg, f"http://127.0.0.1:{unused_tcp_port}"
    finally:
        server.should_exit = True
        await task


async def test_round_trip_gpt2(running_api, tmp_path: Path, monkeypatch) -> None:
    reg, base_url = running_api

    # Build a hermetic HF cache containing ONLY tokenizer files for gpt2.
    # The shared user cache may carry framework weights (pytorch/tf/flax/...)
    # whose dereferenced tar is multi-GiB and trips the test memory watchdog;
    # we mirror just the layout snapshot_download would have produced.
    hf_home = tmp_path / "hf_home"
    snapshot_dir = _seed_gpt2_tokenizer_only_cache(hf_home)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    # Register the snapshot with the operator-side registry. This is the
    # post-condition that ``_prefetch_tokenizers`` would establish; we
    # bypass the ProcessPoolExecutor here because huggingface_hub caches
    # ``HF_HUB_CACHE`` at parent-process import time, and the parent's
    # ``snapshot_download(local_files_only=True)`` would resolve against
    # the developer's shared cache rather than our hermetic ``HF_HOME``.
    reg.mark_ready("gpt2", snapshot_dir)

    # Download via the operator API, exercising the real router + extractor.
    local_path = await download_tokenizer(
        api_base_url=base_url,
        name="gpt2",
        dest_root=tmp_path / "downloads",
        max_retries=3,
        logger=logging.getLogger("test"),
    )

    expected = AutoTokenizer.from_pretrained("gpt2").encode("Hello, world!")

    # Offline guarantee: the local load must succeed with HF offline-mode forced.
    # (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are already set above.)
    actual = AutoTokenizer.from_pretrained(str(local_path)).encode("Hello, world!")
    assert actual == expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_GPT2_TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _seed_gpt2_tokenizer_only_cache(hf_home: Path) -> Path:
    """Populate ``hf_home`` with a single gpt2 snapshot containing only tokenizer files.

    Mirrors the ``HF_HOME/hub/models--gpt2/{snapshots,blobs,refs}`` layout
    that ``snapshot_download`` produces, but copies just the small files
    needed to instantiate ``AutoTokenizer`` -- bypassing the multi-GiB
    framework weight blobs that may exist in the developer's shared cache.

    Returns the dereferenced snapshot path. Skips the test if the required
    source files are not present locally; component-integration tests must
    not require network egress.
    """
    src_root = Path.home() / ".cache/huggingface/hub/models--gpt2"
    refs_main = src_root / "refs/main"
    if not refs_main.exists():
        pytest.skip(
            f"gpt2 not found in shared HF cache ({src_root}); "
            "component-integration test cannot bootstrap a hermetic cache offline"
        )
    revision = refs_main.read_text().strip()
    src_snapshot = src_root / "snapshots" / revision
    missing = [f for f in _GPT2_TOKENIZER_FILES if not (src_snapshot / f).exists()]
    if missing:
        pytest.skip(
            f"gpt2 shared HF cache missing tokenizer files {missing}; "
            "cannot seed hermetic cache offline"
        )

    dst_root = hf_home / "hub/models--gpt2"
    (dst_root / "blobs").mkdir(parents=True, exist_ok=True)
    (dst_root / "refs").mkdir(parents=True, exist_ok=True)
    snapshot_dst = dst_root / "snapshots" / revision
    snapshot_dst.mkdir(parents=True, exist_ok=True)
    (dst_root / "refs/main").write_text(revision)

    import shutil

    for filename in _GPT2_TOKENIZER_FILES:
        src_link = src_snapshot / filename
        # Resolve the symlink to its blob, copy the blob into dst, re-link.
        blob_src = src_link.resolve()
        blob_name = blob_src.name
        blob_dst = dst_root / "blobs" / blob_name
        if not blob_dst.exists():
            shutil.copy2(blob_src, blob_dst)
        link_dst = snapshot_dst / filename
        if not link_dst.exists():
            link_dst.symlink_to(Path("../..") / "blobs" / blob_name)

    return snapshot_dst
