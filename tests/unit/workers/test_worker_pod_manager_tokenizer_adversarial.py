# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the WGM tokenizer-distribution flow.

These tests reproduce real production bugs that the existing unit tests
missed and that only surfaced after deploys to DGX:

1. ``_publish_group_message`` filtered to ``ServiceType.WORKER`` only,
   so the sibling-container ``RecordProcessor`` never received
   ``GroupTokenizerReady`` even though it had registered as a peer.
   Symptom: WGM logs "prefetch complete" with the bundle on disk, but
   the RP container's parser is stuck forever in
   ``self._tokenizer_ready.wait()``.

2. WGM's ``_prefetch_tokenizers`` passed the full
   ``runtime.dataset_api_base_url`` (which ends in ``/api/dataset``) to
   ``download_tokenizer``, producing a ``/api/dataset/api/tokenizer/...``
   URL that 404'd. The RP/parser path used to do the strip in
   ``resolve_tokenizer_load_target``; after the architecture refactor
   that helper went away and the strip got lost.

3. Silent hangs: ``_prefetch_tokenizers`` had no log statements between
   "task scheduled" and "transient error", so a hang in the middle was
   invisible from kubectl logs alone.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady
from aiperf.plugin.enums import ServiceType


@pytest.mark.asyncio
async def test_publish_group_tokenizer_ready_reaches_record_processor() -> None:
    """The bug that broke Qwen2.5-7B smoke after deploy: the RP sibling
    container never received GroupTokenizerReady.

    Workers do not tokenize — only RecordProcessor does. The publisher
    must target ``ServiceType.RECORD_PROCESSOR`` and skip ``ServiceType.WORKER``.
    """
    from aiperf.workers import worker_pod_manager as wpm

    sent_to: list[bytes] = []

    mgr = MagicMock()
    mgr._pod_peer_identities = {
        "worker-0": b"worker-0-id",
        "worker-1": b"worker-1-id",
        "record-processor-0": b"rp-0-id",
    }
    mgr._pod_peer_types = {
        "worker-0": str(ServiceType.WORKER),
        "worker-1": str(ServiceType.WORKER),
        "record-processor-0": str(ServiceType.RECORD_PROCESSOR),
    }

    async def _capture(identity, message):
        sent_to.append(identity)

    mgr.pod_lifecycle_router = MagicMock()
    mgr.pod_lifecycle_router.send_to = AsyncMock(side_effect=_capture)

    msg = GroupTokenizerReady(
        service_id="wgm-0",
        bundles={"gpt2": "/aiperf/aiperf_tokenizers/run-1/gpt2"},
    )

    await wpm.WorkerGroupManagerBase._publish_group_message(mgr, msg)

    # RP sibling container must receive — it's the actual consumer.
    assert b"rp-0-id" in sent_to, (
        "RecordProcessor peer was not sent GroupTokenizerReady; "
        "_publish_group_message is filtering it out. RP will block "
        "forever on _tokenizer_ready.wait()."
    )
    # Workers do NOT tokenize and must NOT receive the message.
    assert b"worker-0-id" not in sent_to
    assert b"worker-1-id" not in sent_to


@pytest.mark.asyncio
async def test_publish_group_tokenizer_ready_skips_unknown_peer_types() -> None:
    """Belt-and-suspenders: peers of an unrelated type should not get the
    tokenizer-ready broadcast. Catches accidental over-broadening of the
    RP-only filter.
    """
    from aiperf.workers import worker_pod_manager as wpm

    sent_to: list[bytes] = []

    mgr = MagicMock()
    mgr._pod_peer_identities = {
        "rp-0": b"rp-0-id",
        "wgm-self": b"wgm-id",
        "worker-0": b"worker-0-id",
    }
    mgr._pod_peer_types = {
        "rp-0": str(ServiceType.RECORD_PROCESSOR),
        "wgm-self": str(ServiceType.WORKER_GROUP_MANAGER),
        "worker-0": str(ServiceType.WORKER),
    }

    async def _capture(identity, message):
        sent_to.append(identity)

    mgr.pod_lifecycle_router = MagicMock()
    mgr.pod_lifecycle_router.send_to = AsyncMock(side_effect=_capture)

    msg = GroupTokenizerReady(service_id="wgm-0", bundles={})
    await wpm.WorkerGroupManagerBase._publish_group_message(mgr, msg)

    assert sent_to == [b"rp-0-id"], (
        f"only RECORD_PROCESSOR should receive GroupTokenizerReady; got {sent_to}"
    )


@pytest.mark.asyncio
async def test_prefetch_tokenizers_strips_api_dataset_suffix(monkeypatch) -> None:
    """The other deploy regression: WGM passed the full
    runtime.dataset_api_base_url (ending in /api/dataset) to
    download_tokenizer, producing a /api/dataset/api/tokenizer/{name}/bundle
    URL that 404'd. download_tokenizer must receive only the host:port
    base, no /api/dataset suffix.
    """
    from aiperf.workers import worker_pod_manager as wpm

    captured_api_base: list[str] = []

    async def fake_download_tokenizer(*, api_base_url, name, dest_root, **_):
        captured_api_base.append(api_base_url)
        return dest_root / name

    monkeypatch.setattr(wpm, "download_tokenizer", fake_download_tokenizer)

    mgr = MagicMock()
    mgr.run.cfg.runtime.dataset_api_base_url = (
        "http://aiperf-smoke-controller-0-0:9090/api/dataset"
    )
    mgr.run.cfg.runtime.service_run_type = "kubernetes"
    mgr.service_id = "wgm-0"
    mgr._unique_tokenizer_names = MagicMock(return_value=["gpt2"])
    mgr._tokenizer_dest_root = MagicMock(
        return_value=__import__("pathlib").Path("/tmp/test-prefetch")
    )
    mgr._publish_group_message = AsyncMock()
    mgr.info = MagicMock()

    await wpm.WorkerGroupManagerBase._prefetch_tokenizers(mgr)

    assert captured_api_base == ["http://aiperf-smoke-controller-0-0:9090"], (
        f"download_tokenizer received {captured_api_base[0]!r} as api_base_url. "
        f"Should have stripped the /api/dataset suffix; otherwise the actual "
        f"URL becomes /api/dataset/api/tokenizer/{{name}}/bundle and 404s."
    )


@pytest.mark.asyncio
async def test_prefetch_tokenizers_logs_lifecycle_transitions(monkeypatch) -> None:
    """When a hang happens between scheduled-task and HTTP request, the only
    way to diagnose from kubectl logs is having info logs at every state
    transition. Asserts the function logs at least:
    - task starting
    - tokenizer names resolved
    - prefetch complete (or failure on exception path)
    """
    from aiperf.workers import worker_pod_manager as wpm

    async def fake_download_tokenizer(*, name, dest_root, **_):
        return dest_root / name

    monkeypatch.setattr(wpm, "download_tokenizer", fake_download_tokenizer)

    info_calls: list[str] = []
    mgr = MagicMock()
    mgr.run.cfg.runtime.dataset_api_base_url = "http://x:9090/api/dataset"
    mgr.run.cfg.runtime.service_run_type = "kubernetes"
    mgr.service_id = "wgm-0"
    mgr._unique_tokenizer_names = MagicMock(return_value=["gpt2"])
    mgr._tokenizer_dest_root = MagicMock(
        return_value=__import__("pathlib").Path("/tmp/test-prefetch-logs")
    )
    mgr._publish_group_message = AsyncMock()
    mgr.info = MagicMock(side_effect=lambda msg, *a, **k: info_calls.append(str(msg)))

    await wpm.WorkerGroupManagerBase._prefetch_tokenizers(mgr)

    joined = " | ".join(info_calls)
    assert "starting" in joined.lower(), f"missing 'starting' info log in: {info_calls}"
    assert "gpt2" in joined or "Tokenizers to fetch" in joined, (
        f"missing tokenizer-names info log in: {info_calls}"
    )
    assert "complete" in joined.lower() or "ready" in joined.lower(), (
        f"missing 'complete' info log in: {info_calls}"
    )
