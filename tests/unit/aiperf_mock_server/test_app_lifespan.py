# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifespan-ordering tests for the AIPerf mock server.

These cover failure modes that the request-handler integration tests can't
exercise — specifically, what happens during shutdown when one of the
lifespan teardown steps raises.
"""

import pytest
from aiperf_mock_server.config import MockServerConfig
from fastapi import FastAPI


@pytest.mark.asyncio
async def test_lifespan_closes_recorder_when_scheduler_shutdown_raises(
    tmp_path, monkeypatch
) -> None:
    """If `shutdown_scheduler()` raises during teardown, the request recorder
    must still be closed so `<path>.summary.json` is written — otherwise the
    `--record-requests` user loses the artifact they enabled the mode for.

    Regression test for the prior single-`finally` ordering where any
    scheduler-shutdown exception silently skipped `recorder.close()`.
    """
    from aiperf_mock_server.app import lifespan

    rec_path = tmp_path / "rec.jsonl"
    summary_path = tmp_path / "rec.jsonl.summary.json"

    test_cfg = MockServerConfig(
        record_requests=str(rec_path),
        tokenizer="builtin",
        fast=True,
        dcgm_auto_load=False,
    )
    monkeypatch.setattr("aiperf_mock_server.app.server_config", test_cfg)

    async def fake_init_scheduler(_cfg) -> None:
        return None

    async def boom_shutdown_scheduler() -> None:
        raise RuntimeError("simulated scheduler shutdown failure")

    monkeypatch.setattr("aiperf_mock_server.app.init_scheduler", fake_init_scheduler)
    monkeypatch.setattr(
        "aiperf_mock_server.app.shutdown_scheduler", boom_shutdown_scheduler
    )

    assert not summary_path.exists()

    fastapi_app = FastAPI()
    with pytest.raises(RuntimeError, match="simulated scheduler shutdown failure"):
        async with lifespan(fastapi_app):
            pass

    assert summary_path.exists(), (
        "recorder.close() did not run after scheduler shutdown failed — "
        "summary.json was not written"
    )


@pytest.mark.asyncio
async def test_lifespan_closes_recorder_when_scheduler_init_raises(
    tmp_path, monkeypatch
) -> None:
    """If `init_scheduler()` raises during startup *after* `recorder.open()`
    has already run, the recorder must still be closed and the global
    handle unregistered.

    Without this, the `try: yield ... finally:` cleanup block is never
    entered (the exception propagates out of `lifespan` before `yield`),
    so the global recorder stays installed, the JSONL file handle leaks,
    and `<path>.summary.json` is never written.

    Symmetric to `test_lifespan_closes_recorder_when_scheduler_shutdown_raises`
    above; this one covers the startup side.
    """
    from aiperf_mock_server.app import lifespan
    from aiperf_mock_server.request_recorder import get_global_recorder

    rec_path = tmp_path / "rec.jsonl"
    summary_path = tmp_path / "rec.jsonl.summary.json"

    test_cfg = MockServerConfig(
        record_requests=str(rec_path),
        tokenizer="builtin",
        fast=True,
        dcgm_auto_load=False,
    )
    monkeypatch.setattr("aiperf_mock_server.app.server_config", test_cfg)

    async def boom_init_scheduler(_cfg) -> None:
        raise RuntimeError("simulated scheduler init failure")

    async def noop_shutdown_scheduler() -> None:
        return None

    monkeypatch.setattr("aiperf_mock_server.app.init_scheduler", boom_init_scheduler)
    monkeypatch.setattr(
        "aiperf_mock_server.app.shutdown_scheduler", noop_shutdown_scheduler
    )

    assert not summary_path.exists()

    fastapi_app = FastAPI()
    with pytest.raises(RuntimeError, match="simulated scheduler init failure"):
        async with lifespan(fastapi_app):
            pytest.fail("yield should not have been reached")  # pragma: no cover

    assert summary_path.exists(), (
        "recorder.close() did not run after init_scheduler failed — "
        "summary.json was not written"
    )
    assert get_global_recorder() is None, (
        "global recorder was left installed after init_scheduler failed"
    )
