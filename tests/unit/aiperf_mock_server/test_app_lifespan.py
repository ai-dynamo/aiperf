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
