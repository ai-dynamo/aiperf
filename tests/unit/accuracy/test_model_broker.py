# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-neutral model broker tests."""

from __future__ import annotations

import asyncio

import pytest

from aiperf.accuracy.agentic import AgenticModelResult, EventQueue
from aiperf.accuracy.model_broker import ModelCallBroker


@pytest.mark.asyncio
async def test_sync_harness_call_round_trips_on_worker_event_loop() -> None:
    events = EventQueue()
    broker = ModelCallBroker(events)
    pending = asyncio.create_task(
        asyncio.to_thread(
            broker.call_sync,
            episode_id="browser-episode",
            model="target-model",
            prompt="inspect the page",
            messages=[{"role": "user", "content": "inspect the page"}],
            generation={
                "max_tokens": 256,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": [],
            },
        )
    )
    event = (await events.poll(1, 1_000))[0]
    assert event.model_call is not None
    assert event.model_call.model == "target-model"
    assert event.model_call.turn_index == 0
    broker.submit(
        AgenticModelResult(
            episode_id="browser-episode",
            call_id=event.model_call.call_id,
            status="completed",
            response="<action>click('1')</action>",
            reasoning=None,
            prompt_tokens=20,
            completion_tokens=7,
            cached_tokens=2,
            response_id="response-1",
            finish_reason="stop",
            error_kind=None,
            error_message=None,
        )
    )
    await asyncio.sleep(0)
    result = await pending
    assert result.response == "<action>click('1')</action>"
    assert broker.model_call_count("browser-episode") == 1


@pytest.mark.asyncio
async def test_sync_harness_call_cannot_deadlock_worker_loop() -> None:
    broker = ModelCallBroker(EventQueue())
    with pytest.raises(RuntimeError, match="cannot block"):
        broker.call_sync(
            episode_id="episode",
            model=None,
            prompt="prompt",
            messages=[],
            generation={},
        )
