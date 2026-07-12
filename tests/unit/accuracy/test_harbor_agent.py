# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Harbor callback tests for Rust-owned agent inference."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("harbor", reason="requires agentic-accuracy worker lock")

from aiperf.accuracy.agentic import AgenticModelResult, EventQueue
from aiperf.accuracy.harbor_agent import (
    AIPerfCallbackLLM,
    AIPerfTerminus2,
    ModelCallBroker,
    RustInferenceError,
    register_broker,
    unregister_broker,
)


@pytest.mark.asyncio
async def test_callback_llm_round_trips_messages_and_usage_without_http() -> None:
    events = EventQueue()
    broker = ModelCallBroker(events)
    llm = AIPerfCallbackLLM(
        broker=broker,
        episode_id="episode-1",
        model_name="fixture-model",
        context_limit=32768,
        output_limit=1024,
        temperature=0.0,
    )
    pending = asyncio.create_task(
        llm.call(
            "inspect the repository",
            message_history=[{"role": "system", "content": "use JSON commands"}],
        )
    )
    event = (await events.poll(1, 100))[0]
    assert event.model_call is not None
    call = event.model_call
    assert call.messages == [
        {"role": "system", "content": "use JSON commands"},
        {"role": "user", "content": "inspect the repository"},
    ]
    assert call.generation == {
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": [],
    }
    broker.submit(
        AgenticModelResult(
            episode_id="episode-1",
            call_id=call.call_id,
            status="completed",
            response='{"commands": [], "task_complete": true}',
            reasoning="done",
            prompt_tokens=17,
            completion_tokens=8,
            cached_tokens=3,
            response_id="response-1",
            finish_reason="stop",
            error_kind=None,
            error_message=None,
        )
    )
    response = await pending
    assert response.content == '{"commands": [], "task_complete": true}'
    assert response.reasoning_content == "done"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 17
    assert response.usage.completion_tokens == 8
    assert response.usage.cache_tokens == 3
    assert response.response_id == "response-1"


@pytest.mark.asyncio
async def test_callback_llm_propagates_transport_failure_as_infrastructure_error() -> (
    None
):
    events = EventQueue()
    broker = ModelCallBroker(events)
    llm = AIPerfCallbackLLM(
        broker=broker,
        episode_id="episode-1",
        model_name="fixture-model",
        context_limit=32768,
        output_limit=1024,
        temperature=None,
    )
    pending = asyncio.create_task(llm.call("continue"))
    event = (await events.poll(1, 100))[0]
    assert event.model_call is not None
    broker.submit(
        AgenticModelResult(
            episode_id="episode-1",
            call_id=event.model_call.call_id,
            status="failed",
            response="partial",
            reasoning=None,
            prompt_tokens=None,
            completion_tokens=None,
            cached_tokens=None,
            response_id=None,
            finish_reason=None,
            error_kind="transport_error",
            error_message="connection reset",
        )
    )
    with pytest.raises(RustInferenceError, match="connection reset"):
        await pending


def test_terminus_subclass_installs_callback_backend(tmp_path) -> None:
    broker = ModelCallBroker(EventQueue())
    broker_id = register_broker(broker)
    try:
        agent = AIPerfTerminus2(
            logs_dir=tmp_path,
            model_name="fixture-model",
            aiperf_broker_id=broker_id,
            aiperf_episode_id="episode-1",
            aiperf_context_limit=32768,
            aiperf_output_limit=1024,
            record_terminal_session=False,
            enable_summarize=False,
        )
        assert isinstance(agent._llm, AIPerfCallbackLLM)
        assert agent.name() == "aiperf-terminus-2"
        assert agent.version() == "1.0.0+terminus-2.0.0"
    finally:
        unregister_broker(broker_id)
