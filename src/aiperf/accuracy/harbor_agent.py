# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor Terminus-2 adapter whose LLM calls are fulfilled by Rust.

This module deliberately contains no HTTP client. Harbor imports
``AIPerfTerminus2`` as a custom agent, and the overridden ``_init_llm`` returns
an :class:`AIPerfCallbackLLM`. Each ``call`` publishes a model-call event to the
worker and awaits the terminal response that Rust obtained through AIPerf's
ordinary transport.

The adapter is source-grounded in Harbor 0.18.0 at commit
``4e256b94b43bb8acefd9714b81913fd8bcf1df5c``:

* ``src/harbor/llms/base.py:54-78`` defines the injectable ``BaseLLM`` contract;
* ``src/harbor/llms/chat.py:77-111`` supplies full message history to ``call``;
* ``src/harbor/agents/terminus_2/terminus_2.py:73-151`` constructs the backend;
* ``src/harbor/agents/terminus_2/terminus_2.py:1000-1158`` defines retry and
  length-error behavior; and
* ``src/harbor/agents/terminus_2/terminus_2.py:1249-1555`` owns canonical
  parsing, terminal execution, observations, and completion confirmation.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, override

# Importing Harbor's Terminus module imports LiteLLM even though this adapter
# replaces its backend. Force LiteLLM's bundled immutable model metadata so a
# worker import never performs an ambient GitHub fetch.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import (
    BaseLLM,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.models.metric import UsageInfo

from aiperf.accuracy.agentic import (
    AgenticEvent,
    AgenticModelCall,
    AgenticModelResult,
    EventQueue,
)


class RustInferenceError(RuntimeError):
    """A Rust-owned inference call failed before producing a usable response."""


class ModelCallBroker:
    """Correlate Harbor ``BaseLLM.call`` futures with worker protocol events."""

    def __init__(self, events: EventQueue) -> None:
        self._events = events
        self._pending: dict[str, tuple[str, asyncio.Future[AgenticModelResult]]] = {}
        self._turns: dict[str, int] = {}
        self._closed = False

    def model_call_count(self, episode_id: str) -> int:
        """Return the number of calls authored for one episode."""
        return self._turns.get(episode_id, 0)

    async def call(
        self,
        *,
        episode_id: str,
        model: str | None,
        prompt: str,
        messages: list[dict[str, Any]],
        generation: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: Any | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AgenticModelResult:
        """Publish one model request and wait until Rust submits its terminal result."""
        if self._closed:
            raise RuntimeError("agentic model-call broker is closed")
        turn_index = self._turns.get(episode_id, 0)
        self._turns[episode_id] = turn_index + 1
        call_id = f"{episode_id}:call:{turn_index:08d}"
        if call_id in self._pending:
            raise RuntimeError(f"duplicate agentic model call id {call_id!r}")
        future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = (episode_id, future)
        await self._events.put(
            AgenticEvent.call(
                AgenticModelCall(
                    episode_id=episode_id,
                    call_id=call_id,
                    turn_index=turn_index,
                    model=model,
                    prompt=prompt,
                    messages=messages,
                    generation=generation,
                    tools=list(tools or []),
                    tool_choice=tool_choice,
                    response_format=response_format,
                    extra_body=dict(extra_body or {}),
                )
            )
        )
        try:
            return await future
        finally:
            self._pending.pop(call_id, None)

    def submit(self, result: AgenticModelResult) -> None:
        """Resolve one outstanding callback from a Rust terminal result."""
        pending = self._pending.get(result.call_id)
        if pending is None:
            raise KeyError(f"unknown or already completed call_id {result.call_id!r}")
        episode_id, future = pending
        if result.episode_id != episode_id:
            raise ValueError(
                f"call {result.call_id!r} belongs to episode {episode_id!r}, "
                f"not {result.episode_id!r}"
            )
        if future.done():
            raise ValueError(f"call {result.call_id!r} was submitted more than once")
        future.set_result(result)

    def fail_episode(self, episode_id: str, error: BaseException) -> None:
        """Fail every outstanding model call for a cancelled episode."""
        for pending_episode, future in self._pending.values():
            if pending_episode == episode_id and not future.done():
                future.set_exception(error)

    def close(self) -> None:
        """Fail outstanding calls so no Harbor task remains blocked on shutdown."""
        self._closed = True
        for _, future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("agentic model-call broker closed"))


_BROKERS: dict[str, ModelCallBroker] = {}


def register_broker(broker: ModelCallBroker) -> str:
    """Register a process-local broker and return its opaque lookup id."""
    broker_id = uuid.uuid4().hex
    _BROKERS[broker_id] = broker
    return broker_id


def unregister_broker(broker_id: str) -> None:
    """Remove a broker after all Harbor trials have drained."""
    _BROKERS.pop(broker_id, None)


def _broker(broker_id: str) -> ModelCallBroker:
    try:
        return _BROKERS[broker_id]
    except KeyError as error:
        raise RuntimeError(f"unknown AIPerf Harbor broker {broker_id!r}") from error


class AIPerfCallbackLLM(BaseLLM):
    """Harbor LLM backend that delegates every inference call to Rust."""

    def __init__(
        self,
        *,
        broker: ModelCallBroker,
        episode_id: str,
        model_name: str,
        context_limit: int,
        output_limit: int,
        temperature: float | None,
    ) -> None:
        super().__init__()
        self._broker = broker
        self._episode_id = episode_id
        self._model_name = model_name
        self._context_limit = context_limit
        self._output_limit = output_limit
        self._temperature = temperature

    @override
    async def call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Emit one model-call event and adapt Rust's terminal response for Harbor."""
        history = kwargs.get("message_history") or []
        if not isinstance(history, list):
            raise TypeError("Harbor message_history must be a list")
        messages = [dict(message) for message in history]
        messages.append({"role": "user", "content": prompt})
        max_tokens = kwargs.get("max_tokens", self._output_limit)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("Harbor max_tokens must be an integer")
        temperature = kwargs.get("temperature", self._temperature)
        top_p = kwargs.get("top_p", 1.0)
        stop = kwargs.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        extra_body = kwargs.get("extra_body") or {}
        if not isinstance(extra_body, dict):
            raise TypeError("Harbor extra_body must be an object")
        result = await self._broker.call(
            episode_id=self._episode_id,
            model=self._model_name,
            prompt=prompt,
            messages=messages,
            generation={
                "max_tokens": max_tokens,
                "temperature": 0.0 if temperature is None else float(temperature),
                "top_p": float(top_p),
                "stop": list(stop),
            },
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
            extra_body=extra_body,
        )
        if result.status != "completed":
            message = result.error_message or "Rust inference did not complete"
            if result.error_kind == "context_length_exceeded":
                raise ContextLengthExceededError(message)
            raise RustInferenceError(f"{result.error_kind}: {message}")
        if result.finish_reason == "length":
            raise OutputLengthExceededError(
                "Rust inference reached the configured output limit",
                truncated_response=result.response,
            )
        usage = None
        if result.prompt_tokens is not None or result.completion_tokens is not None:
            usage = UsageInfo(
                prompt_tokens=result.prompt_tokens or 0,
                completion_tokens=result.completion_tokens or 0,
                cache_tokens=result.cached_tokens or 0,
                cost_usd=0.0,
            )
        return LLMResponse(
            content=result.response,
            reasoning_content=result.reasoning,
            model_name=self._model_name,
            usage=usage,
            response_id=result.response_id,
        )

    @override
    def get_model_context_limit(self) -> int:
        """Return the run's explicit context-window limit."""
        return self._context_limit

    @override
    def get_model_output_limit(self) -> int | None:
        """Return the run's explicit per-call output limit."""
        return self._output_limit


class AIPerfTerminus2(Terminus2):
    """Pinned Harbor Terminus-2 scaffold with Rust-owned model inference."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        aiperf_broker_id: str,
        aiperf_episode_id: str,
        aiperf_context_limit: int,
        aiperf_output_limit: int,
        **kwargs: Any,
    ) -> None:
        self._aiperf_broker = _broker(aiperf_broker_id)
        self._aiperf_episode_id = aiperf_episode_id
        self._aiperf_context_limit = aiperf_context_limit
        self._aiperf_output_limit = aiperf_output_limit
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @override
    def _init_llm(
        self,
        llm_backend: Any,
        model_name: str,
        temperature: float | None,
        collect_rollout_details: bool,
        llm_kwargs: dict[str, Any] | None,
        api_base: str | None,
        session_id: str | None,
        max_thinking_tokens: int | None,
        reasoning_effort: str | None,
        model_info: dict[str, Any] | None,
        use_responses_api: bool,
    ) -> BaseLLM:
        del (
            llm_backend,
            collect_rollout_details,
            llm_kwargs,
            api_base,
            session_id,
            max_thinking_tokens,
            reasoning_effort,
            model_info,
            use_responses_api,
        )
        return AIPerfCallbackLLM(
            broker=self._aiperf_broker,
            episode_id=self._aiperf_episode_id,
            model_name=model_name,
            context_limit=self._aiperf_context_limit,
            output_limit=self._aiperf_output_limit,
            temperature=temperature,
        )

    @staticmethod
    @override
    def name() -> str:
        """Return a distinct scaffold identity for reports and Harbor artifacts."""
        return "aiperf-terminus-2"

    @override
    def version(self) -> str | None:
        """Record both the adapter and inherited Terminus scaffold generation."""
        return "1.0.0+terminus-2.0.0"
