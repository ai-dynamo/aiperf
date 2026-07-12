# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful agentic-evaluation contract owned by the Python worker.

The static accuracy protocol returns a complete prompt and later accepts one
terminal model response. Agentic benchmarks instead alternate between model
inference and evaluator-owned environment work. This module defines that
stateful boundary without importing an agent harness or contacting an inference
server.

Rust remains the only inference owner. A harness implementation emits
``model_call`` events containing model-safe messages and generation controls;
Rust sends those calls through AIPerf's ordinary scheduler/transport pipeline
and returns terminal results through :meth:`AgenticHarness.submit_model_results`.
The harness owns task preparation, tool/environment execution, and verification.

The first concrete harness is Harbor 0.18.0. Its callback point is the abstract
``BaseLLM`` interface in ``harbor/llms/base.py`` and Terminus-2's injected
``self._llm`` usage in ``harbor/agents/terminus_2/terminus_2.py`` at upstream
commit ``4e256b94b43bb8acefd9714b81913fd8bcf1df5c``. AIPerf implements that
interface with a queue-backed callback; it does not proxy or perform HTTP in
Python.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

AgenticEventKind = Literal["model_call", "episode_completed"]
AgenticEpisodeOutcome = Literal["completed", "infrastructure_error", "cancelled"]
AgenticInferenceStatus = Literal["completed", "failed", "cancelled"]


def require_identifier(value: Any, field_name: str) -> str:
    """Return a non-empty opaque identifier or raise a protocol error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_non_negative_int(value: Any, field_name: str) -> int:
    """Return a non-negative integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def require_positive_int(value: Any, field_name: str) -> int:
    """Return a positive integer while rejecting booleans."""
    result = require_non_negative_int(value, field_name)
    if result == 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return result


def require_finite_number(value: Any, field_name: str) -> float:
    """Return a finite wire number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class AgenticEpisode:
    """Opaque task instance that Rust may admit into an agentic run."""

    episode_id: str
    task: str
    source: str

    def to_wire(self) -> dict[str, Any]:
        """Serialize the model-safe episode descriptor."""
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "source": self.source,
        }


@dataclass(frozen=True)
class AgenticModelCall:
    """One evaluator-authored inference call waiting for Rust dispatch."""

    episode_id: str
    call_id: str
    turn_index: int
    prompt: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any | None = None
    response_format: Any | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize only information that is safe to send to the model."""
        result: dict[str, Any] = {
            "episode_id": self.episode_id,
            "call_id": self.call_id,
            "turn_index": self.turn_index,
            "prompt": self.prompt,
            "messages": self.messages,
            "generation": self.generation,
            "tools": self.tools,
        }
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            result["response_format"] = self.response_format
        return result


@dataclass(frozen=True)
class AgenticEpisodeResult:
    """Canonical harness/verifier result for one completed task instance."""

    episode_id: str
    task: str
    outcome: AgenticEpisodeOutcome
    rewards: dict[str, float]
    primary_reward: str | None
    duration_seconds: float
    model_calls: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    error_kind: str | None = None
    error_message: str | None = None
    artifact_path: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize the complete task result without hidden verifier inputs."""
        result: dict[str, Any] = {
            "episode_id": self.episode_id,
            "task": self.task,
            "outcome": self.outcome,
            "rewards": self.rewards,
            "primary_reward": self.primary_reward,
            "duration_seconds": self.duration_seconds,
            "model_calls": self.model_calls,
        }
        for key, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("cached_tokens", self.cached_tokens),
            ("error_kind", self.error_kind),
            ("error_message", self.error_message),
            ("artifact_path", self.artifact_path),
        ):
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class AgenticModelResult:
    """Terminal Rust inference result delivered to a waiting harness call."""

    episode_id: str
    call_id: str
    status: AgenticInferenceStatus
    response: str
    reasoning: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    response_id: str | None
    finish_reason: str | None
    error_kind: str | None
    error_message: str | None

    @classmethod
    def from_wire(cls, value: Any) -> AgenticModelResult:
        """Validate one strict ``submit_model_results`` item."""
        if not isinstance(value, dict):
            raise TypeError("submit_model_results item must be an object")
        allowed = {
            "episode_id",
            "call_id",
            "status",
            "response",
            "reasoning",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "response_id",
            "finish_reason",
            "error_kind",
            "error_message",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "submit_model_results item has unknown field(s): " + ", ".join(unknown)
            )
        episode_id = require_identifier(value.get("episode_id"), "episode_id")
        call_id = require_identifier(value.get("call_id"), "call_id")
        status = value.get("status")
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError(
                f"model result {call_id!r} status must be completed|failed|cancelled"
            )
        response = value.get("response", "")
        if not isinstance(response, str):
            raise TypeError(f"model result {call_id!r} response must be a string")
        optional_strings: dict[str, str | None] = {}
        for name in (
            "reasoning",
            "response_id",
            "finish_reason",
            "error_kind",
            "error_message",
        ):
            authored = value.get(name)
            if authored is not None and not isinstance(authored, str):
                raise TypeError(
                    f"model result {call_id!r} {name} must be a string or null"
                )
            optional_strings[name] = authored
        optional_counts: dict[str, int | None] = {}
        for name in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            authored = value.get(name)
            optional_counts[name] = (
                None
                if authored is None
                else require_non_negative_int(
                    authored, f"model result {call_id!r} {name}"
                )
            )
        if status == "completed" and (
            optional_strings["error_kind"] is not None
            or optional_strings["error_message"] is not None
        ):
            raise ValueError(
                f"completed model result {call_id!r} must not contain an error"
            )
        if status != "completed" and optional_strings["error_kind"] is None:
            raise ValueError(
                f"non-completed model result {call_id!r} requires error_kind"
            )
        return cls(
            episode_id=episode_id,
            call_id=call_id,
            status=status,
            response=response,
            reasoning=optional_strings["reasoning"],
            prompt_tokens=optional_counts["prompt_tokens"],
            completion_tokens=optional_counts["completion_tokens"],
            cached_tokens=optional_counts["cached_tokens"],
            response_id=optional_strings["response_id"],
            finish_reason=optional_strings["finish_reason"],
            error_kind=optional_strings["error_kind"],
            error_message=optional_strings["error_message"],
        )


@dataclass(frozen=True)
class AgenticEvent:
    """One model-call or terminal-episode event emitted by a harness."""

    kind: AgenticEventKind
    model_call: AgenticModelCall | None = None
    episode_result: AgenticEpisodeResult | None = None

    @classmethod
    def call(cls, model_call: AgenticModelCall) -> AgenticEvent:
        """Build a model-call event."""
        return cls(kind="model_call", model_call=model_call)

    @classmethod
    def completed(cls, result: AgenticEpisodeResult) -> AgenticEvent:
        """Build a terminal episode event."""
        return cls(kind="episode_completed", episode_result=result)

    def to_wire(self) -> dict[str, Any]:
        """Serialize a tagged event with exactly one payload."""
        if self.kind == "model_call" and self.model_call is not None:
            return {"kind": self.kind, "call": self.model_call.to_wire()}
        if self.kind == "episode_completed" and self.episode_result is not None:
            return {"kind": self.kind, "result": self.episode_result.to_wire()}
        raise RuntimeError(f"invalid agentic event payload for {self.kind!r}")


class AgenticHarness(ABC):
    """Harness trait behind the worker's stateful agentic operations.

    A concrete implementation may use Harbor, a future browser harness, or a
    hermetic test fixture. It must never send an inference request itself.
    """

    @property
    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return immutable harness, dataset, agent, and verifier provenance."""

    @property
    @abstractmethod
    def episodes(self) -> list[AgenticEpisode]:
        """Return all selected episodes in canonical order."""

    @abstractmethod
    async def start_episodes(self, episode_ids: list[str]) -> None:
        """Begin evaluator-owned setup/execution for selected task instances."""

    @abstractmethod
    async def poll_events(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        """Return up to ``limit`` ready model-call or terminal events."""

    @abstractmethod
    async def submit_model_results(self, items: list[AgenticModelResult]) -> None:
        """Resume harness calls with terminal results produced by Rust."""

    @abstractmethod
    async def cancel_episodes(self, episode_ids: list[str]) -> None:
        """Cancel active episodes and release their environments."""

    @abstractmethod
    async def finish(self) -> list[AgenticEpisodeResult]:
        """Validate terminal state and return results in canonical order."""

    @abstractmethod
    async def close(self) -> None:
        """Cancel remaining work and release all harness resources."""


class EventQueue:
    """Small reusable event queue implementing bounded long polling."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgenticEvent] = asyncio.Queue()

    async def put(self, event: AgenticEvent) -> None:
        """Publish one harness event."""
        await self._queue.put(event)

    async def poll(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        """Return a non-empty-ready batch or an empty timeout result."""
        require_positive_int(limit, "poll_agentic.limit")
        require_non_negative_int(wait_ms, "poll_agentic.wait_ms")
        first: AgenticEvent | None = None
        if self._queue.empty() and wait_ms > 0:
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=wait_ms / 1000
                )
            except TimeoutError:
                return []
        elif not self._queue.empty():
            first = self._queue.get_nowait()
        if first is None:
            return []
        result = [first]
        while len(result) < limit and not self._queue.empty():
            result.append(self._queue.get_nowait())
        return result
