# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical MCPMark Verified harness with Rust-owned model inference.

This adapter deliberately reuses MCPMark's evaluator, task/state managers,
``MCPMarkAgent`` loop, pinned MCP servers, task setup, and ``verify.py``
programs.  It replaces only LiteLLM's model-call function with the shared
:class:`~aiperf.accuracy.model_broker.ModelCallBroker`; no Python inference
client or model-server URL exists here.

The implementation is source-grounded in ``eval-sys/mcpmark`` commit
``cd45b7f57923b9b3985467f5139927575f83141c``:

* ``pipeline.py:153-182`` constructs and runs :class:`MCPEvaluator`;
* ``src/evaluator.py:169-264`` owns setup, agent execution, verification, and
  cleanup, while ``src/evaluator.py:266-389`` owns resume and reporting;
* ``src/agents/mcpmark_agent.py:768-1099`` is the canonical LiteLLM/MCP tool
  loop and its generation controls;
* ``src/agents/mcpmark_agent.py:1102-1243`` selects the exact MCP servers;
* ``src/base/task_manager.py:132-245`` owns task selection and makes verifier
  return codes authoritative;
  and
* ``src/mcp_services/filesystem/filesystem_state_manager.py:72-174`` shows the
  canonical isolated-environment lifecycle used by the local proof.

MCPMark Verified became the default task set at that commit.  The dataset
namespace therefore pins both the exact repository commit and the canonical
``standard`` or ``easy`` suite. Hidden verifier inputs remain inside MCPMark.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override

from aiperf.accuracy.agentic import (
    AgenticEpisode,
    AgenticEpisodeResult,
    AgenticEvent,
    AgenticHarness,
    AgenticModelResult,
    EventQueue,
    require_finite_number,
    require_identifier,
    require_positive_int,
)
from aiperf.accuracy.model_broker import ModelCallBroker, RustInferenceError

# MCPMark imports LiteLLM at module import time even though the adapter replaces
# its only model-call entry point. Keep LiteLLM on its bundled immutable model
# metadata so importing the canonical scaffold cannot perform an ambient fetch.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402  # environment must be set before this import
from src.evaluator import MCPEvaluator  # noqa: E402
from src.factory import MCPServiceFactory  # noqa: E402
from src.model_config import ModelConfig  # noqa: E402
from src.results_reporter import TaskResult  # noqa: E402

_LOG = logging.getLogger("aiperf.accuracy.mcpmark")
_MCPMARK_VERSION = "0.0.1"
_MCPMARK_COMMIT = "cd45b7f57923b9b3985467f5139927575f83141c"
_MCPMARK_REPOSITORY = "https://github.com/eval-sys/mcpmark"
_MCPMARK_SOURCE_SHA256 = (
    "55bc1d0e43043101d4eed5b76d97c2efb14c3415e9a4c7e7b74cdc8f81fb21f2"
)
_RUST_API_KEY_SENTINEL = "aiperf-rust-owned-inference-no-http"
_CANONICAL_MAX_TURNS = 100
_CANONICAL_MAX_TOKENS = 32_768
_CANONICAL_COMPACTION_DISABLED = 999_999_999
_CANONICAL_TIMEOUT_SECONDS = 3_600

# These are provenance-only mirrors of the exact server constructors in pinned
# ``src/agents/mcpmark_agent.py:1102-1243``. The source-tree digest above makes
# any upstream change fail closed before these identities can become stale.
_CANONICAL_MCP_SERVER_IDENTITIES: dict[str, dict[str, Any]] = {
    "notion": {
        "transport": "stdio",
        "command": "npx",
        "artifact": "@notionhq/notion-mcp-server@1.9.1",
    },
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "artifact": "@modelcontextprotocol/server-filesystem@2025.12.18",
    },
    "playwright": {
        "transport": "stdio",
        "command": "npx",
        "artifact": "@playwright/mcp@0.0.68",
    },
    "playwright_webarena": {
        "transport": "stdio",
        "command": "npx",
        "artifact": "@playwright/mcp@0.0.68",
    },
    "postgres": {
        "transport": "stdio",
        "command": "pipx",
        "artifact": "postgres-mcp==0.3.0",
    },
    "insforge": {
        "transport": "stdio",
        "command": "npx",
        "artifact": "@insforge/mcp@dev",
    },
    "github": {
        "transport": "stdio",
        "command": "docker",
        "artifact": "ghcr.io/github/github-mcp-server:v0.15.0",
    },
    "supabase": {
        "transport": "http",
        "command": "external-supabase-cli",
        "artifact": "local-supabase-cli:/mcp",
    },
}


class _CallbackMessage:
    """Small LiteLLM-message facade consumed by MCPMark's canonical loop."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = copy.deepcopy(payload)
        self.role = str(payload.get("role") or "assistant")
        self.content = payload.get("content")
        self.reasoning_content = payload.get("reasoning_content")
        self.function_call = payload.get("function_call")
        self.tool_calls = [
            SimpleNamespace(
                id=str(tool_call.get("id") or ""),
                type=str(tool_call.get("type") or "function"),
                function=SimpleNamespace(
                    name=str((tool_call.get("function") or {}).get("name") or ""),
                    arguments=str(
                        (tool_call.get("function") or {}).get("arguments") or "{}"
                    ),
                ),
            )
            for tool_call in payload.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]

    def model_dump(self) -> dict[str, Any]:
        """Return the lossless assistant message MCPMark appends to history."""
        return copy.deepcopy(self._payload)


class _CallbackResponse:
    """LiteLLM-response facade backed exclusively by one Rust terminal result."""

    def __init__(self, result: AgenticModelResult, model_name: str) -> None:
        payload = copy.deepcopy(result.assistant_message or {})
        payload.setdefault("role", "assistant")
        if "content" not in payload:
            payload["content"] = result.response
        if result.reasoning is not None and "reasoning_content" not in payload:
            payload["reasoning_content"] = result.reasoning
        message = _CallbackMessage(payload)
        prompt_tokens = result.prompt_tokens or 0
        completion_tokens = result.completion_tokens or 0
        self.model = model_name
        self.id = result.response_id
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            input_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        finish_reason = result.finish_reason
        if finish_reason is None and message.tool_calls:
            finish_reason = "tool_calls"
        self.choices = [
            SimpleNamespace(message=message, finish_reason=finish_reason or "stop")
        ]


class MCPMarkHarness(AgenticHarness):
    """Run one exact MCPMark Verified service/suite through its native loop."""

    def __init__(
        self,
        *,
        dataset_name: str,
        service: str,
        task_suite: str,
        model_name: str,
        config: dict[str, Any],
        tasks: list[Any],
        environment_sha256: str | None,
    ) -> None:
        self._dataset_name = dataset_name
        self._service = service
        self._task_suite = task_suite
        self._model_name = model_name
        self._output_root = (
            Path(config["output_dir"]) / f"mcpmark-{os.getpid()}-{uuid.uuid4().hex}"
        )
        self._primary_reward = "pass"
        self._events = EventQueue()
        self._broker = ModelCallBroker(self._events)
        self._episodes: list[AgenticEpisode] = []
        self._task_filter_by_episode: dict[str, str] = {}
        self._task_by_episode: dict[str, Any] = {}
        dataset_revision = _dataset_revision(tasks, environment_sha256)
        for index, task in enumerate(tasks):
            task_filter = f"{task.category_id}/{task.task_id}"
            digest = hashlib.sha256(
                f"{_MCPMARK_COMMIT}\0{service}\0{task_suite}\0{task_filter}".encode()
            ).hexdigest()[:20]
            episode_id = f"mcpmark:{index:08d}:{digest}"
            episode = AgenticEpisode(
                episode_id=episode_id,
                task=task_filter,
                source=dataset_name,
            )
            self._episodes.append(episode)
            self._task_filter_by_episode[episode_id] = task_filter
            self._task_by_episode[episode_id] = task
        self._episode_by_id = {
            episode.episode_id: episode for episode in self._episodes
        }
        self._active: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, AgenticEpisodeResult] = {}
        self._closed = False
        self._identity = {
            "harness": "mcpmark-verified",
            "harness_version": f"mcpmark-{_MCPMARK_VERSION}@{_MCPMARK_COMMIT}",
            "harness_source_sha256": _mcpmark_source_digest(),
            "dataset": {
                "provider": "MCPMark Verified canonical task registry",
                "benchmark": dataset_name,
                "repository": _MCPMARK_REPOSITORY,
                "revision": dataset_revision,
                "evaluation_splits": [task_suite],
            },
            "agent": "MCPMarkAgent canonical LiteLLM/MCP loop",
            "agent_version": f"mcpmark@{_MCPMARK_COMMIT}+aiperf-rust-callback-1",
            "canonical_agent_config": {
                "agent_name": "mcpmark",
                "task_suite": task_suite,
                "mcp_service": service,
                "mcp_server": copy.deepcopy(_CANONICAL_MCP_SERVER_IDENTITIES[service]),
                "max_turns": _CANONICAL_MAX_TURNS,
                "max_tokens": _CANONICAL_MAX_TOKENS,
                "temperature": 1.0,
                "compaction_token": _CANONICAL_COMPACTION_DISABLED,
                "enable_summarize": False,
                "parser": "openai_tool_calls",
                "reasoning_effort": "default",
                "timeout_seconds": _CANONICAL_TIMEOUT_SECONDS,
            },
            "environment": service,
            "verifier": "MCPMark Verified task-local verify.py",
            "episode_count": len(self._episodes),
            "primary_reward": self._primary_reward,
        }

    @classmethod
    async def create(
        cls, dataset: str, model_name: str, authored_config: Any
    ) -> MCPMarkHarness:
        """Freeze MCPMark's exact registry selection without starting a task."""
        _require_mcpmark_environment()
        dataset_name, service, task_suite = _parse_dataset(dataset)
        config = _validate_config(authored_config, service)
        task_manager = MCPServiceFactory.create_task_manager(
            service, task_suite=task_suite
        )
        tasks = _select_tasks(task_manager, config["task_names"])
        if config["max_episodes"] is not None:
            tasks = tasks[: config["max_episodes"]]
        if not tasks:
            raise ValueError("MCPMark selection produced zero episodes")
        environment_sha256 = await asyncio.to_thread(
            _prepare_environment_revision, service, tasks
        )
        return cls(
            dataset_name=dataset_name,
            service=service,
            task_suite=task_suite,
            model_name=require_identifier(model_name, "model"),
            config=config,
            tasks=tasks,
            environment_sha256=environment_sha256,
        )

    @property
    @override
    def identity(self) -> dict[str, Any]:
        """Return exact MCPMark commit, suite, agent, and verifier provenance."""
        return copy.deepcopy(self._identity)

    @property
    @override
    def episodes(self) -> list[AgenticEpisode]:
        """Return selected MCPMark tasks in its canonical registry order."""
        return list(self._episodes)

    @override
    async def start_episodes(self, episode_ids: list[str]) -> None:
        """Start one canonical task; MCPMark's process globals require serial use."""
        self._ensure_open()
        if len(episode_ids) != 1:
            raise ValueError("MCPMark Verified requires one sequential episode start")
        episode_id = episode_ids[0]
        if episode_id not in self._episode_by_id:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        if episode_id in self._active or episode_id in self._results:
            raise ValueError(f"episode {episode_id!r} was already started")
        if self._active:
            raise RuntimeError("MCPMark Verified already has an active episode")
        self._active[episode_id] = asyncio.create_task(
            self._run_episode(episode_id), name=f"aiperf-mcpmark-{episode_id}"
        )

    @override
    async def poll_events(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        """Return canonical model calls and terminal verifier results."""
        self._ensure_open()
        return await self._events.poll(limit, wait_ms)

    @override
    async def submit_model_results(self, items: list[AgenticModelResult]) -> None:
        """Resume MCPMark's LiteLLM loop with Rust-produced terminal results."""
        self._ensure_open()
        if not items:
            raise ValueError("submit_model_results.items must not be empty")
        call_ids: set[str] = set()
        for item in items:
            if item.call_id in call_ids:
                raise ValueError(
                    f"duplicate submit_model_results call_id {item.call_id!r}"
                )
            call_ids.add(item.call_id)
            self._broker.submit(item)
        await asyncio.sleep(0)

    @override
    async def cancel_episodes(self, episode_ids: list[str]) -> None:
        """Cancel selected task wrappers and unblock their Rust callbacks."""
        self._ensure_open()
        tasks = []
        for episode_id in episode_ids:
            task = self._active.get(episode_id)
            if task is None:
                raise KeyError(f"episode {episode_id!r} is not active")
            self._broker.fail_episode(
                episode_id, RuntimeError("episode cancelled by Rust scheduler")
            )
            task.cancel()
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @override
    async def finish(self) -> list[AgenticEpisodeResult]:
        """Require one terminal canonical result for every selected task."""
        self._ensure_open()
        if self._active:
            raise RuntimeError(
                "finish_agentic called with active MCPMark episodes: "
                + ", ".join(sorted(self._active))
            )
        missing = [
            episode.episode_id
            for episode in self._episodes
            if episode.episode_id not in self._results
        ]
        if missing:
            raise RuntimeError(
                "finish_agentic called before every MCPMark episode was started: "
                + ", ".join(missing)
            )
        return [self._results[episode.episode_id] for episode in self._episodes]

    @override
    async def close(self) -> None:
        """Cancel active wrappers and close the process-local inference broker."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._active.items())
        for episode_id, task in tasks:
            self._broker.fail_episode(
                episode_id, RuntimeError("MCPMark harness closed")
            )
            task.cancel()
        if tasks:
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        self._broker.close()

    async def _run_episode(self, episode_id: str) -> None:
        episode = self._episode_by_id[episode_id]
        started = time.monotonic()
        thread_result = _start_daemon_thread(
            lambda: self._execute_episode(episode_id),
            name=f"aiperf-mcpmark-env-{episode_id}",
        )
        try:
            artifact_path, task_result = await asyncio.shield(thread_result)
            result = _convert_task_result(
                episode,
                task_result,
                duration_seconds=time.monotonic() - started,
                model_calls=self._broker.model_call_count(episode_id),
                artifact_path=artifact_path,
            )
        except asyncio.CancelledError:
            self._broker.fail_episode(
                episode_id, RuntimeError("episode cancelled by Rust scheduler")
            )
            try:
                # A Python thread cannot be killed safely. Do not release this
                # process-global MCPMark/LiteLLM adapter until canonical
                # verification and environment cleanup have actually drained;
                # Rust may terminate the supervised worker for force-cancel.
                await asyncio.shield(thread_result)
            except Exception:
                _LOG.exception(
                    "MCPMark environment failed while draining cancellation: %s",
                    episode_id,
                )
            result = AgenticEpisodeResult(
                episode_id=episode_id,
                task=episode.task,
                outcome="cancelled",
                rewards={},
                primary_reward=None,
                duration_seconds=time.monotonic() - started,
                model_calls=self._broker.model_call_count(episode_id),
                error_kind="CancelledError",
                error_message="episode cancelled by Rust scheduler",
            )
        except Exception as error:
            result = AgenticEpisodeResult(
                episode_id=episode_id,
                task=episode.task,
                outcome="infrastructure_error",
                rewards={},
                primary_reward=None,
                duration_seconds=time.monotonic() - started,
                model_calls=self._broker.model_call_count(episode_id),
                error_kind=type(error).__name__,
                error_message=str(error),
            )
        self._active.pop(episode_id, None)
        if episode_id in self._results:
            raise RuntimeError(f"episode {episode_id!r} produced duplicate results")
        self._results[episode_id] = result
        await self._events.put(AgenticEvent.completed(result))

    def _execute_episode(self, episode_id: str) -> tuple[str, TaskResult]:
        task_filter = self._task_filter_by_episode[episode_id]
        exp_name = episode_id.replace(":", "-")
        with (
            _temporary_environment(
                _model_api_key_variable(self._model_name), _RUST_API_KEY_SENTINEL
            ),
            _rust_litellm_calls(
                broker=self._broker,
                episode_id=episode_id,
                model_name=self._model_name,
            ),
        ):
            evaluator = MCPEvaluator(
                mcp_service=self._service,
                model=self._model_name,
                timeout=_CANONICAL_TIMEOUT_SECONDS,
                exp_name=exp_name,
                output_dir=self._output_root,
                reasoning_effort="default",
                agent_name="mcpmark",
                task_suite=self._task_suite,
                compaction_token=_CANONICAL_COMPACTION_DISABLED,
            )
            report = evaluator.run_evaluation(task_filter)
            if len(report.task_results) != 1:
                raise RuntimeError(
                    f"MCPMark returned {len(report.task_results)} results for "
                    f"single task {task_filter!r}"
                )
            task_result = report.task_results[0]
            if task_result.task_name != self._task_by_episode[episode_id].name:
                raise RuntimeError(
                    "MCPMark returned a different task than the frozen episode: "
                    f"{task_result.task_name!r}"
                )
            artifact_path = evaluator._get_task_output_dir(  # noqa: SLF001
                self._task_by_episode[episode_id]
            )
        return str(artifact_path), task_result

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MCPMark harness is closed")


async def create_mcpmark_harness(
    dataset: str, model_name: str, config: Any
) -> MCPMarkHarness:
    """Create the worker-selected canonical MCPMark Verified provider."""
    return await MCPMarkHarness.create(dataset, model_name, config)


@contextmanager
def _rust_litellm_calls(
    *, broker: ModelCallBroker, episode_id: str, model_name: str
) -> Iterator[None]:
    """Replace only MCPMark's model backend for one serial canonical episode."""
    original = litellm.acompletion

    async def acompletion(**kwargs: Any) -> _CallbackResponse:
        messages = kwargs.get("messages") or []
        if not isinstance(messages, list):
            raise TypeError("MCPMark LiteLLM messages must be an array")
        authored_messages = [copy.deepcopy(message) for message in messages]
        max_tokens = kwargs.get("max_tokens", _CANONICAL_MAX_TOKENS)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("MCPMark max_tokens must be an integer")
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        stop = kwargs.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        extra_body = _wire_extra_body(kwargs)
        result = broker.call_sync(
            episode_id=episode_id,
            model=model_name,
            prompt=_last_prompt(authored_messages),
            messages=authored_messages,
            generation={
                "max_tokens": max_tokens,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "stop": list(stop),
            },
            tools=copy.deepcopy(kwargs.get("tools") or []),
            tool_choice=copy.deepcopy(kwargs.get("tool_choice")),
            response_format=copy.deepcopy(kwargs.get("response_format")),
            extra_body=extra_body,
        )
        if result.status != "completed":
            detail = result.error_message or "Rust inference did not complete"
            raise RustInferenceError(
                f"AIPERF_RUST_INFERENCE:{result.error_kind}:{detail}"
            )
        return _CallbackResponse(result, model_name)

    litellm.acompletion = acompletion
    try:
        yield
    finally:
        litellm.acompletion = original


def _wire_extra_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    reserved = {
        "model",
        "messages",
        "api_key",
        "base_url",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
    }
    result = {}
    for key, value in kwargs.items():
        if key in reserved or value is None:
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        result[key] = copy.deepcopy(value)
    return result


def _last_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _parse_dataset(dataset: str) -> tuple[str, str, str]:
    authored = require_identifier(dataset, "dataset")
    name, separator, revision = authored.partition("@")
    if separator and revision != _MCPMARK_COMMIT:
        raise ValueError(
            f"MCPMark dataset revision must be {_MCPMARK_COMMIT!r}, got {revision!r}"
        )
    parts = name.lower().split("/")
    if len(parts) not in {2, 3} or parts[0] != "mcpmark":
        raise ValueError(
            "MCPMark dataset must be mcpmark/<service>[/standard|easy]"
            f"@{_MCPMARK_COMMIT}"
        )
    service = require_identifier(parts[1], "MCPMark service")
    task_suite = parts[2] if len(parts) == 3 else "standard"
    if task_suite not in {"standard", "easy"}:
        raise ValueError("MCPMark task suite must be 'standard' or 'easy'")
    supported = set(MCPServiceFactory.get_supported_mcp_services())
    if service not in supported:
        raise ValueError(
            f"unknown MCPMark service {service!r}; available: "
            + ", ".join(sorted(supported))
        )
    dataset_name = f"mcpmark/{service}"
    if task_suite != "standard":
        dataset_name += f"/{task_suite}"
    dataset_name += f"@{_MCPMARK_COMMIT}"
    return dataset_name, service, task_suite


def _validate_config(authored: Any, service: str) -> dict[str, Any]:
    if not isinstance(authored, dict):
        raise TypeError("MCPMark config must be an object")
    task_concurrency = require_positive_int(
        authored.get("task_concurrency"), "task_concurrency"
    )
    if task_concurrency != 1:
        raise ValueError("MCPMark process globals require task_concurrency=1")
    environment = require_identifier(authored.get("environment"), "environment")
    if environment != service:
        raise ValueError(
            f"MCPMark dataset service {service!r} requires "
            f"--agentic-environment {service}"
        )
    task_names = authored.get("task_names")
    if task_names is not None and not (
        isinstance(task_names, list)
        and all(isinstance(task, str) and task.strip() for task in task_names)
    ):
        raise TypeError("MCPMark task_names must be an array of non-empty strings")
    max_episodes = authored.get("max_episodes")
    if max_episodes is not None:
        max_episodes = require_positive_int(max_episodes, "max_episodes")
    max_turns = authored.get("max_turns")
    if max_turns is not None and max_turns != _CANONICAL_MAX_TURNS:
        raise ValueError(
            "MCPMark owns its canonical max-turn limit of 100; omit --agentic-max-turns"
        )
    output_dir = require_identifier(authored.get("output_dir"), "output_dir")
    primary_reward = authored.get("primary_reward")
    if primary_reward not in {None, "pass"}:
        raise ValueError("MCPMark primary_reward must be 'pass' or null")
    return {
        "task_names": None
        if task_names is None
        else [task.strip() for task in task_names],
        "max_episodes": max_episodes,
        "output_dir": output_dir,
    }


def _select_tasks(task_manager: Any, task_names: list[str] | None) -> list[Any]:
    all_tasks = task_manager.discover_all_tasks()
    if task_names is None:
        return list(all_tasks)
    selected_names = {
        task.name
        for authored in task_names
        for task in task_manager.filter_tasks(authored)
    }
    return [task for task in all_tasks if task.name in selected_names]


def _convert_task_result(
    episode: AgenticEpisode,
    result: TaskResult,
    *,
    duration_seconds: float,
    model_calls: int,
    artifact_path: str,
) -> AgenticEpisodeResult:
    infrastructure_error = _infrastructure_error(result, model_calls)
    if infrastructure_error is not None:
        kind, message = infrastructure_error
        return AgenticEpisodeResult(
            episode_id=episode.episode_id,
            task=episode.task,
            outcome="infrastructure_error",
            rewards={},
            primary_reward=None,
            duration_seconds=duration_seconds,
            model_calls=model_calls,
            error_kind=kind,
            error_message=message,
            artifact_path=artifact_path,
        )
    reward = require_finite_number(1.0 if result.success else 0.0, "pass")
    return AgenticEpisodeResult(
        episode_id=episode.episode_id,
        task=episode.task,
        outcome="completed",
        rewards={"pass": reward},
        primary_reward="pass",
        duration_seconds=duration_seconds,
        model_calls=model_calls,
        artifact_path=artifact_path,
    )


def _infrastructure_error(
    result: TaskResult, model_calls: int
) -> tuple[str, str] | None:
    agent_error = str(result.error_message or "")
    if "AIPERF_RUST_INFERENCE:" in agent_error:
        return "RustInferenceError", agent_error
    if agent_error == "State Duplication Error":
        return "MCPMarkStateSetupError", agent_error
    if result.verification_output is None and result.verification_error:
        return "MCPMarkVerifierError", str(result.verification_error)
    if model_calls == 0 and agent_error:
        return "MCPMarkAgentInfrastructureError", agent_error
    return None


def _dataset_revision(tasks: list[Any], environment_sha256: str | None) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        for path in (
            task.task_instruction_path,
            task.task_instruction_path.parent / "meta.json",
            task.task_verification_path,
        ):
            relative = path.relative_to(_mcpmark_root()).as_posix().encode()
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    revision = f"git:{_MCPMARK_COMMIT}+selection-sha256:{digest.hexdigest()}"
    if environment_sha256 is not None:
        revision += f"+environment-sha256:{environment_sha256}"
    return revision


def _prepare_environment_revision(service: str, tasks: list[Any]) -> str | None:
    """Prepare and hash MCPMark's concrete local state before measurement."""
    if service != "filesystem":
        return None
    state_manager = MCPServiceFactory.create_state_manager(service)
    category_roots: dict[str, Path] = {}
    for task in tasks:
        category = str(task.category_id)
        if category in category_roots:
            continue
        # This is MCPMark's own category-to-archive preparation path from
        # filesystem_state_manager.py:151-185. Calling it during load keeps
        # downloads outside the measured run while leaving setup/backup,
        # MCP execution, verification, and cleanup canonical.
        state_manager._set_dynamic_test_root(task)  # noqa: SLF001
        root = Path(state_manager.test_root).resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"MCPMark filesystem environment is not a directory: {root}"
            )
        category_roots[category] = root
    digest = hashlib.sha256()
    for category, root in sorted(category_roots.items()):
        _update_digest_field(digest, category.encode())
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix().encode()
            if path.is_symlink():
                kind = b"symlink"
                payload = os.readlink(path).encode()
                modified_seconds = 0
            elif path.is_file():
                kind = b"file"
                payload = None
                modified_seconds = path.stat().st_mtime_ns // 1_000_000_000
            elif path.is_dir():
                continue
            else:
                raise RuntimeError(f"unsupported MCPMark environment entry: {path}")
            _update_digest_field(digest, relative)
            _update_digest_field(digest, kind)
            digest.update(modified_seconds.to_bytes(8, "big", signed=True))
            if payload is not None:
                _update_digest_field(digest, payload)
            else:
                size = path.stat().st_size
                digest.update(size.to_bytes(8, "big"))
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
    return digest.hexdigest()


def _update_digest_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _mcpmark_root() -> Path:
    import src.evaluator as evaluator_module

    return Path(evaluator_module.__file__).resolve().parents[1]


def _mcpmark_source_digest() -> str:
    digest = hashlib.sha256()
    source_root = _mcpmark_root() / "src"
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_mcpmark_environment() -> None:
    actual_version = importlib.metadata.version("MCPMark")
    if actual_version != _MCPMARK_VERSION:
        raise RuntimeError(
            f"MCPMark evaluator has version {actual_version!r}; "
            f"expected {_MCPMARK_VERSION!r}"
        )
    actual_digest = _mcpmark_source_digest()
    if actual_digest != _MCPMARK_SOURCE_SHA256:
        raise RuntimeError(
            "MCPMark evaluator source does not match pinned commit "
            f"{_MCPMARK_COMMIT}: sha256={actual_digest}"
        )


def _model_api_key_variable(model_name: str) -> str:
    """Return MCPMark's canonical credential selector without reading a secret."""
    model_info = ModelConfig.MODEL_CONFIGS.get(model_name)
    if model_info is None:
        return "OPENAI_API_KEY"
    api_key_variable = model_info.get("api_key_var")
    if not isinstance(api_key_variable, str) or not api_key_variable:
        raise RuntimeError(
            f"MCPMark model {model_name!r} has no valid api_key_var mapping"
        )
    return api_key_variable


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _start_daemon_thread(
    function: Callable[[], Any], *, name: str
) -> asyncio.Future[Any]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def run() -> None:
        try:
            value = function()
        except BaseException as error:
            try:
                loop.call_soon_threadsafe(_finish_thread_error, future, error)
            except RuntimeError:
                _LOG.exception("MCPMark thread outlived its worker event loop")
        else:
            try:
                loop.call_soon_threadsafe(_finish_thread_value, future, value)
            except RuntimeError:
                _LOG.exception("MCPMark thread outlived its worker event loop")

    threading.Thread(target=run, name=name, daemon=True).start()
    return future


def _finish_thread_value(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _finish_thread_error(future: asyncio.Future[Any], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)
