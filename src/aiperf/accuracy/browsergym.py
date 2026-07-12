# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned AgentLab/BrowserGym harness with exclusively Rust-owned inference.

This adapter calls the exact AgentLab 0.4.2 and BrowserGym 0.14.3 APIs rather
than reproducing browser-agent semantics:

* AgentLab ``src/agentlab/llm/base_api.py:5-34`` defines the injectable
  synchronous chat-model interface;
* ``src/agentlab/agents/generic_agent/generic_agent.py:29-158`` owns benchmark
  adaptation, prompt construction, truncation, retry parsing, and actions;
* ``src/agentlab/experiments/loop.py:307-580`` owns environment reset/step,
  trajectory artifacts, cleanup, and canonical ``summary_info.json`` rewards;
* BrowserGym ``browsergym/experiments/.../configs.py:93-294`` owns the benchmark
  task lists, seeds, action sets, and step limits; and
* its WebArena-Verified adapter
  ``browsergym/webarena_verified/.../evaluators.py:34-130`` invokes the canonical
  ``WebArenaVerifiedEvaluatorAPI`` over the captured trace.

The only replacement is ``BaseModelArgs.make_model``: it returns a callback
model that publishes the complete AgentLab-authored message list to Rust's
normal AIPerf pipeline.  This module contains no model HTTP client.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import importlib.metadata
import logging
import threading
import time
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar, override

import orjson
from agentlab.agents.generic_agent.agent_configs import (
    FLAGS_GPT_4o,
    FLAGS_GPT_4o_VISION,
)
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.experiments.loop import ExpArgs
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, Discussion
from agentlab.llm.tracking import TRACKER, LLMTracker
from bgym import DEFAULT_BENCHMARKS, Benchmark

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
from aiperf.accuracy.model_broker import (
    ModelCallBroker,
    RustInferenceError,
    broker_for_id,
    register_broker,
    unregister_broker,
)

_LOG = logging.getLogger(__name__)
_AGENTLAB_VERSION = "0.4.2"
_AGENTLAB_COMMIT = "367d4e8a9c2cd97eab4524f6898ac98010fc99a8"
_BROWSERGYM_VERSION = "0.14.3"
_BROWSERGYM_COMMIT = "0a785fbed075224ae81ca9c1fe924f66050696fe"
_DATASET_PREFIX = "browsergym/"
_MODEL_PROFILE = "gpt-4o-2024-05-13"
_DEFAULT_CONTEXT_WINDOW = 128_000
_THREAD_DRAIN_SECONDS = 30.0
_COMMON_CONFIG_FIELDS = {
    "task_names",
    "max_episodes",
    "task_concurrency",
    "environment",
    "output_dir",
    "max_turns",
    "max_tokens",
    "context_window",
    "parser",
    "enable_summarize",
    "primary_reward",
    "overwrite",
    "inference_gateway",
}

_T = TypeVar("_T")


@dataclass
class AIPerfAgentLabModelArgs(BaseModelArgs):
    """Picklable AgentLab model arguments resolving a process-local broker."""

    broker_id: str = ""
    episode_id: str = ""
    target_model: str = ""

    @override
    def make_model(self) -> AbstractChatModel:
        """Build the synchronous callback expected by AgentLab's agent loop."""
        return AIPerfAgentLabChatModel(
            broker=broker_for_id(self.broker_id),
            episode_id=self.episode_id,
            target_model=self.target_model,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )


class AIPerfAgentLabChatModel(AbstractChatModel):
    """AgentLab chat model whose only backend is a Rust protocol callback."""

    def __init__(
        self,
        *,
        broker: ModelCallBroker,
        episode_id: str,
        target_model: str,
        max_tokens: int | None,
        temperature: float,
    ) -> None:
        self._broker = broker
        self._episode_id = episode_id
        self._target_model = target_model
        self._max_tokens = max_tokens or 4_096
        self._temperature = temperature

    @override
    def __call__(
        self,
        messages: list[dict[str, Any]] | Discussion,
        n_samples: int = 1,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Block AgentLab's environment thread on one Rust inference result."""
        if isinstance(n_samples, bool) or n_samples != 1:
            raise ValueError("AIPerf AgentLab evaluation requires n_samples=1")
        wire_messages = _openai_messages(messages)
        result = self._broker.call_sync(
            episode_id=self._episode_id,
            model=self._target_model,
            prompt=_last_user_text(wire_messages),
            messages=wire_messages,
            generation={
                "max_tokens": self._max_tokens,
                "temperature": (
                    self._temperature if temperature is None else float(temperature)
                ),
                "top_p": 1.0,
                "stop": [],
            },
        )
        if result.status != "completed":
            message = result.error_message or "Rust inference did not complete"
            raise RustInferenceError(f"{result.error_kind}: {message}")
        if hasattr(TRACKER, "instance") and isinstance(TRACKER.instance, LLMTracker):
            TRACKER.instance(
                result.prompt_tokens or 0,
                result.completion_tokens or 0,
                0.0,
            )
        return AIMessage(result.response)

    @override
    def get_stats(self) -> dict[str, int]:
        """Match AgentLab's one successful backend attempt per callback."""
        return {"n_retry_llm": 1}


class BrowserGymHarness(AgenticHarness):
    """Run one pinned BrowserGym benchmark through AgentLab's canonical loop."""

    def __init__(
        self,
        *,
        dataset_name: str,
        benchmark_name: str,
        benchmark: Benchmark,
        dataset_revision: str,
        model_name: str,
        config: dict[str, Any],
    ) -> None:
        self._dataset_name = dataset_name
        self._benchmark_name = benchmark_name
        self._benchmark = benchmark
        self._model_name = model_name
        self._environment = config["environment"]
        self._output_dir = Path(config["output_dir"])
        self._max_tokens = config["max_tokens"]
        self._context_window = config["context_window"]
        self._primary_reward = config["primary_reward"] or "reward"
        self._vision = benchmark_name.startswith("visualwebarena")
        self._events = EventQueue()
        self._broker = ModelCallBroker(self._events)
        self._broker_id = register_broker(self._broker)
        self._episodes: list[AgenticEpisode] = []
        self._env_by_episode: dict[str, Any] = {}
        for index, env_args in enumerate(benchmark.env_args_list):
            seed = env_args.task_seed
            digest = hashlib.sha256(
                f"{dataset_revision}\0{env_args.task_name}\0{seed}".encode()
            ).hexdigest()[:20]
            episode_id = f"browsergym:{index:08d}:{digest}"
            self._episodes.append(
                AgenticEpisode(
                    episode_id=episode_id,
                    task=f"{env_args.task_name}[seed={seed}]",
                    source=dataset_name,
                )
            )
            self._env_by_episode[episode_id] = env_args
        self._episode_by_id = {
            episode.episode_id: episode for episode in self._episodes
        }
        self._active: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, AgenticEpisodeResult] = {}
        self._closed = False
        distributions = _benchmark_distributions(benchmark_name)
        self._identity = {
            "harness": "agentlab-browsergym",
            "harness_version": (
                f"agentlab-{_AGENTLAB_VERSION}+browsergym-{_BROWSERGYM_VERSION}"
            ),
            "harness_source_sha256": _distribution_source_digest(distributions),
            "dataset": {
                "provider": "BrowserGym DEFAULT_BENCHMARKS",
                "benchmark": dataset_name,
                "repository": "https://github.com/ServiceNow/BrowserGym",
                "revision": dataset_revision,
                "evaluation_splits": _evaluation_splits(benchmark),
            },
            "agent": "AgentLab GenericAgent (GPT-4o prompt profile)",
            "agent_version": (
                f"agentlab-{_AGENTLAB_VERSION}@{_AGENTLAB_COMMIT}"
                "+aiperf-rust-callback-1"
            ),
            "environment": self._environment,
            "verifier": _verifier_identity(benchmark_name),
            "episode_count": len(self._episodes),
            "primary_reward": self._primary_reward,
        }

    @classmethod
    async def create(
        cls, dataset: str, model_name: str, authored_config: Any
    ) -> BrowserGymHarness:
        """Freeze a pinned benchmark and prepare its canonical backend."""
        _require_browsergym_environment()
        config = _validate_config(authored_config)
        dataset_name, benchmark_name = _parse_dataset(dataset)
        try:
            builder = DEFAULT_BENCHMARKS[benchmark_name]
        except KeyError as error:
            choices = ", ".join(sorted(DEFAULT_BENCHMARKS))
            raise ValueError(
                f"unknown BrowserGym benchmark {benchmark_name!r}; available: {choices}"
            ) from error
        benchmark = builder()
        benchmark = _select_tasks(benchmark, config["task_names"])
        if config["max_episodes"] is not None:
            limited = copy(benchmark)
            limited.env_args_list = benchmark.env_args_list[: config["max_episodes"]]
            benchmark = limited
        if not benchmark.env_args_list:
            raise ValueError("BrowserGym selection produced zero episodes")
        benchmark.env_args_list = _canonical_sequential_env_args(benchmark)
        revision = _benchmark_revision(benchmark_name, benchmark)
        await asyncio.to_thread(benchmark.prepare_backends)
        return cls(
            dataset_name=dataset_name,
            benchmark_name=benchmark_name,
            benchmark=benchmark,
            dataset_revision=revision,
            model_name=require_identifier(model_name, "model"),
            config=config,
        )

    @property
    @override
    def identity(self) -> dict[str, Any]:
        """Return immutable AgentLab, BrowserGym, task, and verifier identity."""
        return dict(self._identity)

    @property
    @override
    def episodes(self) -> list[AgenticEpisode]:
        """Return BrowserGym's selected episodes in dependency-safe order."""
        return list(self._episodes)

    @override
    async def start_episodes(self, episode_ids: list[str]) -> None:
        """Start the Rust-admitted canonical environment episode."""
        self._ensure_open()
        if len(episode_ids) != 1:
            raise ValueError(
                "AgentLab/BrowserGym currently requires one sequential episode start"
            )
        episode_id = episode_ids[0]
        if episode_id not in self._episode_by_id:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        if episode_id in self._active or episode_id in self._results:
            raise ValueError(f"episode {episode_id!r} was already started")
        if self._active:
            raise RuntimeError("AgentLab/BrowserGym already has an active episode")
        self._active[episode_id] = asyncio.create_task(
            self._run_episode(episode_id),
            name=f"aiperf-browsergym-{episode_id}",
        )

    @override
    async def poll_events(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        """Return model calls and terminal canonical reward events."""
        self._ensure_open()
        return await self._events.poll(limit, wait_ms)

    @override
    async def submit_model_results(self, items: list[AgenticModelResult]) -> None:
        """Resume synchronous AgentLab callbacks with Rust terminal results."""
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
        """Cancel selected active AgentLab episodes and unblock model callbacks."""
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
        """Require one terminal canonical result for every selected episode."""
        self._ensure_open()
        if self._active:
            raise RuntimeError(
                "finish_agentic called with active BrowserGym episodes: "
                + ", ".join(sorted(self._active))
            )
        missing = [
            episode.episode_id
            for episode in self._episodes
            if episode.episode_id not in self._results
        ]
        if missing:
            raise RuntimeError(
                "finish_agentic called before every BrowserGym episode was started: "
                + ", ".join(missing)
            )
        return [self._results[episode.episode_id] for episode in self._episodes]

    @override
    async def close(self) -> None:
        """Cancel active episodes and release the process-local model broker."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._active.items())
        for episode_id, task in tasks:
            self._broker.fail_episode(
                episode_id, RuntimeError("AgentLab/BrowserGym harness closed")
            )
            task.cancel()
        if tasks:
            await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
        self._broker.close()
        unregister_broker(self._broker_id)

    async def _run_episode(self, episode_id: str) -> None:
        episode = self._episode_by_id[episode_id]
        started = time.monotonic()
        thread_result = _start_daemon_thread(
            lambda: self._execute_episode(episode_id),
            name=f"aiperf-browsergym-env-{episode_id}",
        )
        try:
            artifact_path, summary = await asyncio.shield(thread_result)
            result = _convert_summary(
                episode,
                summary,
                duration_seconds=time.monotonic() - started,
                model_calls=self._broker.model_call_count(episode_id),
                primary_reward=self._primary_reward,
                artifact_path=artifact_path,
            )
        except asyncio.CancelledError:
            self._broker.fail_episode(
                episode_id, RuntimeError("episode cancelled by Rust scheduler")
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(thread_result), timeout=_THREAD_DRAIN_SECONDS
                )
            except Exception:
                _LOG.warning(
                    "BrowserGym environment thread did not drain during cancellation: %s",
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

    def _execute_episode(self, episode_id: str) -> tuple[str, dict[str, Any]]:
        env_args = deepcopy(self._env_by_episode[episode_id])
        flags = deepcopy(FLAGS_GPT_4o_VISION if self._vision else FLAGS_GPT_4o)
        model_args = AIPerfAgentLabModelArgs(
            model_name=_MODEL_PROFILE,
            max_total_tokens=self._context_window,
            max_input_tokens=max(1, self._context_window - self._max_tokens),
            max_new_tokens=self._max_tokens,
            temperature=0.1,
            vision_support=self._vision,
            broker_id=self._broker_id,
            episode_id=episode_id,
            target_model=self._model_name,
        )
        agent_args = GenericAgentArgs(chat_model_args=model_args, flags=flags)
        agent_args.set_benchmark(self._benchmark, demo_mode=False)
        exp_args = ExpArgs(
            agent_args=agent_args,
            env_args=env_args,
            logging_level=logging.INFO,
            logging_level_stdout=logging.WARNING,
            save_screenshot=True,
            save_som=self._vision,
        )
        exp_args.prepare(self._output_dir)
        exp_args.run()
        artifact_path = Path(exp_args.exp_dir)
        summary_path = artifact_path / "summary_info.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"AgentLab did not produce canonical summary {summary_path}"
            )
        summary = orjson.loads(summary_path.read_bytes())
        if not isinstance(summary, dict):
            raise TypeError("AgentLab summary_info.json must contain an object")
        return artifact_path.as_posix(), summary

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AgentLab/BrowserGym harness is closed")


async def create_browsergym_harness(
    dataset: str, model_name: str, config: Any
) -> AgenticHarness:
    """Factory called lazily by the evaluator's agentic harness registry."""
    return await BrowserGymHarness.create(dataset, model_name, config)


def is_browsergym_dataset(dataset: str) -> bool:
    """Return whether the opaque dataset name selects this harness provider."""
    return dataset.strip().lower().startswith(_DATASET_PREFIX)


def _openai_messages(
    messages: list[dict[str, Any]] | Discussion,
) -> list[dict[str, Any]]:
    if isinstance(messages, Discussion):
        authored = messages.to_openai()
    elif isinstance(messages, list):
        authored = messages
    else:
        raise TypeError("AgentLab model messages must be a Discussion or list")
    result = []
    for index, message in enumerate(authored):
        if not isinstance(message, dict):
            raise TypeError(f"AgentLab message {index} must be an object")
        result.append(deepcopy(dict(message)))
    if not result:
        raise ValueError("AgentLab model call must contain at least one message")
    return result


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"text", "input_text"} and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
            return "\n".join(parts)
    return ""


def _parse_dataset(authored: str) -> tuple[str, str]:
    dataset = require_identifier(authored, "agentic dataset")
    if not is_browsergym_dataset(dataset):
        raise ValueError(f"BrowserGym dataset must use {_DATASET_PREFIX}<benchmark>")
    name_and_ref = dataset[len(_DATASET_PREFIX) :]
    name, separator, revision = name_and_ref.partition("@")
    benchmark_name = require_identifier(name, "BrowserGym benchmark").lower()
    if separator and revision != _BROWSERGYM_VERSION:
        raise ValueError(
            f"BrowserGym benchmark revision must be {_BROWSERGYM_VERSION!r}, "
            f"not {revision!r}"
        )
    return f"{_DATASET_PREFIX}{benchmark_name}@{_BROWSERGYM_VERSION}", benchmark_name


def _validate_config(authored: Any) -> dict[str, Any]:
    if authored is None:
        authored = {}
    if not isinstance(authored, dict):
        raise TypeError("load_agentic.config must be an object")
    unknown = sorted(set(authored) - _COMMON_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "load_agentic.config has unknown field(s): " + ", ".join(unknown)
        )
    task_names = authored.get("task_names")
    if task_names is not None:
        if not isinstance(task_names, list) or not task_names:
            raise ValueError("task_names must be a non-empty array or null")
        task_names = [
            require_identifier(item, "task_names item") for item in task_names
        ]
    max_episodes = authored.get("max_episodes")
    if max_episodes is not None:
        max_episodes = require_positive_int(max_episodes, "max_episodes")
    task_concurrency = require_positive_int(
        authored.get("task_concurrency", 1), "task_concurrency"
    )
    if task_concurrency != 1:
        raise ValueError(
            "AgentLab/BrowserGym preserves stateful benchmark dependencies with "
            "task_concurrency=1"
        )
    environment = authored.get("environment", "browsergym")
    if environment != "browsergym":
        raise ValueError(
            "AgentLab/BrowserGym requires --agentic-environment browsergym"
        )
    output_dir = require_identifier(
        authored.get("output_dir", "artifacts/agentic"), "output_dir"
    )
    if authored.get("max_turns") is not None:
        raise ValueError(
            "AgentLab/BrowserGym uses canonical per-benchmark max_steps and does not "
            "accept max_turns"
        )
    max_tokens = require_positive_int(authored.get("max_tokens", 4_096), "max_tokens")
    context_window = require_positive_int(
        authored.get("context_window", _DEFAULT_CONTEXT_WINDOW), "context_window"
    )
    if max_tokens > context_window:
        raise ValueError("max_tokens must not exceed context_window")
    if authored.get("parser", "json") != "json":
        raise ValueError("AgentLab/BrowserGym does not use the Harbor parser option")
    if authored.get("enable_summarize", True) is not True:
        raise ValueError(
            "AgentLab/BrowserGym does not use the Harbor summarization option"
        )
    if authored.get("overwrite", False) is not False:
        raise ValueError("AgentLab/BrowserGym does not use the Harbor overwrite option")
    primary_reward = authored.get("primary_reward")
    if primary_reward not in {None, "reward", "raw_reward"}:
        raise ValueError("BrowserGym primary_reward must be reward or raw_reward")
    inference_gateway = authored.get("inference_gateway")
    if inference_gateway is not None and not isinstance(inference_gateway, dict):
        raise TypeError("inference_gateway must be an object or null")
    return {
        "task_names": task_names,
        "max_episodes": max_episodes,
        "task_concurrency": task_concurrency,
        "environment": environment,
        "output_dir": output_dir,
        "max_tokens": max_tokens,
        "context_window": context_window,
        "primary_reward": primary_reward,
    }


def _select_tasks(benchmark: Benchmark, patterns: list[str] | None) -> Benchmark:
    if patterns is None:
        return benchmark
    available = [env_args.task_name for env_args in benchmark.env_args_list]
    matched_by_pattern = {
        pattern: [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        for pattern in patterns
    }
    unmatched = [
        pattern for pattern, matches in matched_by_pattern.items() if not matches
    ]
    if unmatched:
        raise ValueError(
            "BrowserGym task pattern(s) matched nothing: " + ", ".join(unmatched)
        )
    selected = set(name for matches in matched_by_pattern.values() for name in matches)
    ordered_names = list(dict.fromkeys(name for name in available if name in selected))
    return benchmark.subset_from_list(
        ordered_names, benchmark_name_suffix="aiperf-selection"
    )


def _canonical_sequential_env_args(benchmark: Benchmark) -> list[Any]:
    env_args = list(benchmark.env_args_list)
    dependencies = benchmark.dependency_graph_over_tasks()
    if not any(dependencies.values()):
        return env_args
    grouped: dict[str, list[Any]] = {}
    first_index: dict[str, int] = {}
    for index, item in enumerate(env_args):
        grouped.setdefault(item.task_name, []).append(item)
        first_index.setdefault(item.task_name, index)
    repeated = [name for name, items in grouped.items() if len(items) != 1]
    if repeated:
        raise ValueError(
            "BrowserGym dependency scheduling requires one seed per task: "
            + ", ".join(sorted(repeated))
        )
    children = {name: [] for name in grouped}
    indegree = {name: 0 for name in grouped}
    for child, parents in dependencies.items():
        if child not in grouped:
            continue
        for parent in parents:
            if parent not in grouped:
                raise ValueError(
                    f"BrowserGym dependency {parent!r} for {child!r} is absent"
                )
            children[parent].append(child)
            indegree[child] += 1
    ready = sorted(
        (first_index[name], name) for name, degree in indegree.items() if degree == 0
    )
    ordered = []
    while ready:
        _, name = ready.pop(0)
        ordered.extend(grouped[name])
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append((first_index[child], child))
                ready.sort()
    if len(ordered) != len(env_args):
        raise ValueError("BrowserGym task dependency graph contains a cycle")
    return ordered


def _benchmark_revision(benchmark_name: str, benchmark: Benchmark) -> str:
    distributions = _benchmark_distributions(benchmark_name)
    payload = {
        "browsergym_commit": _BROWSERGYM_COMMIT,
        "benchmark": benchmark_name,
        "env_args": [asdict(item) for item in benchmark.env_args_list],
        "task_metadata": benchmark.task_metadata.to_dict(orient="records"),
        "distributions": {
            name: importlib.metadata.version(name) for name in distributions
        },
    }
    digest = hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS, default=str)
    ).hexdigest()
    return f"sha256:{digest}"


def _evaluation_splits(benchmark: Benchmark) -> list[str]:
    metadata = benchmark.task_metadata
    if "browsergym_split" not in metadata.columns:
        return ["tasks"]
    selected = {item.task_name for item in benchmark.env_args_list}
    values = metadata[metadata["task_name"].isin(selected)]["browsergym_split"]
    result = sorted({str(value) for value in values if str(value)})
    return result or ["tasks"]


def _benchmark_distributions(benchmark_name: str) -> tuple[str, ...]:
    names = ["agentlab", "browsergym-core", "browsergym-experiments"]
    if benchmark_name.startswith("miniwob"):
        names.append("browsergym-miniwob")
    elif benchmark_name == "webarena_verified":
        names.extend(
            [
                "browsergym-webarena",
                "browsergym-webarena-verified",
                "libwebarena",
                "webarena-verified",
            ]
        )
    elif benchmark_name == "webarena_lite":
        names.extend(["browsergym-webarena", "browsergym-webarenalite", "libwebarena"])
    elif benchmark_name.startswith("webarena"):
        names.extend(["browsergym-webarena", "libwebarena"])
    elif benchmark_name.startswith("visualwebarena"):
        names.extend(
            [
                "browsergym-webarena",
                "browsergym-visualwebarena",
                "libwebarena",
                "libvisualwebarena",
            ]
        )
    elif benchmark_name.startswith("workarena"):
        names.append("browsergym-workarena")
    elif benchmark_name == "assistantbench":
        names.append("browsergym-assistantbench")
    elif benchmark_name == "weblinx":
        names.append("weblinx-browsergym")
    return tuple(names)


def _distribution_source_digest(distributions: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(distributions):
        distribution = importlib.metadata.distribution(name)
        version = distribution.version
        identity = f"{name}=={version}".encode()
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        files = distribution.files or []
        for relative in sorted(files, key=str):
            if Path(str(relative)).suffix not in {
                ".csv",
                ".json",
                ".py",
                ".yaml",
                ".yml",
            }:
                continue
            path = Path(distribution.locate_file(relative))
            if not path.is_file():
                continue
            relative_bytes = str(relative).encode()
            payload = path.read_bytes()
            digest.update(len(relative_bytes).to_bytes(8, "big"))
            digest.update(relative_bytes)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _verifier_identity(benchmark_name: str) -> str:
    if benchmark_name == "webarena_verified":
        return "WebArenaVerifiedEvaluatorAPI over BrowserGym trace"
    return "BrowserGym canonical environment reward"


def _require_browsergym_environment() -> None:
    expected = {
        "agentlab": _AGENTLAB_VERSION,
        "browsergym-core": _BROWSERGYM_VERSION,
        "browsergym-experiments": _BROWSERGYM_VERSION,
    }
    mismatches = []
    for package, wanted in expected.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != wanted:
            mismatches.append(f"{package}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise RuntimeError(
            "BrowserGym evaluator environment does not match its pinned lock: "
            + ", ".join(mismatches)
        )


def _convert_summary(
    episode: AgenticEpisode,
    summary: dict[str, Any],
    *,
    duration_seconds: float,
    model_calls: int,
    primary_reward: str,
    artifact_path: str,
) -> AgenticEpisodeResult:
    error_message = summary.get("err_msg")
    if error_message is not None:
        if not isinstance(error_message, str):
            raise TypeError("AgentLab summary err_msg must be a string or null")
        return AgenticEpisodeResult(
            episode_id=episode.episode_id,
            task=episode.task,
            outcome="infrastructure_error",
            rewards={},
            primary_reward=None,
            duration_seconds=duration_seconds,
            model_calls=model_calls,
            error_kind="AgentLabEpisodeError",
            error_message=error_message,
            artifact_path=artifact_path,
        )
    if not summary.get("terminated", False) and not summary.get("truncated", False):
        raise RuntimeError("AgentLab episode ended without terminal or truncated state")
    rewards = {"reward": require_finite_number(summary.get("cum_reward"), "cum_reward")}
    raw_reward = summary.get("cum_raw_reward")
    if raw_reward is not None:
        rewards["raw_reward"] = require_finite_number(raw_reward, "cum_raw_reward")
    if primary_reward not in rewards:
        raise RuntimeError(
            f"AgentLab summary omitted selected primary reward {primary_reward!r}"
        )
    return AgenticEpisodeResult(
        episode_id=episode.episode_id,
        task=episode.task,
        outcome="completed",
        rewards=rewards,
        primary_reward=primary_reward,
        duration_seconds=duration_seconds,
        model_calls=model_calls,
        artifact_path=artifact_path,
    )


def _start_daemon_thread(
    function: Callable[[], _T], *, name: str
) -> asyncio.Future[_T]:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def run() -> None:
        try:
            value = function()
        except BaseException as error:
            try:
                loop.call_soon_threadsafe(_finish_thread_error, future, error)
            except RuntimeError:
                _LOG.exception("BrowserGym thread outlived its worker event loop")
        else:
            try:
                loop.call_soon_threadsafe(_finish_thread_value, future, value)
            except RuntimeError:
                _LOG.exception("BrowserGym thread outlived its worker event loop")

    threading.Thread(target=run, name=name, daemon=True).start()
    return future


def _finish_thread_value(future: asyncio.Future[_T], value: _T) -> None:
    if not future.done():
        future.set_result(value)


def _finish_thread_error(future: asyncio.Future[_T], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)
