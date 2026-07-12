# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Harbor implementation of the agentic evaluator contract.

Harbor owns task resolution, sandbox lifecycle, the Terminus-2 agent loop,
terminal execution, trajectories, and task verifiers. Its model backend is
replaced by :mod:`aiperf.accuracy.harbor_agent`, whose only operation is to
publish a callback event for Rust inference.

No benchmark-specific scorer is reproduced here. Consequently every versioned
Harbor task package—including SWE-bench, Terminal-Bench, BFCL, tau-bench,
OSWorld, GAIA, and future registry datasets—uses its packaged environment and
verifier unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any

from aiperf.accuracy.agentic import (
    AgenticEpisode,
    AgenticEpisodeResult,
    AgenticEvent,
    AgenticHarness,
    AgenticModelResult,
    EventQueue,
    require_identifier,
    require_positive_int,
)

_HARBOR_VERSION = "0.18.0"
_AGENT_IMPORT_PATH = "aiperf.accuracy.harbor_agent:AIPerfTerminus2"
_CONFIG_FIELDS = {
    "context_window",
    "enable_summarize",
    "environment",
    "max_episodes",
    "max_tokens",
    "max_turns",
    "output_dir",
    "overwrite",
    "parser",
    "primary_reward",
    "task_concurrency",
    "task_names",
}
_ENVIRONMENTS = {
    "ack",
    "apple-container",
    "beam",
    "blaxel",
    "cua-cloud",
    "cwsandbox",
    "daytona",
    "docker",
    "e2b",
    "ec2",
    "gke",
    "islo",
    "langsmith",
    "modal",
    "novita",
    "opensandbox",
    "openshift",
    "runloop",
    "singularity",
    "tensorlake",
    "use-computer",
    "wandb",
}


class HarborHarness(AgenticHarness):
    """Run versioned Harbor tasks with Rust-backed Terminus-2 inference."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_revision: str,
        task_configs: list[Any],
        model_name: str,
        config: dict[str, Any],
    ) -> None:
        from aiperf.accuracy.harbor_agent import ModelCallBroker, register_broker

        self._dataset_name = dataset_name
        self._dataset_revision = dataset_revision
        self._task_configs = task_configs
        self._model_name = model_name
        self._environment = str(config.get("environment", "docker"))
        self._output_dir = Path(config.get("output_dir", "artifacts/agentic"))
        self._max_active = int(config.get("task_concurrency", 1))
        self._max_turns = config.get("max_turns")
        self._max_tokens = int(config.get("max_tokens", 4096))
        self._context_window = int(config.get("context_window", 131072))
        self._parser = str(config.get("parser", "json"))
        self._enable_summarize = bool(config.get("enable_summarize", True))
        self._primary_reward = config.get("primary_reward")
        self._events = EventQueue()
        self._broker = ModelCallBroker(self._events)
        self._broker_id = register_broker(self._broker)
        self._episodes: list[AgenticEpisode] = []
        self._task_by_episode: dict[str, Any] = {}
        for index, task_config in enumerate(task_configs):
            task_name = task_config.get_task_id().get_name()
            digest = hashlib.sha256(
                f"{dataset_revision}\0{task_name}".encode()
            ).hexdigest()[:20]
            episode_id = f"harbor:{index:08d}:{digest}"
            episode = AgenticEpisode(
                episode_id=episode_id,
                task=task_name,
                source=dataset_name,
            )
            self._episodes.append(episode)
            self._task_by_episode[episode_id] = task_config
        self._episode_by_id = {
            episode.episode_id: episode for episode in self._episodes
        }
        self._active: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, AgenticEpisodeResult] = {}
        self._closed = False
        self._identity = {
            "harness": "harbor",
            "harness_version": _HARBOR_VERSION,
            "harness_source_sha256": _harbor_source_digest(),
            "dataset": {
                "provider": "harbor package registry",
                "benchmark": dataset_name,
                "repository": dataset_name,
                "revision": dataset_revision,
                "evaluation_splits": ["tasks"],
            },
            "agent": "aiperf-terminus-2",
            "agent_version": "1.0.0+terminus-2.0.0",
            "environment": self._environment,
            "verifier": "harbor packaged task verifier",
            "episode_count": len(self._episodes),
            "primary_reward": self._primary_reward,
        }

    @classmethod
    async def create(
        cls, dataset: str, model_name: str, authored_config: Any
    ) -> HarborHarness:
        """Resolve one immutable Harbor dataset and freeze its ordered task list."""
        _require_harbor()
        config = _validate_config(authored_config)
        from harbor.models.job.config import DatasetConfig

        dataset_name = require_identifier(dataset, "agentic dataset")
        max_episodes = config.get("max_episodes")
        task_names = config.get("task_names")
        overwrite = bool(config.get("overwrite", False))
        dataset_path = Path(dataset_name).expanduser()
        if dataset_path.exists():
            dataset_config = DatasetConfig(
                path=dataset_path,
                task_names=task_names,
                n_tasks=max_episodes,
                overwrite=overwrite,
            )
            dataset_revision = _directory_digest(dataset_path)
            canonical_name = dataset_path.resolve().as_posix()
        else:
            package_name, separator, ref = dataset_name.partition("@")
            if "/" not in package_name:
                raise ValueError(
                    "agentic dataset must be a Harbor package org/name[@revision] "
                    "or an existing local dataset directory"
                )
            dataset_config = DatasetConfig(
                name=package_name,
                ref=ref if separator else "latest",
                task_names=task_names,
                n_tasks=max_episodes,
                overwrite=overwrite,
            )
            canonical_name = package_name
            dataset_revision = ""
        task_configs = await dataset_config.get_task_configs()
        if not task_configs:
            raise ValueError(f"Harbor dataset {dataset_name!r} resolved no tasks")
        if not dataset_revision:
            dataset_revision = require_identifier(
                dataset_config.ref, "resolved Harbor dataset revision"
            )
        return cls(
            dataset_name=canonical_name,
            dataset_revision=dataset_revision,
            task_configs=task_configs,
            model_name=require_identifier(model_name, "model"),
            config=config,
        )

    @property
    def identity(self) -> dict[str, Any]:
        """Return immutable Harbor, dataset, scaffold, environment, and verifier identity."""
        return self._identity

    @property
    def episodes(self) -> list[AgenticEpisode]:
        """Return selected Harbor tasks in registry order."""
        return list(self._episodes)

    async def start_episodes(self, episode_ids: list[str]) -> None:
        """Start selected trials without waiting for environment setup."""
        self._ensure_open()
        if not episode_ids:
            raise ValueError("start_episodes.episode_ids must not be empty")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("start_episodes contains duplicate episode IDs")
        available = self._max_active - len(self._active)
        if len(episode_ids) > available:
            raise ValueError(
                f"starting {len(episode_ids)} episode(s) exceeds the configured "
                f"task_concurrency={self._max_active} ({available} slot(s) free)"
            )
        for episode_id in episode_ids:
            if episode_id not in self._episode_by_id:
                raise KeyError(f"unknown episode_id {episode_id!r}")
            if episode_id in self._active or episode_id in self._results:
                raise ValueError(f"episode {episode_id!r} was already started")
        for episode_id in episode_ids:
            self._active[episode_id] = asyncio.create_task(
                self._run_episode(episode_id),
                name=f"aiperf-harbor-{episode_id}",
            )

    async def poll_events(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        """Drive Harbor tasks while returning ready callback/final events."""
        self._ensure_open()
        return await self._events.poll(limit, wait_ms)

    async def submit_model_results(self, items: list[AgenticModelResult]) -> None:
        """Resume outstanding Harbor LLM calls with Rust inference results."""
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
        # Let resumed agent tasks reach their next environment wait or model call
        # before the next JSONL operation is handled.
        await asyncio.sleep(0)

    async def cancel_episodes(self, episode_ids: list[str]) -> None:
        """Cancel selected active trials; each emits a cancelled terminal event."""
        self._ensure_open()
        tasks = []
        for episode_id in episode_ids:
            task = self._active.get(episode_id)
            if task is None:
                raise KeyError(f"episode {episode_id!r} is not active")
            task.cancel()
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def finish(self) -> list[AgenticEpisodeResult]:
        """Require every selected task to have one terminal result."""
        self._ensure_open()
        if self._active:
            raise RuntimeError(
                "finish_agentic called with active episodes: "
                + ", ".join(sorted(self._active))
            )
        missing = [
            episode.episode_id
            for episode in self._episodes
            if episode.episode_id not in self._results
        ]
        if missing:
            raise RuntimeError(
                "finish_agentic called before every episode was started: "
                + ", ".join(missing)
            )
        return [self._results[episode.episode_id] for episode in self._episodes]

    async def close(self) -> None:
        """Cancel active Harbor trials and unregister the callback broker."""
        if self._closed:
            return
        self._closed = True
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._broker.close()
        from aiperf.accuracy.harbor_agent import unregister_broker

        unregister_broker(self._broker_id)

    async def _run_episode(self, episode_id: str) -> None:
        episode = self._episode_by_id[episode_id]
        started = time.monotonic()
        result: AgenticEpisodeResult
        try:
            from harbor.models.environment_type import EnvironmentType
            from harbor.models.trial.config import (
                AgentConfig,
                EnvironmentConfig,
                TrialConfig,
            )
            from harbor.trial.trial import Trial

            agent_kwargs: dict[str, Any] = {
                "aiperf_broker_id": self._broker_id,
                "aiperf_episode_id": episode_id,
                "aiperf_context_limit": self._context_window,
                "aiperf_output_limit": self._max_tokens,
                "enable_summarize": self._enable_summarize,
                "parser_name": self._parser,
            }
            if self._max_turns is not None:
                agent_kwargs["max_turns"] = self._max_turns
                agent_kwargs["suppress_max_turns_warning"] = True
            trial_name = episode_id.replace(":", "-")
            config = TrialConfig(
                task=self._task_by_episode[episode_id],
                trial_name=trial_name,
                trials_dir=self._output_dir,
                agent=AgentConfig(
                    import_path=_AGENT_IMPORT_PATH,
                    model_name=self._model_name,
                    kwargs=agent_kwargs,
                ),
                environment=EnvironmentConfig(type=EnvironmentType(self._environment)),
            )
            trial = await Trial.create(config)
            trial_result = await trial.run()
            result = _convert_trial_result(
                episode,
                trial_result,
                duration_seconds=time.monotonic() - started,
                model_calls=self._broker.model_call_count(episode_id),
                primary_reward=self._primary_reward,
                artifact_path=(self._output_dir / trial_name).as_posix(),
            )
        except asyncio.CancelledError:
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

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Harbor harness is closed")


async def create_harbor_harness(
    dataset: str, model_name: str, config: Any
) -> AgenticHarness:
    """Factory used by the worker without importing Harbor during static runs."""
    return await HarborHarness.create(dataset, model_name, config)


def _validate_config(authored: Any) -> dict[str, Any]:
    if authored is None:
        authored = {}
    if not isinstance(authored, dict):
        raise TypeError("load_agentic.config must be an object")
    unknown = sorted(set(authored) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "load_agentic.config has unknown field(s): " + ", ".join(unknown)
        )
    config = dict(authored)
    environment = config.get("environment", "docker")
    if environment not in _ENVIRONMENTS:
        raise ValueError(
            f"unsupported Harbor environment {environment!r}; available: "
            + ", ".join(sorted(_ENVIRONMENTS))
        )
    config["task_concurrency"] = require_positive_int(
        config.get("task_concurrency", 1), "task_concurrency"
    )
    config["max_tokens"] = require_positive_int(
        config.get("max_tokens", 4096), "max_tokens"
    )
    config["context_window"] = require_positive_int(
        config.get("context_window", 131072), "context_window"
    )
    if config["max_tokens"] > config["context_window"]:
        raise ValueError("max_tokens must not exceed context_window")
    for name in ("max_episodes", "max_turns"):
        if config.get(name) is not None:
            config[name] = require_positive_int(config[name], name)
    parser = config.get("parser", "json")
    if parser not in {"json", "xml"}:
        raise ValueError("parser must be 'json' or 'xml'")
    task_names = config.get("task_names")
    if task_names is not None:
        if not isinstance(task_names, list) or not task_names:
            raise TypeError("task_names must be a non-empty array of strings or null")
        config["task_names"] = [
            require_identifier(task_name, "task_names item") for task_name in task_names
        ]
    primary_reward = config.get("primary_reward")
    if primary_reward is not None:
        config["primary_reward"] = require_identifier(primary_reward, "primary_reward")
    for name in ("enable_summarize", "overwrite"):
        if name in config and not isinstance(config[name], bool):
            raise TypeError(f"{name} must be a boolean")
    output_dir = config.get("output_dir")
    if output_dir is not None and (
        not isinstance(output_dir, str) or not output_dir.strip()
    ):
        raise TypeError("output_dir must be a non-empty string")
    return config


def _convert_trial_result(
    episode: AgenticEpisode,
    trial_result: Any,
    *,
    duration_seconds: float,
    model_calls: int,
    primary_reward: str | None,
    artifact_path: str,
) -> AgenticEpisodeResult:
    exception = trial_result.exception_info
    if exception is not None:
        return AgenticEpisodeResult(
            episode_id=episode.episode_id,
            task=episode.task,
            outcome="infrastructure_error",
            rewards={},
            primary_reward=None,
            duration_seconds=duration_seconds,
            model_calls=model_calls,
            error_kind=exception.exception_type,
            error_message=exception.exception_message,
            artifact_path=artifact_path,
        )
    verifier = trial_result.verifier_result
    if verifier is None or verifier.rewards is None:
        return AgenticEpisodeResult(
            episode_id=episode.episode_id,
            task=episode.task,
            outcome="infrastructure_error",
            rewards={},
            primary_reward=None,
            duration_seconds=duration_seconds,
            model_calls=model_calls,
            error_kind="MissingVerifierResult",
            error_message="Harbor trial completed without verifier rewards",
            artifact_path=artifact_path,
        )
    rewards: dict[str, float] = {}
    for name, authored in verifier.rewards.items():
        value = float(authored)
        if not math.isfinite(value):
            raise ValueError(f"Harbor verifier reward {name!r} was not finite")
        rewards[name] = value
    selected_reward = primary_reward
    if selected_reward is not None and selected_reward not in rewards:
        raise ValueError(
            f"configured primary_reward {selected_reward!r} was absent; "
            f"available: {sorted(rewards)}"
        )
    if selected_reward is None:
        if "reward" in rewards:
            selected_reward = "reward"
        elif len(rewards) == 1:
            selected_reward = next(iter(rewards))
    prompt_tokens, cached_tokens, completion_tokens, _ = (
        trial_result.compute_token_cost_totals()
    )
    return AgenticEpisodeResult(
        episode_id=episode.episode_id,
        task=episode.task,
        outcome="completed",
        rewards=rewards,
        primary_reward=selected_reward,
        duration_seconds=duration_seconds,
        model_calls=model_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        artifact_path=artifact_path,
    )


def _require_harbor() -> None:
    try:
        actual = importlib.metadata.version("harbor")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "agentic evaluation requires the pinned agentic worker environment "
            f"with harbor=={_HARBOR_VERSION}"
        ) from error
    if actual != _HARBOR_VERSION:
        raise RuntimeError(
            f"agentic evaluator has harbor={actual!r}; expected {_HARBOR_VERSION!r}"
        )


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _harbor_source_digest() -> str:
    import harbor

    package_file = Path(harbor.__file__).resolve()
    return _directory_digest(package_file.parent)
