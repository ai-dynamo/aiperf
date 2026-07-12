# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived stdio accuracy evaluator for the native Rust AIPerf frontend.

Rust owns inference I/O and sends completed response text here only after the
normal transport reaches a terminal state. This process owns benchmark loading,
prompt construction, hidden test material, and canonical Python/Lighteval
grading. stdin/stdout are a versioned JSONL protocol; logs use stderr only.

The inherited benchmark and grader implementations are intentionally reused in
this process. Their complete ownership path is documented in
``src/aiperf/dataset/loader/accuracy_dataset_loader.py:21-150`` and
``src/aiperf/accuracy/accuracy_record_processor.py:21-147``. MMLU-Pro delegates
dataset, prompt, and metric construction to the pinned Lighteval task registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

from aiperf.accuracy.agentic import (
    AgenticHarness,
    AgenticHarnessProvider,
    AgenticModelResult,
    require_identifier,
    require_non_negative_int,
    require_positive_int,
)

PROTOCOL_VERSION = 1
WORKER_VERSION = "1.6.0"
_LOG = logging.getLogger("aiperf.accuracy.worker")
_LOCKED_PACKAGE_VERSIONS = {
    "datasets": "5.0.0",
    "deepeval": "4.0.9",
    "latex2sympy2-extended": "1.0.6",
    "lighteval": "0.13.0",
    "sympy": "1.14.0",
    "word2number": "1.1",
}
_HARBOR_LOCKED_PACKAGE_VERSIONS = {"harbor": "0.18.0"}
_BROWSERGYM_LOCKED_PACKAGE_VERSIONS = {
    "agentlab": "0.4.2",
    "browsergym-core": "0.14.3",
    "browsergym-experiments": "0.14.3",
}
_LCB_MAX_RELEASE = 6


class _LazyAgenticHarnessProvider(AgenticHarnessProvider):
    """Import one optional canonical harness only after namespace selection."""

    def __init__(
        self,
        *,
        capability: str,
        namespace_prefix: str | None,
        factory: str,
        required_versions: dict[str, str],
    ) -> None:
        self._capability = capability
        self._namespace_prefix = namespace_prefix
        self._factory = factory
        self._required_versions = required_versions

    @property
    def capability(self) -> str:
        """Return this provider's handshake capability."""
        return self._capability

    def is_available(self) -> bool:
        """Require every evaluator-critical package at its exact pinned version."""
        return all(
            _package_version(package) == expected
            for package, expected in self._required_versions.items()
        )

    def matches(self, dataset: str) -> bool:
        """Reserve a namespace prefix or act as the final Harbor fallback."""
        if self._namespace_prefix is None:
            return True
        return dataset.strip().lower().startswith(self._namespace_prefix)

    async def create(
        self, dataset: str, model_name: str, config: Any
    ) -> AgenticHarness:
        """Import and invoke the provider's async factory."""
        factory = _import_symbol(self._factory)
        return await factory(dataset, model_name, config)


_AGENTIC_HARNESS_PROVIDERS: tuple[AgenticHarnessProvider, ...] = (
    _LazyAgenticHarnessProvider(
        capability="agentic_browsergym",
        namespace_prefix="browsergym/",
        factory="aiperf.accuracy.browsergym:create_browsergym_harness",
        required_versions=_BROWSERGYM_LOCKED_PACKAGE_VERSIONS,
    ),
    _LazyAgenticHarnessProvider(
        capability="agentic_harbor",
        namespace_prefix=None,
        factory="aiperf.accuracy.harbor:create_harbor_harness",
        required_versions=_HARBOR_LOCKED_PACKAGE_VERSIONS,
    ),
)


@dataclass(frozen=True)
class _Registration:
    benchmark_class: str
    grader_class: str
    dataset_repository: str
    dataset_revision: str
    evaluation_splits: tuple[str, ...]
    dataset_subset: str | None = None
    default_n_shots: int = 0
    default_enable_cot: bool = False
    default_system_prompt: str | None = None


_REGISTRATIONS: dict[str, _Registration] = {
    "mmlu": _Registration(
        "aiperf.accuracy.benchmarks.mmlu:MMLUBenchmark",
        "aiperf.accuracy.graders.multiple_choice:MultipleChoiceGrader",
        "lighteval/mmlu",
        "31d46ab06e6934bb0d95f6918668716d1db6f921",
        ("dev", "test"),
        default_n_shots=5,
    ),
    "aime": _Registration(
        "aiperf.accuracy.benchmarks.aime:AIMEBenchmark",
        "aiperf.accuracy.graders.math:MathGrader",
        "Maxwell-Jia/AIME_2024",
        "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        ("train",),
        default_n_shots=8,
        default_enable_cot=True,
        default_system_prompt=(
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    ),
    "hellaswag": _Registration(
        "aiperf.accuracy.benchmarks.hellaswag:HellaSwagBenchmark",
        "aiperf.accuracy.graders.exact_match:ExactMatchGrader",
        "Rowan/hellaswag",
        "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        ("train", "validation"),
        default_n_shots=10,
    ),
    "bigbench": _Registration(
        "aiperf.accuracy.benchmarks.bigbench:BigBenchBenchmark",
        "aiperf.accuracy.graders.exact_match:ExactMatchGrader",
        "lukaemon/bbh",
        "982bb89fd79532a8ac676a61fc42eb1aeec63f99",
        ("test",),
        default_n_shots=3,
        default_enable_cot=True,
    ),
    "aime24": _Registration(
        "aiperf.accuracy.benchmarks.aime24:AIME24Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalExprGrader",
        "HuggingFaceH4/aime_2024",
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        ("train",),
    ),
    "aime25": _Registration(
        "aiperf.accuracy.benchmarks.aime25:AIME25Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalExprGrader",
        "yentinglin/aime_2025",
        "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
        ("train",),
    ),
    "math-500": _Registration(
        "aiperf.accuracy.benchmarks.math_500:Math500Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalLatexGrader",
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        ("test",),
    ),
    "gsm8k": _Registration(
        "aiperf.accuracy.benchmarks.gsm8k:GSM8KBenchmark",
        "aiperf.accuracy.graders.gsm8k_grader:LightevalGSM8KGrader",
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        ("test",),
        dataset_subset="main",
    ),
    "gpqa-diamond": _Registration(
        "aiperf.accuracy.benchmarks.gpqa_diamond:GPQADiamondBenchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalGPQAGrader",
        "Idavidrein/gpqa",
        "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        ("train",),
        dataset_subset="gpqa_diamond",
    ),
    "lcb-codegeneration": _Registration(
        "aiperf.accuracy.benchmarks.lcb_codegeneration:LCBCodeGenerationBenchmark",
        "aiperf.accuracy.graders.code_execution:CodeExecutionGrader",
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        ("test",),
    ),
}

_ALIASES = {
    "mmlu_pro": "mmlu-pro",
    "math_500": "math-500",
    "gpqa_diamond": "gpqa-diamond",
    "lcb_codegeneration": "lcb-codegeneration",
}


@dataclass
class _Problem:
    problem_id: str
    task: str
    prompt: str
    messages: list[dict[str, Any]]
    generation: dict[str, Any]
    ground_truth: str | None = None
    lighteval_doc: Any | None = None


class _ConfigFacade:
    def __init__(self, accuracy: Any, seed: int) -> None:
        self.accuracy = accuracy
        self._dataset = SimpleNamespace(random_seed=seed)

    def get_default_dataset(self) -> Any:
        return self._dataset


class AccuracyWorker:
    """Stateful evaluator serving one loaded benchmark at a time."""

    def __init__(self) -> None:
        self._benchmark: str | None = None
        self._problems: list[_Problem] = []
        self._by_id: dict[str, _Problem] = {}
        self._grader: Any | None = None
        self._uses_lcb_batch_grader = False
        self._lighteval_task: Any | None = None
        self._dataset_identity: dict[str, Any] = {}
        self._agentic: AgenticHarness | None = None

    def hello(self, protocol: int) -> dict[str, Any]:
        if protocol != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol {protocol}; worker requires {PROTOCOL_VERSION}"
            )
        packages = {
            name: _package_version(name)
            for name in (
                "aiperf",
                "lighteval",
                "datasets",
                "deepeval",
                "sympy",
                "latex2sympy2-extended",
                "word2number",
                "harbor",
                "agentlab",
                "browsergym-core",
                "browsergym-experiments",
            )
        }
        capabilities = [
            "load",
            "next_problems",
            "grade_batch",
            "grader_override",
            "shutdown",
        ]
        capabilities.extend(
            [
                "load_agentic",
                "next_episodes",
                "start_episodes",
                "poll_agentic",
                "submit_model_results",
                "cancel_episodes",
                "finish_agentic",
            ]
        )
        available_agentic = [
            provider
            for provider in _AGENTIC_HARNESS_PROVIDERS
            if provider.is_available()
        ]
        if available_agentic:
            capabilities.extend(["agentic", "agentic_inference_gateway"])
            capabilities.extend(provider.capability for provider in available_agentic)
        return {
            "protocol": PROTOCOL_VERSION,
            "worker_version": WORKER_VERSION,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "packages": packages,
            "worker_source_sha256": _source_digest(),
            "dependency_lock_sha256": _dependency_lock_digest(),
            "container_digest": os.getenv("AIPERF_ACCURACY_WORKER_IMAGE_DIGEST"),
            "capabilities": capabilities,
        }

    async def load(self, request: dict[str, Any]) -> dict[str, Any]:
        await self._close_agentic()
        self._benchmark = None
        self._problems = []
        self._by_id = {}
        self._grader = None
        self._uses_lcb_batch_grader = False
        self._lighteval_task = None
        self._dataset_identity = {}
        benchmark = _canonical_benchmark(_required_string(request, "benchmark"))
        config = request.get("config") or {}
        if not isinstance(config, dict):
            raise TypeError("load.config must be an object")
        _verify_locked_environment()
        grader = request.get("grader")
        if grader is not None and (not isinstance(grader, str) or not grader.strip()):
            raise ValueError("load.grader must be a non-empty string or null")
        if benchmark == "mmlu-pro":
            if grader is not None:
                raise ValueError(
                    "MMLU-Pro is graded by its pinned Lighteval task metrics and "
                    "does not accept a grader override"
                )
            await self._load_mmlu_pro(config)
        else:
            await self._load_inherited(benchmark, config, grader)
        self._benchmark = benchmark
        self._by_id = {problem.problem_id: problem for problem in self._problems}
        if len(self._by_id) != len(self._problems):
            raise RuntimeError("benchmark produced duplicate opaque problem IDs")
        return {
            "benchmark": benchmark,
            "problem_count": len(self._problems),
            "dataset": self._dataset_identity,
            "grader": (
                type(self._grader).__name__
                if self._grader is not None
                else "lighteval task metrics"
            ),
        }

    def next_problems(self, offset: int, limit: int) -> dict[str, Any]:
        self._require_loaded()
        if offset < 0 or limit <= 0:
            raise ValueError("next_problems requires offset >= 0 and limit > 0")
        page = self._problems[offset : offset + limit]
        return {
            "items": [
                {
                    "problem_id": problem.problem_id,
                    "task": problem.task,
                    "prompt": problem.prompt,
                    "messages": problem.messages,
                    "generation": problem.generation,
                }
                for problem in page
            ],
            "next_offset": offset + len(page),
            "done": offset + len(page) >= len(self._problems),
        }

    async def grade_batch(self, items: Any) -> dict[str, Any]:
        self._require_loaded()
        if not isinstance(items, list) or not items:
            raise ValueError("grade_batch.items must be a non-empty array")
        submitted: list[tuple[_Problem, str]] = []
        submitted_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("grade_batch item must be an object")
            problem_id = _required_string(item, "problem_id")
            response = item.get("response")
            if not isinstance(response, str):
                raise TypeError(f"response for {problem_id!r} must be a string")
            problem = self._by_id.get(problem_id)
            if problem is None:
                raise KeyError(f"unknown problem_id {problem_id!r}")
            if problem_id in submitted_ids:
                raise ValueError(f"duplicate grade_batch problem_id {problem_id!r}")
            submitted_ids.add(problem_id)
            submitted.append((problem, response))
        if self._benchmark == "lcb-codegeneration" and self._uses_lcb_batch_grader:
            grades = await self._grade_lcb_batch(submitted)
        else:
            grades = [
                await self._grade_one(problem, response)
                for problem, response in submitted
            ]
        results = []
        for (problem, _), grade in zip(submitted, grades, strict=True):
            result = {
                "problem_id": problem.problem_id,
                "task": problem.task,
                "correct": bool(grade["correct"]),
                "unparsed": bool(grade.get("unparsed", False)),
                "confidence": float(grade.get("confidence", 0.0)),
                "reasoning": str(grade.get("reasoning", "")),
                "extracted_answer": grade.get("extracted_answer"),
            }
            results.append(result)
        return {"items": results}

    async def load_agentic(self, request: dict[str, Any]) -> dict[str, Any]:
        """Load one canonical agentic dataset without starting model inference."""
        _reject_unknown_fields(
            request, {"id", "op", "dataset", "model", "config"}, "load_agentic"
        )
        await self._close_agentic()
        self._benchmark = None
        self._problems = []
        self._by_id = {}
        self._grader = None
        self._uses_lcb_batch_grader = False
        self._lighteval_task = None
        self._dataset_identity = {}
        _verify_agentic_environment()
        dataset = _required_string(request, "dataset")
        model = _required_string(request, "model")
        self._agentic = await _create_agentic_harness(
            dataset, model, request.get("config")
        )
        return dict(self._agentic.identity)

    def next_episodes(self, offset: int, limit: int) -> dict[str, Any]:
        """Return one ordered page of opaque agentic task instances."""
        harness = self._require_agentic()
        if offset < 0 or limit <= 0:
            raise ValueError("next_episodes requires offset >= 0 and limit > 0")
        page = harness.episodes[offset : offset + limit]
        return {
            "items": [episode.to_wire() for episode in page],
            "next_offset": offset + len(page),
            "done": offset + len(page) >= len(harness.episodes),
        }

    async def start_episodes(self, authored_ids: Any) -> dict[str, Any]:
        """Begin environment setup for a scheduler-selected episode batch."""
        harness = self._require_agentic()
        episode_ids = _identifier_array(authored_ids, "start_episodes.episode_ids")
        await harness.start_episodes(episode_ids)
        return {"started": episode_ids}

    async def poll_agentic(self, limit: Any, wait_ms: Any) -> dict[str, Any]:
        """Long-poll ready model calls and terminal episode results."""
        harness = self._require_agentic()
        resolved_limit = require_positive_int(limit, "poll_agentic.limit")
        resolved_wait = require_non_negative_int(wait_ms, "poll_agentic.wait_ms")
        events = await harness.poll_events(resolved_limit, resolved_wait)
        return {"events": [event.to_wire() for event in events]}

    async def submit_model_results(self, authored_items: Any) -> dict[str, Any]:
        """Resume evaluator-owned agent loops with Rust inference results."""
        harness = self._require_agentic()
        if not isinstance(authored_items, list) or not authored_items:
            raise ValueError("submit_model_results.items must be a non-empty array")
        items = [AgenticModelResult.from_wire(item) for item in authored_items]
        await harness.submit_model_results(items)
        return {"accepted": [item.call_id for item in items]}

    async def cancel_episodes(self, authored_ids: Any) -> dict[str, Any]:
        """Cancel active evaluator environments selected by the Rust scheduler."""
        harness = self._require_agentic()
        episode_ids = _identifier_array(authored_ids, "cancel_episodes.episode_ids")
        await harness.cancel_episodes(episode_ids)
        return {"cancelled": episode_ids}

    async def finish_agentic(self) -> dict[str, Any]:
        """Return all canonical verifier results in frozen dataset order."""
        harness = self._require_agentic()
        results = await harness.finish()
        return {"items": [result.to_wire() for result in results]}

    async def close(self) -> None:
        """Release an active agent harness before process shutdown."""
        await self._close_agentic()

    async def _close_agentic(self) -> None:
        if self._agentic is not None:
            harness, self._agentic = self._agentic, None
            await harness.close()

    def _require_agentic(self) -> AgenticHarness:
        if self._agentic is None:
            raise RuntimeError("load_agentic must succeed before this operation")
        return self._agentic

    async def _grade_lcb_batch(
        self, submitted: list[tuple[_Problem, str]]
    ) -> list[dict[str, Any]]:
        """Execute one Lighteval pool for the complete submitted LCB batch."""
        if self._grader is None:
            raise RuntimeError("worker has no LiveCodeBench grader state")
        import orjson

        from aiperf.accuracy.graders.code_execution import (
            _build_evaluation_sample,
            _payload_to_test_cases,
            _run_codegen_metrics,
        )

        grades: list[dict[str, Any] | None] = [None] * len(submitted)
        evaluation_samples = []
        generated_code = []
        evaluated_positions = []
        for position, (problem, response) in enumerate(submitted):
            if problem.ground_truth is None:
                raise RuntimeError(
                    f"LiveCodeBench problem {problem.problem_id!r} has no private tests"
                )
            try:
                payload = orjson.loads(problem.ground_truth)
                inputs, outputs, fn_name = _payload_to_test_cases(payload)
            except Exception as error:
                raise RuntimeError(
                    f"failed to decode canonical tests for {problem.problem_id!r}: {error}"
                ) from error
            snippet = self._grader.extract_answer(response)
            if not snippet:
                grades[position] = {
                    "correct": False,
                    "unparsed": True,
                    "confidence": 0.0,
                    "reasoning": "canonical Lighteval LCB: no code block extracted",
                    "extracted_answer": "",
                    "ground_truth": "<lcb test cases>",
                }
                continue
            evaluation_samples.extend(
                _build_evaluation_sample(inputs, outputs, fn_name)
            )
            generated_code.append([snippet])
            evaluated_positions.append((position, snippet))

        if evaluation_samples:
            metrics, _ = await asyncio.to_thread(
                _run_codegen_metrics, evaluation_samples, generated_code
            )
            detail = metrics.get("detail", {}).get("pass@1")
            if not isinstance(detail, dict):
                raise RuntimeError(
                    f"canonical Lighteval LCB omitted detail.pass@1: {metrics!r}"
                )
            for batch_index, (position, snippet) in enumerate(evaluated_positions):
                raw_score = detail.get(batch_index)
                if isinstance(raw_score, bool) or not isinstance(
                    raw_score, int | float
                ):
                    raise RuntimeError(
                        "canonical Lighteval LCB returned non-numeric pass@1 "
                        f"for batch index {batch_index}: {raw_score!r}"
                    )
                score = float(raw_score)
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise RuntimeError(
                        f"canonical Lighteval LCB returned invalid pass@1 {score!r}"
                    )
                grades[position] = {
                    "correct": score >= 1.0,
                    "unparsed": False,
                    "confidence": score,
                    "reasoning": (
                        f"canonical Lighteval LCB batch pass@1={score:.3f} "
                        f"(snippet length={len(snippet)})"
                    ),
                    "extracted_answer": snippet,
                    "ground_truth": "<lcb test cases>",
                }
        if any(grade is None for grade in grades):
            raise RuntimeError("canonical LiveCodeBench batch omitted a grade")
        return [grade for grade in grades if grade is not None]

    async def _load_inherited(
        self, benchmark: str, config: dict[str, Any], grader_override: str | None
    ) -> None:
        registration = _REGISTRATIONS.get(benchmark)
        if registration is None:
            raise ValueError(
                f"unsupported benchmark {benchmark!r}; available: "
                + ", ".join(sorted([*_REGISTRATIONS, "mmlu-pro"]))
            )
        n_shots = config.get("n_shots")
        if n_shots is None:
            n_shots = registration.default_n_shots
        if not isinstance(n_shots, int) or not 0 <= n_shots <= 32:
            raise ValueError("n_shots must be an integer in 0..=32")
        enable_cot = config.get("enable_cot")
        if enable_cot is None:
            enable_cot = registration.default_enable_cot
        system_prompt = config.get("system_prompt")
        if system_prompt is None:
            system_prompt = registration.default_system_prompt
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or null")
        tasks = config.get("tasks")
        if tasks is not None and not (
            isinstance(tasks, list) and all(isinstance(task, str) for task in tasks)
        ):
            raise TypeError("tasks must be an array of strings or null")
        accuracy = SimpleNamespace(
            benchmark=benchmark.replace("-", "_"),
            tasks=tasks,
            n_shots=n_shots,
            enable_cot=bool(enable_cot),
            system_prompt=system_prompt,
            grader=grader_override,
            enabled=True,
        )
        seed = int(config.get("seed", 0))
        run = SimpleNamespace(cfg=_ConfigFacade(accuracy, seed), random_seed=seed)
        benchmark_module_name, benchmark_symbol = registration.benchmark_class.split(
            ":", 1
        )
        benchmark_module = importlib.import_module(benchmark_module_name)
        benchmark_class = getattr(benchmark_module, benchmark_symbol)
        if grader_override is None:
            grader_class = _import_symbol(registration.grader_class)
        else:
            from aiperf.plugin import plugins
            from aiperf.plugin.enums import PluginType

            grader_class = plugins.get_class(
                PluginType.ACCURACY_GRADER, grader_override
            )
        benchmark_instance = benchmark_class(run=run)
        grader_class.check_available()
        self._grader = grader_class(run=run)
        self._uses_lcb_batch_grader = (
            benchmark == "lcb-codegeneration"
            and grader_class.__name__ == "CodeExecutionGrader"
        )
        original_load_dataset = getattr(benchmark_module, "load_dataset", None)
        if original_load_dataset is None:
            raise RuntimeError(
                f"canonical benchmark module {benchmark_module_name} has no load_dataset"
            )
        benchmark_module.load_dataset = (
            _lcb_script_free_dataset_loader(original_load_dataset, registration)
            if benchmark == "lcb-codegeneration"
            else _pinned_dataset_loader(
                original_load_dataset, registration.dataset_revision
            )
        )
        try:
            authored = await benchmark_instance.load_problems(
                tasks=tasks,
                n_shots=n_shots,
                enable_cot=bool(enable_cot),
            )
        finally:
            benchmark_module.load_dataset = original_load_dataset
        max_problems = _optional_positive_int(config, "max_problems")
        if max_problems is not None:
            authored = authored[:max_problems]
        max_tokens_override = _optional_positive_int(config, "max_tokens")
        problems = []
        for index, problem in enumerate(authored):
            messages = list(
                problem.raw_messages or [{"role": "user", "content": problem.prompt}]
            )
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            generation_size = max_tokens_override or int(
                problem.metadata.get("generation_size", 100)
            )
            stop = problem.metadata.get("stop_sequence", [])
            if isinstance(stop, str):
                stop = [stop]
            problem_id = _problem_id(benchmark, index, problem.prompt)
            problems.append(
                _Problem(
                    problem_id=problem_id,
                    task=str(problem.task),
                    prompt=(
                        f"{system_prompt}\n\n{problem.prompt}"
                        if system_prompt
                        else problem.prompt
                    ),
                    messages=messages,
                    generation={
                        "max_tokens": generation_size,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "stop": list(stop),
                    },
                    ground_truth=problem.ground_truth,
                )
            )
        if not problems:
            raise ValueError(f"benchmark {benchmark!r} returned no problems")
        self._problems = problems
        self._lighteval_task = None
        self._dataset_identity = {
            "provider": "aiperf-python benchmark over pinned Hugging Face revision",
            "benchmark": benchmark,
            "repository": registration.dataset_repository,
            "subset": _resolved_subset(benchmark, registration, tasks),
            "revision": registration.dataset_revision,
            "evaluation_splits": list(registration.evaluation_splits),
        }

    async def _load_mmlu_pro(self, config: dict[str, Any]) -> None:
        # Lighteval 0.13's tasks/tasks/mmlu_pro.py sets `instruction=query`.
        # prompt_manager.py then removes that instruction from every few-shot
        # query, yielding empty user turns. Upstream's authored endpoint command
        # uses `mmlu_pro|0`; fail closed until the pinned evaluator fixes the task.
        n_shots = config.get("n_shots")
        if n_shots is None:
            n_shots = 0
        if not isinstance(n_shots, int) or isinstance(n_shots, bool):
            raise ValueError("n_shots must be an integer")
        if n_shots != 0:
            raise ValueError(
                "the pinned Lighteval 0.13.0 MMLU-Pro endpoint task is canonical "
                "only at n_shots=0; its authored Doc sets instruction equal to the "
                "full query, so PromptManager strips nonzero-shot example queries"
            )
        try:
            from lighteval.tasks.lighteval_task import LightevalTask
            from lighteval.tasks.prompt_manager import PromptManager
            from lighteval.tasks.registry import Registry
        except ImportError as error:
            raise RuntimeError(
                "MMLU-Pro requires the pinned Lighteval worker environment"
            ) from error
        if config.get("enable_cot") is False:
            raise ValueError(
                "MMLU-Pro's canonical Lighteval task requires its authored "
                "chain-of-thought prompt; enable_cot=false is unsupported"
            )
        registry = Registry(tasks=f"mmlu_pro|{n_shots}")
        tasks = registry.load_tasks()
        if len(tasks) != 1:
            raise RuntimeError(f"Lighteval resolved {len(tasks)} MMLU-Pro tasks")
        LightevalTask.load_datasets(tasks, dataset_loading_processes=1)
        task = next(iter(tasks.values()))
        categories = config.get("tasks")
        if categories:
            requested = {str(category).strip() for category in categories}
            for split in task.evaluation_split:
                task.dataset[split] = task.dataset[split].filter(
                    lambda row: row.get("category") in requested
                )
        category_by_doc_id = {
            str(index): str(row.get("category", "mmlu_pro"))
            for split in task.evaluation_split
            for index, row in enumerate(task.dataset[split])
        }
        max_problems = _optional_positive_int(config, "max_problems")
        docs = task.get_docs(max_samples=max_problems)
        prompt_manager = PromptManager(
            use_chat_template=False,
            system_prompt=config.get("system_prompt"),
        )
        max_tokens_override = _optional_positive_int(config, "max_tokens")
        problems = []
        for index, doc in enumerate(docs):
            messages = prompt_manager.prepare_prompt_api(doc)
            prompt = prompt_manager.prepare_prompt(doc)
            if not prompt.strip() or any(
                not str(message.get("content", "")).strip() for message in messages
            ):
                raise RuntimeError(
                    "pinned Lighteval produced an empty MMLU-Pro prompt message"
                )
            problem_id = _problem_id("mmlu-pro", index, prompt)
            task_name = category_by_doc_id.get(str(doc.id), "mmlu_pro")
            problems.append(
                _Problem(
                    problem_id=problem_id,
                    task=task_name,
                    prompt=prompt,
                    messages=messages,
                    generation={
                        "max_tokens": max_tokens_override
                        or doc.generation_size
                        or 32768,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "stop": list(doc.stop_sequences or []),
                    },
                    lighteval_doc=doc,
                )
            )
        if not problems:
            raise ValueError("MMLU-Pro returned no problems")
        self._problems = problems
        self._grader = None
        self._uses_lcb_batch_grader = False
        self._lighteval_task = task
        self._dataset_identity = {
            "provider": "lighteval",
            "repository": task.dataset_path,
            "subset": task.dataset_config_name,
            "revision": task.dataset_revision,
            "evaluation_splits": list(task.evaluation_split),
            "task_version": task.version,
        }

    async def _grade_one(self, problem: _Problem, response: str) -> dict[str, Any]:
        if self._lighteval_task is not None:
            from lighteval.models.model_output import ModelResponse

            model_response = ModelResponse(text=[response])
            scores: dict[str, Any] = {}
            for metric in self._lighteval_task.metrics:
                scores.update(
                    metric.compute_sample(
                        doc=problem.lighteval_doc,
                        model_response=model_response,
                    )
                )
            if set(scores) != {"extractive_match"}:
                raise RuntimeError(
                    "pinned MMLU-Pro task returned unexpected sample metrics: "
                    f"{scores!r}"
                )
            raw_score = scores["extractive_match"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise RuntimeError(
                    f"Lighteval extractive_match was not numeric: {raw_score!r}"
                )
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"Lighteval extractive_match was outside [0, 1]: {score!r}"
                )
            return {
                "correct": score > 0.5,
                "unparsed": False,
                "confidence": score,
                "reasoning": f"canonical Lighteval sample metrics: {scores}",
                "extracted_answer": response.strip(),
                "ground_truth": None,
            }
        if self._grader is None or problem.ground_truth is None:
            raise RuntimeError("worker has no grader state for loaded problem")
        grade = await self._grader.grade(response, problem.ground_truth)
        return grade.model_dump()

    def _require_loaded(self) -> None:
        if self._benchmark is None:
            raise RuntimeError("load must succeed before this operation")


async def _dispatch(
    worker: AccuracyWorker, request: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    op = _required_string(request, "op")
    if op == "hello":
        return worker.hello(int(request.get("protocol", -1))), False
    if op == "load":
        return await worker.load(request), False
    if op == "next_problems":
        return worker.next_problems(
            int(request.get("offset", 0)), int(request.get("limit", 0))
        ), False
    if op == "grade_batch":
        return await worker.grade_batch(request.get("items")), False
    if op == "load_agentic":
        return await worker.load_agentic(request), False
    if op == "next_episodes":
        _reject_unknown_fields(
            request, {"id", "op", "offset", "limit"}, "next_episodes"
        )
        return worker.next_episodes(
            int(request.get("offset", 0)), int(request.get("limit", 0))
        ), False
    if op == "start_episodes":
        _reject_unknown_fields(request, {"id", "op", "episode_ids"}, op)
        return await worker.start_episodes(request.get("episode_ids")), False
    if op == "poll_agentic":
        _reject_unknown_fields(request, {"id", "op", "limit", "wait_ms"}, op)
        return await worker.poll_agentic(
            request.get("limit", 0), request.get("wait_ms", 0)
        ), False
    if op == "submit_model_results":
        _reject_unknown_fields(request, {"id", "op", "items"}, op)
        return await worker.submit_model_results(request.get("items")), False
    if op == "cancel_episodes":
        _reject_unknown_fields(request, {"id", "op", "episode_ids"}, op)
        return await worker.cancel_episodes(request.get("episode_ids")), False
    if op == "finish_agentic":
        _reject_unknown_fields(request, {"id", "op"}, op)
        return await worker.finish_agentic(), False
    if op == "shutdown":
        await worker.close()
        return {"shutdown": True}, True
    raise ValueError(f"unknown operation {op!r}")


def serve() -> int:
    """Serve JSONL requests until an explicit shutdown or stdin EOF."""
    protocol_stdout = _reserve_protocol_stdout()
    logging.basicConfig(
        level=os.getenv("AIPERF_ACCURACY_WORKER_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = AccuracyWorker()
    loop = asyncio.new_event_loop()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id: Any = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("protocol request must be an object")
                request_id = request.get("id")
                if not isinstance(request_id, int) or request_id < 0:
                    raise ValueError(
                        "protocol request id must be a non-negative integer"
                    )
                result, shutdown = loop.run_until_complete(_dispatch(worker, request))
                response = {"id": request_id, "ok": True, "result": result}
            except Exception as error:  # return structured operation failures
                _LOG.error("accuracy worker operation failed: %s", error)
                traceback.print_exc(file=sys.stderr)
                response = {
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "kind": type(error).__name__,
                        "message": str(error),
                        "retryable": False,
                    },
                }
                shutdown = False
            protocol_stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            protocol_stdout.flush()
            if shutdown:
                return 0
        return 0
    finally:
        loop.run_until_complete(worker.close())
        loop.close()
        protocol_stdout.close()


def _reserve_protocol_stdout() -> TextIO:
    """Keep one private protocol stream and redirect all ambient stdout to logs."""
    protocol_fd = os.dup(sys.stdout.fileno())
    protocol_stdout = open(  # noqa: SIM115 - serve() owns this process-lifetime stream
        protocol_fd,
        mode="w",
        buffering=1,
        encoding=sys.stdout.encoding or "utf-8",
        errors="backslashreplace",
        closefd=True,
    )
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return protocol_stdout


def _canonical_benchmark(authored: str) -> str:
    normalized = authored.strip().lower().replace("_", "-")
    return _ALIASES.get(normalized, normalized)


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return result.strip()


def _reject_unknown_fields(
    value: dict[str, Any], allowed: set[str], operation: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{operation} has unknown field(s): " + ", ".join(unknown))


def _identifier_array(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty array")
    result = [require_identifier(item, f"{field} item") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate identifiers")
    return result


async def _create_agentic_harness(
    dataset: str, model_name: str, config: Any
) -> AgenticHarness:
    provider = next(
        (
            candidate
            for candidate in _AGENTIC_HARNESS_PROVIDERS
            if candidate.matches(dataset)
        ),
        None,
    )
    if provider is None:
        raise ValueError(f"no canonical agentic harness owns dataset {dataset!r}")
    if not provider.is_available():
        raise RuntimeError(
            f"agentic dataset {dataset!r} requires unavailable pinned capability "
            f"{provider.capability!r}"
        )
    return await provider.create(dataset, model_name, config)


def _optional_positive_int(config: dict[str, Any], field: str) -> int | None:
    value = config.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _import_symbol(spec: str) -> Any:
    module_name, symbol_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


def _pinned_dataset_loader(loader: Any, revision: str) -> Any:
    def load(*args: Any, **kwargs: Any) -> Any:
        authored = kwargs.get("revision")
        if authored is not None and authored != revision:
            raise ValueError(
                f"benchmark requested dataset revision {authored!r}; "
                f"canonical worker requires {revision!r}"
            )
        kwargs["revision"] = revision
        return loader(*args, **kwargs)

    return load


def _lcb_release_files(release: str) -> list[str]:
    """Resolve an LCB config into the JSONL files selected by its pinned script.

    This is the exact ``ALLOWED_FILES`` construction in
    ``livecodebench/code_generation_lite:code_generation_lite.py`` at commit
    ``0fe84c3912ea0c4d4a78037083943e8f0c4dd505``. Lighteval 0.13 requires
    ``datasets>=4``, whose removal of repository loading scripts makes that
    canonical script unexecutable. Selecting the same immutable raw files via
    datasets' built-in JSON loader preserves the script's row order and schema
    without downgrading Lighteval or executing a mutable remote loader.
    """
    if release == "release_latest":
        indices = range(1, _LCB_MAX_RELEASE + 1)
    else:
        cumulative = re.fullmatch(r"release_v([1-6])", release)
        interval = re.fullmatch(r"v([1-6])(?:_v([1-6]))?", release)
        if cumulative:
            indices = range(1, int(cumulative.group(1)) + 1)
        elif interval:
            start = int(interval.group(1))
            end = int(interval.group(2) or start)
            if end < start or (interval.group(2) is not None and end == start):
                raise ValueError(f"unsupported LiveCodeBench release {release!r}")
            indices = range(start, end + 1)
        else:
            raise ValueError(f"unsupported LiveCodeBench release {release!r}")
    return ["test.jsonl" if index == 1 else f"test{index}.jsonl" for index in indices]


def _lcb_script_free_dataset_loader(loader: Any, registration: _Registration) -> Any:
    """Load LCB's pinned raw files with datasets' built-in JSON reader."""

    def load(*args: Any, **kwargs: Any) -> Any:
        if not args or args[0] != registration.dataset_repository:
            raise ValueError(
                "canonical LiveCodeBench loader received an unexpected repository"
            )
        if len(args) != 2 or not isinstance(args[1], str):
            raise ValueError(
                "canonical LiveCodeBench loader requires one release subset"
            )
        if kwargs.get("split") != "test":
            raise ValueError("canonical LiveCodeBench loader requires split='test'")
        authored_revision = kwargs.get("revision")
        if (
            authored_revision is not None
            and authored_revision != registration.dataset_revision
        ):
            raise ValueError(
                f"benchmark requested dataset revision {authored_revision!r}; "
                f"canonical worker requires {registration.dataset_revision!r}"
            )
        release = args[1]
        base = (
            "https://huggingface.co/datasets/"
            f"{registration.dataset_repository}/resolve/{registration.dataset_revision}"
        )
        urls = [f"{base}/{name}" for name in _lcb_release_files(release)]
        return loader("json", data_files={"test": urls}, split="test")

    return load


def _resolved_subset(
    benchmark: str, registration: _Registration, tasks: list[str] | None
) -> str | None:
    if registration.dataset_subset is not None:
        return registration.dataset_subset
    if benchmark in {"mmlu", "bigbench"}:
        return ",".join(tasks) if tasks else "all canonical task subsets"
    if benchmark == "lcb-codegeneration":
        from aiperf.common.environment import Environment

        return Environment.ACCURACY.LCB_RELEASE_TAG
    return None


def _problem_id(benchmark: str, index: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{benchmark}:{index:08d}:{digest}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_digest() -> str:
    """Hash every Python source file that can define evaluator semantics."""
    root = Path(__file__).resolve().parent
    try:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root).as_posix().encode()
            payload = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()
    except OSError:
        return "unavailable"


def _dependency_lock_digest() -> str | None:
    authored = os.getenv("AIPERF_ACCURACY_WORKER_LOCK_SHA256")
    if authored:
        return authored
    if all(
        _package_version(package) == expected
        for package, expected in _BROWSERGYM_LOCKED_PACKAGE_VERSIONS.items()
    ):
        lock_name = "browser-agentic-accuracy-worker.txt"
    elif all(
        _package_version(package) == expected
        for package, expected in _HARBOR_LOCKED_PACKAGE_VERSIONS.items()
    ):
        lock_name = "agentic-accuracy-worker.txt"
    else:
        lock_name = "accuracy-worker.txt"
    lock = Path(__file__).resolve().parents[3] / "requirements" / lock_name
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        return None


def _verify_locked_environment() -> None:
    mismatches = []
    for package, expected in _LOCKED_PACKAGE_VERSIONS.items():
        actual = _package_version(package)
        if actual != expected:
            mismatches.append(f"{package}={actual!r} (expected {expected!r})")
    if mismatches:
        raise RuntimeError(
            "accuracy evaluator environment does not match "
            "requirements/accuracy-worker.txt: " + ", ".join(mismatches)
        )


def _verify_agentic_environment() -> None:
    """Require the static evaluator substrate shared by every agentic provider."""
    _verify_locked_environment()


def main() -> None:
    raise SystemExit(serve())


if __name__ == "__main__":
    main()
