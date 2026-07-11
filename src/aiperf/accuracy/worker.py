# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived stdio accuracy evaluator for the native Rust AIPerf frontend.

Rust owns inference I/O and sends completed response text here only after the
normal transport reaches a terminal state. This process owns benchmark loading,
prompt construction, hidden test material, and canonical Python/Lighteval
grading. stdin/stdout are a versioned JSONL protocol; logs use stderr only.

The inherited benchmark and grader implementations are intentionally reused in
this process. Their complete ownership path is documented in
``accuracy_dataset_loader.py:21-150`` and
``accuracy_record_processor.py:21-147``. MMLU-Pro delegates dataset, few-shot,
prompt, and metric construction to the pinned Lighteval task registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

PROTOCOL_VERSION = 1
WORKER_VERSION = "1.0.0"
_LOG = logging.getLogger("aiperf.accuracy.worker")


@dataclass(frozen=True)
class _Registration:
    benchmark_class: str
    grader_class: str
    default_n_shots: int = 0
    default_enable_cot: bool = False
    default_system_prompt: str | None = None


_REGISTRATIONS: dict[str, _Registration] = {
    "mmlu": _Registration(
        "aiperf.accuracy.benchmarks.mmlu:MMLUBenchmark",
        "aiperf.accuracy.graders.multiple_choice:MultipleChoiceGrader",
        default_n_shots=5,
    ),
    "aime": _Registration(
        "aiperf.accuracy.benchmarks.aime:AIMEBenchmark",
        "aiperf.accuracy.graders.math:MathGrader",
        default_n_shots=8,
        default_enable_cot=True,
        default_system_prompt=(
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    ),
    "hellaswag": _Registration(
        "aiperf.accuracy.benchmarks.hellaswag:HellaSwagBenchmark",
        "aiperf.accuracy.graders.exact_match:ExactMatchGrader",
        default_n_shots=10,
    ),
    "bigbench": _Registration(
        "aiperf.accuracy.benchmarks.bigbench:BigBenchBenchmark",
        "aiperf.accuracy.graders.exact_match:ExactMatchGrader",
        default_n_shots=3,
        default_enable_cot=True,
    ),
    "aime24": _Registration(
        "aiperf.accuracy.benchmarks.aime24:AIME24Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalExprGrader",
    ),
    "aime25": _Registration(
        "aiperf.accuracy.benchmarks.aime25:AIME25Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalExprGrader",
    ),
    "math-500": _Registration(
        "aiperf.accuracy.benchmarks.math_500:Math500Benchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalLatexGrader",
    ),
    "gsm8k": _Registration(
        "aiperf.accuracy.benchmarks.gsm8k:GSM8KBenchmark",
        "aiperf.accuracy.graders.gsm8k_grader:LightevalGSM8KGrader",
    ),
    "gpqa-diamond": _Registration(
        "aiperf.accuracy.benchmarks.gpqa_diamond:GPQADiamondBenchmark",
        "aiperf.accuracy.graders.lighteval_grader:LightevalGPQAGrader",
    ),
    "lcb-codegeneration": _Registration(
        "aiperf.accuracy.benchmarks.lcb_codegeneration:LCBCodeGenerationBenchmark",
        "aiperf.accuracy.graders.code_execution:CodeExecutionGrader",
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
        self._lighteval_task: Any | None = None
        self._dataset_identity: dict[str, Any] = {}
        self._include_ground_truth = False

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
            )
        }
        return {
            "protocol": PROTOCOL_VERSION,
            "worker_version": WORKER_VERSION,
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "packages": packages,
            "worker_source_sha256": _source_digest(),
            "container_digest": os.getenv("AIPERF_ACCURACY_WORKER_IMAGE_DIGEST"),
            "capabilities": ["load", "next_problems", "grade_batch", "shutdown"],
        }

    async def load(self, request: dict[str, Any]) -> dict[str, Any]:
        benchmark = _canonical_benchmark(_required_string(request, "benchmark"))
        config = request.get("config") or {}
        if not isinstance(config, dict):
            raise TypeError("load.config must be an object")
        self._include_ground_truth = bool(config.get("include_ground_truth", False))
        if benchmark == "mmlu-pro":
            await self._load_mmlu_pro(config)
        else:
            await self._load_inherited(benchmark, config)
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
        results = []
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
            grade = await self._grade_one(problem, response)
            result = {
                "problem_id": problem_id,
                "task": problem.task,
                "correct": bool(grade["correct"]),
                "unparsed": bool(grade.get("unparsed", False)),
                "confidence": float(grade.get("confidence", 0.0)),
                "reasoning": str(grade.get("reasoning", "")),
                "extracted_answer": grade.get("extracted_answer"),
            }
            if self._include_ground_truth:
                result["ground_truth"] = grade.get("ground_truth")
            results.append(result)
        return {"items": results}

    async def _load_inherited(self, benchmark: str, config: dict[str, Any]) -> None:
        registration = _REGISTRATIONS.get(benchmark)
        if registration is None:
            raise ValueError(
                f"unsupported benchmark {benchmark!r}; available: "
                + ", ".join(sorted([*_REGISTRATIONS, "mmlu-pro"]))
            )
        grader_override = config.get("grader")
        if grader_override:
            raise ValueError(
                "grader overrides are disabled for canonical worker runs; "
                "the benchmark's pinned grader must be used"
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
            enabled=True,
        )
        seed = int(config.get("seed", 0))
        run = SimpleNamespace(cfg=_ConfigFacade(accuracy, seed), random_seed=seed)
        benchmark_class = _import_symbol(registration.benchmark_class)
        grader_class = _import_symbol(registration.grader_class)
        benchmark_instance = benchmark_class(run=run)
        grader_class.check_available()
        self._grader = grader_class(run=run)
        authored = await benchmark_instance.load_problems(
            tasks=tasks,
            n_shots=n_shots,
            enable_cot=bool(enable_cot),
        )
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
            "provider": "aiperf-python benchmark plugin",
            "benchmark": benchmark,
            "revision": "resolved by pinned evaluator environment",
        }

    async def _load_mmlu_pro(self, config: dict[str, Any]) -> None:
        try:
            from lighteval.tasks.lighteval_task import LightevalTask
            from lighteval.tasks.prompt_manager import PromptManager
            from lighteval.tasks.registry import Registry
        except ImportError as error:
            raise RuntimeError(
                "MMLU-Pro requires the pinned Lighteval worker environment"
            ) from error
        n_shots = config.get("n_shots")
        if n_shots is None:
            n_shots = 5
        if not isinstance(n_shots, int) or not 0 <= n_shots <= 32:
            raise ValueError("n_shots must be an integer in 0..=32")
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
            problem_id = _problem_id("mmlu-pro", index, prompt)
            task_name = (
                str(doc.specific.get("category", "mmlu_pro"))
                if doc.specific
                else "mmlu_pro"
            )
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
            numeric_scores = [
                float(value)
                for value in scores.values()
                if isinstance(value, int | float)
            ]
            if not numeric_scores:
                raise RuntimeError(f"Lighteval returned no numeric score: {scores!r}")
            score = max(numeric_scores)
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
    if op == "shutdown":
        return {"shutdown": True}, True
    raise ValueError(f"unknown operation {op!r}")


def serve() -> int:
    """Serve JSONL requests until an explicit shutdown or stdin EOF."""
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
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            if shutdown:
                return 0
        return 0
    finally:
        loop.close()


def _canonical_benchmark(authored: str) -> str:
    normalized = authored.strip().lower().replace("_", "-")
    return _ALIASES.get(normalized, normalized)


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return result.strip()


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


def _problem_id(benchmark: str, index: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{benchmark}:{index:08d}:{digest}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_digest() -> str:
    try:
        with open(__file__, "rb") as source:
            return hashlib.sha256(source.read()).hexdigest()
    except OSError:
        return "unavailable"


def main() -> None:
    raise SystemExit(serve())


if __name__ == "__main__":
    main()
