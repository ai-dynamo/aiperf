# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BigBench-Hard benchmark loader, aligned with the trt-llm DeepEval reference.

The trt-llm benchmark recipe routes ``bigbench`` through DeepEval's
``deepeval.benchmarks.BigBenchHard`` class
(``trt-llm-benchmark-recipe/src/tools/acc_benchmark.py:338-356``). This
loader produces prompts byte-equal to what DeepEval's
``BigBenchHardTemplate.generate_output`` produces, by importing and
calling that template directly. The 27 canonical CoT and non-CoT
prompt files (one per BBH subtask) ship inside DeepEval as package
data — DeepEval's template reads them via ``importlib.resources`` at
load time.

Pair with ``ExactMatchGrader`` for strict ``pred.strip() ==
gold.strip()`` semantics matching DeepEval's
``Scorer.exact_match_score``.

Reference:
    deepeval/benchmarks/big_bench_hard/big_bench_hard.py
    deepeval/benchmarks/big_bench_hard/template.py
    deepeval/benchmarks/big_bench_hard/cot_prompts/*.txt (27 files)
    deepeval/benchmarks/big_bench_hard/shot_prompts/*.txt (27 files)
    trt-llm-benchmark-recipe/src/tools/acc_benchmark.py:338
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

try:
    from deepeval.benchmarks.big_bench_hard.big_bench_hard import (
        bbh_confinement_statements_dict,
    )
    from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
    from deepeval.benchmarks.big_bench_hard.template import (
        BigBenchHardTemplate,
    )

    _HAS_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only without optional dep
    _HAS_DEEPEVAL = False
    BigBenchHardTask = None  # type: ignore[assignment]
    BigBenchHardTemplate = None  # type: ignore[assignment]
    bbh_confinement_statements_dict = None  # type: ignore[assignment]


_MISSING_DEEPEVAL_HINT = (
    "deepeval is not installed; BigBench-Hard's prompt templates and "
    "the per-task confinement dict (the trt-llm reference) cannot be "
    "loaded. Install with: uv pip install 'aiperf[accuracy]'."
)

DATASET_NAME = "lukaemon/bbh"
TASK_NAME = "bigbench"

# DeepEval's BigBenchHard caps n_shots at 3 (the canonical CoT files
# only contain 3 worked examples each). We mirror both bounds.
DEFAULT_N_SHOTS = 3
MAX_N_SHOTS = 3

# DeepEval's BigBenchHard default is ``enable_cot=True``.
DEFAULT_ENABLE_COT = True

# CoT solutions can run several hundred tokens; non-CoT answers are
# typically a single bare token. 1024 covers both with headroom.
DEFAULT_GENERATION_SIZE = 1024

# Schema field names in lukaemon/bbh.
INPUT_FIELD = "input"
TARGET_FIELD = "target"


def _resolve_tasks(tasks: list[str] | None) -> list[Any]:
    """Convert ``--accuracy-tasks`` strings to ``BigBenchHardTask`` enums.

    DeepEval evaluates one task at a time. Aiperf accepts either:
      - ``None`` / empty / ``["all"]`` (case-insensitive) → every
        BigBenchHardTask enum (27 subtasks).
      - Lower-snake-case strings matching the enum's ``value``
        (e.g. ``"boolean_expressions"``).
      - Upper-snake-case enum names (e.g. ``"BOOLEAN_EXPRESSIONS"``)
        for parity with the recipe's ``getattr(BigBenchHardTask,
        task_name.upper(), None)`` lookup.

    Mixing ``"all"`` with other task names is rejected so typos like
    ``["all", "NOT_A_TASK"]`` don't silently bypass validation — that
    used to slip through and return every task while swallowing the
    invalid entry (the parallel HellaSwag bug fixed in AIP-877).

    Unknown names raise ``ValueError`` with the full valid list so
    typos fail loudly.
    """
    if not tasks:
        return list(BigBenchHardTask)
    lowered = [t.lower() for t in tasks]
    if "all" in lowered:
        if lowered == ["all"]:
            return list(BigBenchHardTask)
        raise ValueError(
            "'all' cannot be mixed with other task names. Pass 'all' "
            "by itself (or omit --accuracy-tasks) to select every BBH "
            f"subtask, or list specific subtasks. Got: {tasks!r}"
        )
    valid_values = {t.value for t in BigBenchHardTask}
    resolved: list[Any] = []
    unknown: list[str] = []
    for name in tasks:
        if name in valid_values:
            resolved.append(next(t for t in BigBenchHardTask if t.value == name))
            continue
        enum_member = getattr(BigBenchHardTask, name.upper(), None)
        if enum_member is not None:
            resolved.append(enum_member)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"Unknown BBH subtask(s): {unknown}. Valid subtasks: {sorted(valid_values)}"
        )
    return resolved


class BigBenchBenchmark(AIPerfLoggerMixin):
    """BigBench-Hard benchmark loader, byte-equal to DeepEval's prompts.

    Iterates the requested BBH subtasks and renders each problem's
    prompt via ``BigBenchHardTemplate.generate_output`` (which reads
    DeepEval's bundled CoT/shot prompt files). Pair with
    ``ExactMatchGrader`` for the recipe's strict equality scoring.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _HAS_DEEPEVAL:
            raise RuntimeError(_MISSING_DEEPEVAL_HINT)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load BBH problems and format them DeepEval-style.

        Args:
            tasks: Subtask names (lower-snake-case enum values like
                ``boolean_expressions`` or upper-snake-case enum names
                like ``BOOLEAN_EXPRESSIONS``). ``None`` / ``["all"]``
                selects every subtask. Unknown names raise.
            n_shots: 0..3 (DeepEval asserts ``n_shots <= 3`` because
                the canonical prompt files ship exactly 3 examples).
            enable_cot: When True (the DeepEval default), use the
                bundled CoT prompt files; when False, use the non-CoT
                ``shot_prompts/`` files.

        Returns:
            One ``BenchmarkProblem`` per row across all selected
            subtasks. ``task`` is the subtask name so results
            aggregate per-subtask.
        """
        if n_shots > MAX_N_SHOTS:
            raise ValueError(
                f"BBH supports at most {MAX_N_SHOTS} few-shot examples "
                f"(got {n_shots}); DeepEval asserts ``n_shots <= 3`` "
                f"because the canonical prompt files ship exactly "
                f"{MAX_N_SHOTS} worked examples per subtask."
            )
        task_enums = _resolve_tasks(tasks)
        problems: list[BenchmarkProblem] = []
        for task in task_enums:
            ds: Dataset = await asyncio.to_thread(
                load_dataset, DATASET_NAME, task.value
            )
            sub_problems = await asyncio.to_thread(
                self._build_subtask_problems,
                ds["test"],
                task,
                n_shots,
                enable_cot,
            )
            problems.extend(sub_problems)
        return problems

    def _build_subtask_problems(
        self,
        ds: Any,
        task: Any,
        n_shots: int,
        enable_cot: bool,
    ) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            template_prompt = BigBenchHardTemplate.generate_output(
                input=row[INPUT_FIELD],
                task=task,
                n_shots=n_shots,
                enable_cot=enable_cot,
            )
            prompt = f"{template_prompt}{bbh_confinement_statements_dict[task]}"
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": prompt}]
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    # ``BenchmarkProblem.ground_truth`` is typed ``str`` in
                    # strict mode; the upstream BBH schema stores targets
                    # as strings today, but coerce defensively so a future
                    # numeric column doesn't break the loader. Mirrors
                    # DeepEval's ``str(expected_output)`` in its grader.
                    ground_truth=str(row[TARGET_FIELD]),
                    task=task.value,
                    metadata={
                        "bbh_task": task.value,
                        "confinement": bbh_confinement_statements_dict.get(task, ""),
                        "generation_size": DEFAULT_GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems
