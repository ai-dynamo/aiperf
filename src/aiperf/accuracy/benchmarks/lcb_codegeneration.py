# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveCodeBench code-generation loader, aligned with the trt-llm lighteval reference.

Mirrors lighteval's ``lcb_codegeneration_prompt_fn`` byte-for-byte.
The recipe routes ``lcb:codegeneration`` through lighteval (see
``run_benchmark.py:3409`` — ``acc_dataset in [..., 'lcb:codegeneration']``
sets ``acc_backend='lighteval'``), so the prompt this loader emits
must match what lighteval's prompt manager produces.

Loader and grader pipeline:

- Prompt = ``prepare_prompt(line)`` from lighteval's
  ``tasks/tasks/lcb/main.py``: a fixed instruction followed by the
  ``question_content`` and a python code-block scaffold that's
  starter-code-aware (different scaffolds for "use this starter" vs
  "read from stdin").
- Ground truth = orjson-serialized public + private test cases plus
  the upstream ``metadata`` (so ``CodeExecutionGrader`` has
  everything it needs at grade time without re-loading the dataset).
- Pair with ``CodeExecutionGrader`` (the new lighteval-backed grader
  introduced in the lighteval foundation commit on branch 874). The
  grader extracts the model's code block via lighteval's
  ``extract_code``, then runs it via lighteval's sandboxed
  ``codegen_metrics`` against the test cases.

Reference:
    lighteval/tasks/tasks/lcb/main.py:lcb_codegeneration_prompt_fn
    trt-llm-benchmark-recipe/run_benchmark.py:3409 (lighteval routing)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import orjson
from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "livecodebench/code_generation_lite"
TASK_NAME = "lcb_codegeneration"

# Lighteval's LCB tasks use the model's full reasoning budget;
# generations can be hundreds of lines for hard problems.
DEFAULT_GENERATION_SIZE = 32768

# Schema field names for livecodebench/code_generation_lite (lighteval
# canonical). The recipe and lighteval both use these exact names.
QUESTION_ID_FIELD = "question_id"
QUESTION_TITLE_FIELD = "question_title"
QUESTION_CONTENT_FIELD = "question_content"
STARTER_CODE_FIELD = "starter_code"
DIFFICULTY_FIELD = "difficulty"
PLATFORM_FIELD = "platform"
PUBLIC_TESTS_FIELD = "public_test_cases"
PRIVATE_TESTS_FIELD = "private_test_cases"
EXTRA_METADATA_FIELD = "metadata"

# Fixed leading instruction from lighteval ``prepare_prompt``. We
# inline it (instead of importing from lighteval) so the prompt format
# stays correct even when lighteval isn't installed and we're only
# loading data — and so the byte-equality is auditable in this file.
_PREAMBLE = (
    "You will be given a question (problem specification) and will "
    "generate a correct Python program that matches the specification "
    "and passes all tests.\n\n"
)
_STARTER_INSTRUCTIONS = (
    "You will use the following starter code to write the solution to "
    "the problem and enclose your code within delimiters.\n"
)
_STDIN_INSTRUCTIONS = (
    "Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). Enclose "
    "your code within delimiters as follows. Ensure that when the "
    "python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT.\n"
)
_STDIN_SCAFFOLD = "```python\n# YOUR CODE HERE\n```\n\n"


def _prepare_prompt(row: dict[str, Any]) -> str:
    """Render the LCB prompt byte-equal to lighteval's ``prepare_prompt``."""
    question_content = row.get(QUESTION_CONTENT_FIELD, "")
    starter_code = row.get(STARTER_CODE_FIELD)
    query = _PREAMBLE
    query += f"Question: {question_content}\n\n"
    if starter_code:
        query += _STARTER_INSTRUCTIONS
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += _STDIN_INSTRUCTIONS
        query += _STDIN_SCAFFOLD
    return query


class LCBCodeGenerationBenchmark(AIPerfLoggerMixin):
    """LiveCodeBench code-generation lighteval-aligned benchmark loader.

    Loads ``livecodebench/code_generation_lite`` (test split) and
    emits prompts byte-equal to lighteval's
    ``lcb_codegeneration_prompt_fn``. Pair with
    ``CodeExecutionGrader`` (which itself wraps lighteval's
    ``codegen_metrics`` for sandboxed pass@1 grading).
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load LCB problems lighteval-style.

        Args:
            tasks: Ignored — the lighteval reference doesn't filter
                LCB by difficulty (a per-row ``difficulty`` field is
                kept in metadata for post-run reporting).
            n_shots: Ignored — the lighteval reference is zero-shot.
            enable_cot: Ignored — lighteval's prompt scaffold doesn't
                add a CoT trigger; the model's natural response will
                contain reasoning before the code block.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
            ``ground_truth`` is an orjson payload of the four upstream
            fields (``starter_code``, ``public_test_cases``,
            ``private_test_cases``, ``metadata``) — exactly what
            ``CodeExecutionGrader`` consumes at grade time.
        """
        ds: Dataset = await asyncio.to_thread(load_dataset, DATASET_NAME, split="test")
        return await asyncio.to_thread(self._build_problems, ds)

    def _build_problems(self, ds: Dataset) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            prompt = _prepare_prompt(row)
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": prompt}]
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=self._build_ground_truth(row),
                    task=TASK_NAME,
                    metadata={
                        "question_id": row.get(QUESTION_ID_FIELD, ""),
                        "question_title": row.get(QUESTION_TITLE_FIELD, ""),
                        "platform": row.get(PLATFORM_FIELD, ""),
                        "difficulty": (row.get(DIFFICULTY_FIELD) or "").lower(),
                        "generation_size": DEFAULT_GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems

    @staticmethod
    def _build_ground_truth(row: dict[str, Any]) -> str:
        """Serialize the four upstream fields ``CodeExecutionGrader`` needs.

        The grader (``aiperf.accuracy.graders.code_execution``) parses
        this orjson payload at grade time, lifts test cases out, and
        forwards them to lighteval's ``codegen_metrics`` for sandboxed
        execution. We pass the upstream fields through verbatim
        because their internal shape is grader-defined and not owned
        by this loader.
        """
        payload = {
            "starter_code": row.get(STARTER_CODE_FIELD, ""),
            "public_test_cases": row.get(PUBLIC_TESTS_FIELD, ""),
            "private_test_cases": row.get(PRIVATE_TESTS_FIELD, ""),
            "metadata": row.get(EXTRA_METADATA_FIELD, ""),
        }
        return orjson.dumps(payload).decode("utf-8")
