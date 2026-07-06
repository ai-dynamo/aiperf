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
from aiperf.common.environment import Environment
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


def _datasets_version_hint() -> str:
    """Return a version-aware diagnostic suffix for the load-failure
    remap, naming the installed ``datasets`` version when it crosses
    the v4 cutoff where script-based dataset loaders were removed.

    Returns an empty string when the version is fine or can't be
    determined, so the surrounding error message degrades cleanly.
    """
    try:
        import datasets

        major = int(datasets.__version__.split(".", 1)[0])
    except Exception:  # diagnostic helper for an already-failing load; never mask the original error with a parser crash
        return ""
    if major >= 4:
        return (
            f"Detected datasets=={datasets.__version__} which dropped "
            f"script-based dataset support; LCB's loader is a script. "
        )
    return ""


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
            tasks: Must be ``None``. The lighteval reference doesn't
                filter LCB by difficulty (a per-row ``difficulty``
                field is kept in metadata for post-run reporting), so
                ``--accuracy-tasks`` has no meaningful effect; the
                method raises ``NotImplementedError`` rather than
                silently dropping the user's input.
            n_shots: Must be ``0``. The lighteval reference is
                zero-shot. Any non-zero value raises
                ``NotImplementedError`` so a stray
                ``--accuracy-n-shots`` flag fails loud.
            enable_cot: Must be ``False``. lighteval's prompt scaffold
                has no CoT trigger; ``--accuracy-enable-cot`` would
                silently no-op without this guard.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
            ``ground_truth`` is an orjson payload of the four upstream
            fields (``starter_code``, ``public_test_cases``,
            ``private_test_cases``, ``metadata``) — exactly what
            ``CodeExecutionGrader`` consumes at grade time.

        Raises:
            NotImplementedError: When ``tasks`` is not ``None``,
                ``n_shots != 0``, or ``enable_cot`` is ``True``. The
                error message is prefixed with ``"<TASK_NAME>: "`` per
                aiperf's validator-gate convention.
        """
        if tasks is not None:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-tasks is not supported; the "
                "lighteval reference evaluates the full LCB test split "
                "(difficulty is reported per-row in metadata)."
            )
        if n_shots != 0:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-n-shots != 0 is not supported; "
                "the lighteval reference is zero-shot "
                "(``few_shots_split=None``)."
            )
        if enable_cot:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-enable-cot is not supported; "
                "lighteval's LCB prompt scaffold has no CoT trigger — "
                "the model's natural response carries reasoning before "
                "the code block."
            )
        ds: Dataset = await asyncio.to_thread(self._load_pinned_dataset)
        return await asyncio.to_thread(self._build_problems, ds)

    @staticmethod
    def _load_pinned_dataset() -> Dataset:
        """Load the pinned LCB release, remapping any failure to a
        ``RuntimeError`` with an actionable hint.

        ``load_dataset`` here selects an LCB HuggingFace config (e.g.
        ``"v4_v5"``, ``"v6"``) by passing ``Environment.ACCURACY.LCB_RELEASE_TAG``
        as the **positional ``name`` arg** — the standard HF
        config-name selector — alongside ``trust_remote_code=True`` so
        LCB's dataset-loading script can execute on ``datasets`` v4+
        (which dropped the implicit-trust default). Both mirror
        lighteval's reference (``hf_subset=subset`` +
        ``trust_remote_code=True`` on ``get_dataset_config_names`` +
        ``trust_dataset=True`` on the task config). The release tag is
        overridable via ``AIPERF_ACCURACY_LCB_RELEASE_TAG`` (e.g. when
        the team rebaselines against a newer monthly snapshot) without
        source edits. Failures can still come from a couple of
        independent sources — the HF ``datasets`` library
        no longer recognising the upstream config name, the pinned
        subset being renamed or removed upstream, or the user explicitly
        disabling remote-code execution at the env level — so we don't
        try to enumerate the specific exception class. A broad
        ``except`` keeps the surface small while preserving the
        original cause via ``__cause__``.
        """
        release_tag = Environment.ACCURACY.LCB_RELEASE_TAG
        try:
            # Pass the release as positional ``name`` (the standard HF
            # config-name selector), matching the trt-llm/lighteval
            # reference's ``hf_subset=`` usage. ``trust_remote_code=True``
            # opts in to executing LCB's dataset-loading script — which
            # is what defines the configs and the test-case payload
            # schema — and is required on ``datasets`` v4+ where remote
            # code execution is no longer the default. The reference
            # makes the same opt-in (``get_dataset_config_names(...,
            # trust_remote_code=True)`` plus ``trust_dataset=True`` on
            # the LightevalTaskConfig), so we mirror the contract.
            return load_dataset(
                DATASET_NAME, release_tag, split="test", trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(
                f"{TASK_NAME}: failed to load {DATASET_NAME!r} subset "
                f"{release_tag!r}. "
                f"{_datasets_version_hint()}"
                f"This typically means either "
                f"(a) the installed ``datasets`` package no longer supports "
                f"the LCB config layout (try "
                f"``uv pip install 'datasets<4'`` or use the upstream "
                f"parquet snapshot directly), or (b) the pinned subset "
                f"name was renamed/removed upstream (set "
                f"``AIPERF_ACCURACY_LCB_RELEASE_TAG`` to a current "
                f"subset such as ``v4_v5`` or ``v6``). Original error: "
                f"{type(e).__name__}: {e}"
            ) from e

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
