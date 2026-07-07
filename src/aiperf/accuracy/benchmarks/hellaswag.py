# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HellaSwag benchmark loader, aligned with the trt-llm DeepEval reference.

The trt-llm benchmark recipe routes ``hellaswag`` through DeepEval's
``deepeval.benchmarks.HellaSwag`` class
(``trt-llm-benchmark-recipe/src/tools/acc_benchmark.py:319-336``). This
loader produces prompts byte-equal to what DeepEval's
``HellaSwagTemplate.generate_output`` produces, by importing and calling
that template directly. Pair with ``ExactMatchGrader`` for the
recipe's ``Scorer.exact_match_score`` semantics (strict
``pred.strip() == gold.strip()``).

DeepEval's prompt format (verbatim):

    The following are multiple choice questions (with answers) are
    sentence completion problems about <activity_label>.

    <ctx_with_choices_and_answer>

    <ctx_with_choices_and_answer>
    ... (n_shots times, drawn one per unique activity_label from train)

    <test_ctx>
    A. <ending_0>
    B. <ending_1>
    C. <ending_2>
    D. <ending_3>
    Answer:

The grammar in line 1 ("questions (with answers) are sentence
completion") is verbatim from DeepEval — we don't fix it; reproducing
the bug is part of reference parity.

Dataset revision policy:
    ``load_dataset("Rowan/hellaswag")`` is intentionally **not** pinned
    to a commit ``revision=``. The trt-llm benchmark recipe also leaves
    this unpinned, and matching the recipe's resolution behavior is
    part of reference parity. If Rowan/hellaswag is re-uploaded or
    rebased upstream, downstream callers should expect the byte-equal
    pin against ``HellaSwagTemplate.generate_output`` to drift in
    lockstep with whatever the recipe would resolve. Pin a SHA here
    only if/when the recipe pins one.

Reference:
    deepeval/benchmarks/hellaswag/hellaswag.py
    deepeval/benchmarks/hellaswag/template.py
    trt-llm-benchmark-recipe/src/tools/acc_benchmark.py:319
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import DatasetDict, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

try:
    from deepeval.benchmarks.hellaswag.task import HellaSwagTask
    from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate

    _HAS_DEEPEVAL = True
except ImportError:  # pragma: no cover - exercised only without optional dep
    _HAS_DEEPEVAL = False
    HellaSwagTask = None  # type: ignore[assignment]
    HellaSwagTemplate = None  # type: ignore[assignment]


# Lowercased-value → enum lookup so the docstring's promise of
# case-insensitive activity_label matching is actually honoured — the
# previous per-call ``name in {t.value for t in HellaSwagTask}`` check
# was case-sensitive and rejected ``"applying sunscreen"``. Built once
# at module load since ``HellaSwagTask`` is immutable.
_LOWER_VALUE_TO_TASK: dict[str, Any] = (
    {t.value.lower(): t for t in HellaSwagTask} if _HAS_DEEPEVAL else {}
)


_MISSING_DEEPEVAL_HINT = (
    "deepeval is not installed; HellaSwag's prompt template (the "
    "trt-llm reference) cannot be rendered. Install with: "
    "uv pip install 'aiperf[accuracy]'."
)

DATASET_NAME = "Rowan/hellaswag"
TASK_NAME = "hellaswag"

# DeepEval's HellaSwag default is ``n_shots=10`` (capped at 15). We
# mirror both bounds so the loader's defaults match the recipe.
DEFAULT_N_SHOTS = 10
MAX_N_SHOTS = 15

# A bare A/B/C/D answer fits in a handful of tokens; matches DeepEval's
# expectation that the model emits just the letter.
DEFAULT_GENERATION_SIZE = 5

# DeepEval's ``Rowan/hellaswag`` schema: ``activity_label`` selects the
# subtask, ``label`` is the integer gold index 0..3.
ACTIVITY_LABEL_FIELD = "activity_label"
LABEL_FIELD = "label"

# DeepEval's default ``confinement_instructions`` string for HellaSwag,
# appended to the prompt when the model doesn't support
# ``model.generate(prompt, schema=MultipleChoiceSchema)`` — i.e. for
# every non-DeepEval-aware OpenAI-compatible endpoint, which is the
# only path aiperf takes. Without it, models emit verbose responses
# like ``"The answer is A."`` and the strict ``ExactMatchGrader``
# (which mirrors ``Scorer.exact_match_score``) under-grades them
# vs DeepEval's reference numbers.
#
# Source: ``deepeval/benchmarks/hellaswag/hellaswag.py`` —
# ``HellaSwag.__init__`` default when ``confinement_instructions=None``.
DEEPEVAL_CONFINEMENT = "Output 'A', 'B', 'C', or 'D'. Full answer not needed."


def _build_unique_activity_label_shots_set(train_set: Any) -> list[dict[str, Any]]:
    """Mirror DeepEval's ``shots_dataset`` construction.

    DeepEval iterates the train split and collects the FIRST row for
    each unique ``activity_label`` value
    (``hellaswag.py:255-261``). We reproduce that exactly.
    """
    shots_set: list[dict[str, Any]] = []
    categories_seen: set[str] = set()
    for data in train_set:
        category = data[ACTIVITY_LABEL_FIELD]
        if category not in categories_seen:
            categories_seen.add(category)
            shots_set.append(data)
    return shots_set


def _resolve_tasks(tasks: list[str] | None) -> list[Any]:
    """Convert ``--accuracy-tasks`` CLI strings to ``HellaSwagTask`` enums.

    DeepEval evaluates one task at a time (see
    ``HellaSwag.evaluate``). Aiperf accepts either:
      - ``None`` / empty / ``["all"]`` (case-insensitive) → every
        HellaSwagTask enum.
      - A list of activity_label strings (case-insensitive,
        space-separated as in the dataset, e.g. ``"Applying sunscreen"``
        or ``"applying sunscreen"``).

    Mixing ``"all"`` with other task names is rejected so typos like
    ``["all", "NOT_A_TASK"]`` don't silently bypass validation — that
    used to slip through and return every task while swallowing the
    invalid entry.

    Unknown tasks raise ``ValueError`` listing the valid set so typos
    fail loudly.
    """
    if not tasks:
        return list(HellaSwagTask)
    lowered = [t.lower() for t in tasks]
    if "all" in lowered:
        if lowered == ["all"]:
            return list(HellaSwagTask)
        raise ValueError(
            "'all' cannot be mixed with other task names. Pass 'all' "
            "by itself (or omit --accuracy-tasks) to select every task, "
            f"or list specific activity labels. Got: {tasks!r}"
        )
    resolved: list[Any] = []
    unknown: list[str] = []
    for name in tasks:
        member = _LOWER_VALUE_TO_TASK.get(name.lower())
        if member is not None:
            resolved.append(member)
            continue
        # Fall back to upper-snake-case enum name, matching the recipe's
        # ``getattr(HellaSwagTask, task_name.upper(), None)`` lookup.
        enum_member = getattr(HellaSwagTask, name.upper(), None)
        if enum_member is not None:
            resolved.append(enum_member)
        else:
            unknown.append(name)
    if unknown:
        valid_values = sorted(t.value for t in HellaSwagTask)
        raise ValueError(
            f"Unknown HellaSwag task(s): {unknown}. Valid task values "
            f"include {valid_values[:5]}... ({len(valid_values)} total). "
            "Pass space-separated activity_label values "
            "(e.g. 'Applying sunscreen') or upper-snake-case enum "
            "names (e.g. 'APPLYING_SUNSCREEN')."
        )
    return resolved


class HellaSwagBenchmark(AIPerfLoggerMixin):
    """HellaSwag benchmark loader, byte-equal to DeepEval's prompts.

    Loads ``Rowan/hellaswag`` (validation split, filtered per-task by
    ``activity_label``). Few-shot examples drawn from the train split
    using DeepEval's "one per unique activity_label" rule. Pair with
    ``ExactMatchGrader`` (strict equality) for grading parity.
    """

    @classmethod
    def check_available(cls) -> None:
        """Raise if deepeval is missing (see grader ``check_available``).

        Called by the main-process preflight so a missing optional dependency
        surfaces as a clean ConfigurationError before any service spawns,
        rather than raising deep in the dataset-manager loader.
        """
        if not _HAS_DEEPEVAL:
            raise RuntimeError(_MISSING_DEEPEVAL_HINT)

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.check_available()
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load HellaSwag problems and format them DeepEval-style.

        Args:
            tasks: Activity-label strings (case-sensitive against the
                ``HellaSwagTask`` enum's ``value``) or upper-snake-case
                enum names. ``None`` / ``["all"]`` selects every
                category. Unknown names raise ``ValueError``.
            n_shots: Few-shot count, capped at ``MAX_N_SHOTS`` (15).
                The recipe's ``DeepEval.HellaSwag`` default is 10.
            enable_cot: Ignored — DeepEval's HellaSwag has no
                chain-of-thought variant. Accepting the parameter
                keeps the protocol uniform across benchmarks.

        Returns:
            One ``BenchmarkProblem`` per labeled validation row across
            the selected tasks. ``ground_truth`` is a bare ``A``/``B``/
            ``C``/``D`` letter (DeepEval's convention).
        """
        if enable_cot:
            self.info(
                "--accuracy-enable-cot is ignored for HellaSwag "
                "(DeepEval's HellaSwag has no CoT variant)."
            )
        if n_shots > MAX_N_SHOTS:
            raise ValueError(
                f"HellaSwag supports at most {MAX_N_SHOTS} few-shot "
                f"examples (got {n_shots}); DeepEval asserts "
                "``n_shots <= 15``."
            )
        # Validate ``tasks`` BEFORE the HF download: an invalid
        # ``--accuracy-tasks`` value would otherwise trigger a
        # multi-MB ``load_dataset`` call (and potential network/cache
        # failure) just to surface the user's typo.
        selected_tasks = _resolve_tasks(tasks)
        ds: DatasetDict = await asyncio.to_thread(load_dataset, DATASET_NAME)
        return await asyncio.to_thread(
            self._build_problems, ds, selected_tasks, n_shots
        )

    def _build_problems(
        self, ds: DatasetDict, tasks: list[Any], n_shots: int
    ) -> list[BenchmarkProblem]:
        train_set = ds["train"]
        shots_set = _build_unique_activity_label_shots_set(train_set)
        val_set = ds["validation"]
        problems: list[BenchmarkProblem] = []
        choices = ["A", "B", "C", "D"]
        # Pre-bucket validation rows by activity_label so the per-task
        # loop is O(val_rows + tasks) instead of O(tasks × val_rows).
        # With --accuracy-tasks=all (~190 tasks) over the ~10K-row
        # validation split the naive nested scan does ~1.9M dict
        # lookups; one pass over val_set is enough.
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in val_set:
            by_label.setdefault(row.get(ACTIVITY_LABEL_FIELD), []).append(row)
        for task in tasks:
            for row in by_label.get(task.value, ()):
                label_raw = row.get(LABEL_FIELD)
                if label_raw == "" or label_raw is None:
                    continue
                # DeepEval renders the question via the template's
                # ``format_question(include_answer=False)`` to feed
                # ``generate_output`` as ``input``.
                input_text = HellaSwagTemplate.format_question(
                    row, include_answer=False
                )
                template_prompt = HellaSwagTemplate.generate_output(
                    input=input_text,
                    train_set=shots_set,
                    task=task,
                    n_shots=n_shots,
                )
                # Append DeepEval's confinement instruction. DeepEval's
                # ``predict()`` does this when ``model.generate`` doesn't
                # accept a ``schema`` kwarg (the normal case for every
                # OpenAI-compatible endpoint aiperf hits). Without the
                # append, ``ExactMatchGrader`` systematically grades
                # verbose-but-correct responses ("The answer is A.") as
                # wrong vs DeepEval's reference numbers.
                prompt = f"{template_prompt}\n\n{DEEPEVAL_CONFINEMENT}"
                gold_letter = choices[int(label_raw)]
                problems.append(
                    BenchmarkProblem(
                        prompt=prompt,
                        ground_truth=gold_letter,
                        # Per-row task is the activity_label so the
                        # accuracy CSV breaks down per category.
                        task=task.value,
                        metadata={
                            ACTIVITY_LABEL_FIELD: task.value,
                            "generation_size": DEFAULT_GENERATION_SIZE,
                        },
                        raw_messages=self._build_chat_messages(prompt),
                    )
                )
        return problems

    @staticmethod
    def _build_chat_messages(prompt: str) -> list[AccuracyChatMessage]:
        """DeepEval sends the full prompt as a single string, no
        multi-turn chat. Mirror that for both completions and chat
        endpoints by emitting a single user message with the rendered
        prompt."""
        return [{"role": "user", "content": prompt}]
