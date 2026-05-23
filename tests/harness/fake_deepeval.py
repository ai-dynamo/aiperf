# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-in for the ``deepeval.benchmarks.big_bench_hard`` subtree.

The ``[accuracy]`` extras (deepeval, lighteval, torch, transformers, ...) add
roughly 1 GiB to the install footprint. Installing them in the default CI
matrix would dominate setup time, so the unit-test job runs without them and
every test that touches a real-deepeval-only contract is opt-in.

This module re-creates just enough of deepeval's surface for the BigBench
loader tests to exercise their logic against a synthetic but deterministic
prompt template. Tests that pin byte-equality against deepeval's bundled
CoT/shot ``.txt`` files still need the real install and are marked
``@pytest.mark.requires_deepeval``.

Wiring: ``tests/unit/accuracy/conftest.py`` patches the bigbench loader's
module-level deepeval names with these fakes per-test (autouse, function
scope) when the real deepeval isn't importable. We deliberately do *not*
inject into ``sys.modules`` so adjacent tests like HellaSwag's continue
to use their own ``pytest.importorskip("deepeval")`` skip mechanism
without interference.

Three names need to be present:

- ``BigBenchHardTask`` — enum of 27 BBH subtasks. Mirrors the real values
  one-for-one so resolver tests using ``BOOLEAN_EXPRESSIONS`` /
  ``boolean_expressions`` continue to work.
- ``BigBenchHardTemplate.generate_output(input, task, n_shots, enable_cot)``
  — returns a synthetic prompt. The structure is deliberately
  *not* byte-equal to the real upstream output; tests that need that
  contract are marked ``requires_deepeval``. The format does honour
  ``n_shots`` (longer prompt with more shots) and ``enable_cot`` (CoT
  prompts contain "Let's think step by step.") so
  ``test_more_shots_make_longer_prompt`` and similar loader-behavior tests
  pass against the fake.
- ``bbh_confinement_statements_dict`` — task→confinement-string mapping,
  mirrored from upstream so the per-task confinement assertions stay
  meaningful without needing the real install.
"""

from __future__ import annotations

import enum


class BigBenchHardTask(enum.Enum):
    """27 BBH subtasks. Values mirror the real deepeval enum exactly."""

    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    CAUSAL_JUDGEMENT = "causal_judgement"
    DATE_UNDERSTANDING = "date_understanding"
    DISAMBIGUATION_QA = "disambiguation_qa"
    DYCK_LANGUAGES = "dyck_languages"
    FORMAL_FALLACIES = "formal_fallacies"
    GEOMETRIC_SHAPES = "geometric_shapes"
    HYPERBATON = "hyperbaton"
    LOGICAL_DEDUCTION_FIVE_OBJECTS = "logical_deduction_five_objects"
    LOGICAL_DEDUCTION_SEVEN_OBJECTS = "logical_deduction_seven_objects"
    LOGICAL_DEDUCTION_THREE_OBJECTS = "logical_deduction_three_objects"
    MOVIE_RECOMMENDATION = "movie_recommendation"
    MULTISTEP_ARITHMETIC_TWO = "multistep_arithmetic_two"
    NAVIGATE = "navigate"
    OBJECT_COUNTING = "object_counting"
    PENGUINS_IN_A_TABLE = "penguins_in_a_table"
    REASONING_ABOUT_COLORED_OBJECTS = "reasoning_about_colored_objects"
    RUIN_NAMES = "ruin_names"
    SALIENT_TRANSLATION_ERROR_DETECTION = "salient_translation_error_detection"
    SNARKS = "snarks"
    SPORTS_UNDERSTANDING = "sports_understanding"
    TEMPORAL_SEQUENCES = "temporal_sequences"
    TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS = "tracking_shuffled_objects_five_objects"
    TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS = "tracking_shuffled_objects_seven_objects"
    TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS = "tracking_shuffled_objects_three_objects"
    WEB_OF_LIES = "web_of_lies"
    WORD_SORTING = "word_sorting"


# Mirrored verbatim from the real ``bbh_confinement_statements_dict``.
# Stable upstream data; resync if deepeval ever changes a string.
bbh_confinement_statements_dict: dict[BigBenchHardTask, str] = {
    BigBenchHardTask.BOOLEAN_EXPRESSIONS: "\n\nOutput 'True' or 'False'. Full answer not needed.",
    BigBenchHardTask.CAUSAL_JUDGEMENT: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.DATE_UNDERSTANDING: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.DISAMBIGUATION_QA: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.DYCK_LANGUAGES: "\n\nOutput only the sequence of parentheses characters separated by white space. Full answer not needed.",
    BigBenchHardTask.FORMAL_FALLACIES: "\n\nOutput 'invalid' or 'valid'. Full answer not needed.",
    BigBenchHardTask.GEOMETRIC_SHAPES: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', or '(K)'. Full answer not needed.",
    BigBenchHardTask.HYPERBATON: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.MOVIE_RECOMMENDATION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.NAVIGATE: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.OBJECT_COUNTING: "\n\nOutput the numerical answer. Full answer not needed.",
    BigBenchHardTask.PENGUINS_IN_A_TABLE: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', or '(R)'. Full answer not needed.",
    BigBenchHardTask.RUIN_NAMES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.SALIENT_TRANSLATION_ERROR_DETECTION: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', or '(F)'. Full answer not needed.",
    BigBenchHardTask.SNARKS: "\n\nOutput '(A)' or'(B)'. Full answer not needed.",
    BigBenchHardTask.SPORTS_UNDERSTANDING: "\n\nOutput 'yes' or 'no'. Full answer not needed.",
    BigBenchHardTask.TEMPORAL_SEQUENCES: "\n\nOutput '(A)', '(B)', '(C)', or '(D)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', or '(E)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS: "\n\nOutput '(A)', '(B)', '(C)', '(D)', '(E)', '(F)', or '(G)'. Full answer not needed.",
    BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS: "\n\nOutput '(A)', '(B)', or '(C)'. Full answer not needed.",
    BigBenchHardTask.WEB_OF_LIES: "\n\nOutput 'Yes' or 'No'. Full answer not needed.",
    BigBenchHardTask.WORD_SORTING: "\n\nOutput only the sequence of words separated by white space. Full answer not needed.",
}


class BigBenchHardTemplate:
    """Synthetic stand-in for ``deepeval``'s prompt template.

    Output structure is deliberately *not* byte-equal to upstream. Tests
    that need byte-equality are tagged ``requires_deepeval`` and skip
    without the real install. The fake honours these contracts so
    loader-behavior tests still pass:

    - More ``n_shots`` produces a strictly longer prompt.
    - ``enable_cot=True`` produces a prompt containing ``"step by step"``;
      ``enable_cot=False`` does not.
    - The trailing ``"Q: {input}\\nA: "`` matches the real template
      well enough for the "query is at the end" assertion shape.
    """

    @classmethod
    def generate_output(
        cls,
        input: str,  # noqa: A002 - mirrors upstream kw name
        task: BigBenchHardTask,
        n_shots: int,
        enable_cot: bool,
    ) -> str:
        header = f"Task description: [fake] subtask={task.value}."
        shot_marker = (
            "\n[fake CoT shot] Let's think step by step.\n"
            if enable_cot
            else "\n[fake shot]\n"
        )
        shots = shot_marker * n_shots
        return f"{header}{shots}\n\nQ: {input}\nA: "
