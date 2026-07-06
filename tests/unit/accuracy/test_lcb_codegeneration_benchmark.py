# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``LCBCodeGenerationBenchmark`` after lighteval alignment.

Pins:
1. Prompt is byte-equal to lighteval's ``prepare_prompt`` for both
   the starter-code and stdin scaffolds.
2. ``ground_truth`` is the orjson payload ``CodeExecutionGrader``
   consumes (``starter_code`` / ``public_test_cases`` /
   ``private_test_cases`` / ``metadata``).
3. Per-row ``difficulty`` lower-cased into metadata; ``generation_size``
   matches lighteval's 32768.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.accuracy.benchmarks.lcb_codegeneration import (
    DEFAULT_GENERATION_SIZE,
    TASK_NAME,
    LCBCodeGenerationBenchmark,
    _prepare_prompt,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.LCB_CODEGENERATION},
    )


def _make_row(
    question_id: str = "q1",
    title: str = "Reverse a list",
    content: str = "Given a list, return it reversed.",
    starter: str = "def solve(xs): pass",
    difficulty: str = "easy",
    platform: str = "leetcode",
    public_tests: str = '[{"input": "[1,2]", "output": "[2,1]"}]',
    private_tests: str = '[{"input": "[3]", "output": "[3]"}]',
    extra_metadata: str = "{}",
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_title": title,
        "question_content": content,
        "starter_code": starter,
        "difficulty": difficulty,
        "platform": platform,
        "public_test_cases": public_tests,
        "private_test_cases": private_tests,
        "metadata": extra_metadata,
    }


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


class TestPromptByteEqualWithLighteval:
    """The prompt must match lighteval's ``prepare_prompt`` exactly."""

    def test_starter_code_scaffold(self) -> None:
        row = _make_row(content="The problem.", starter="def f(x): pass")
        prompt = _prepare_prompt(row)
        assert prompt == (
            "You will be given a question (problem specification) and "
            "will generate a correct Python program that matches the "
            "specification and passes all tests.\n\n"
            "Question: The problem.\n\n"
            "You will use the following starter code to write the "
            "solution to the problem and enclose your code within "
            "delimiters.\n"
            "```python\ndef f(x): pass\n```\n\n"
        )

    def test_stdin_scaffold_when_no_starter_code(self) -> None:
        row = _make_row(content="The problem.", starter="")
        prompt = _prepare_prompt(row)
        assert prompt == (
            "You will be given a question (problem specification) and "
            "will generate a correct Python program that matches the "
            "specification and passes all tests.\n\n"
            "Question: The problem.\n\n"
            "Read the inputs from stdin solve the problem and write "
            "the answer to stdout (do not directly test on the sample "
            "inputs). Enclose your code within delimiters as follows. "
            "Ensure that when the python program runs, it reads the "
            "inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )

    def test_stdin_scaffold_when_starter_is_none(self) -> None:
        """``starter_code`` may be ``None`` upstream; treat as empty."""
        row = _make_row(content="Q?")
        row["starter_code"] = None
        prompt = _prepare_prompt(row)
        assert "# YOUR CODE HERE" in prompt
        assert "starter code" not in prompt


class TestGroundTruthIsOrjsonPayload:
    """``CodeExecutionGrader`` parses ground_truth as an orjson payload
    of the four upstream fields. The shape must round-trip cleanly."""

    @pytest.mark.asyncio
    async def test_payload_round_trips(self) -> None:
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        decoded = orjson.loads(problems[0].ground_truth)
        assert decoded["starter_code"] == rows[0]["starter_code"]
        assert decoded["public_test_cases"] == rows[0]["public_test_cases"]
        assert decoded["private_test_cases"] == rows[0]["private_test_cases"]
        assert decoded["metadata"] == rows[0]["metadata"]

    @pytest.mark.asyncio
    async def test_payload_excludes_question_text(self) -> None:
        """Question content goes in the prompt, not the grading payload."""
        rows = [_make_row(content="QUESTION_TEXT")]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "QUESTION_TEXT" not in problems[0].ground_truth


class TestRejectsUnsupportedOverrides:
    """``load_problems`` raises ``NotImplementedError`` (with a
    ``"lcb_codegeneration: "`` prefix per aiperf's validator-gate
    convention) when called with overrides the lighteval reference
    doesn't honour. The previous implementation silently dropped
    these inputs; failing loud is the safer default."""

    @pytest.mark.asyncio
    async def test_tasks_override_raises(self) -> None:
        bench = LCBCodeGenerationBenchmark(run=_make_run())
        with pytest.raises(
            NotImplementedError, match=r"^lcb_codegeneration: .*--accuracy-tasks"
        ):
            await bench.load_problems(tasks=["easy"], n_shots=0, enable_cot=False)

    @pytest.mark.asyncio
    async def test_nonzero_n_shots_raises(self) -> None:
        bench = LCBCodeGenerationBenchmark(run=_make_run())
        with pytest.raises(
            NotImplementedError, match=r"^lcb_codegeneration: .*--accuracy-n-shots"
        ):
            await bench.load_problems(tasks=None, n_shots=5, enable_cot=False)

    @pytest.mark.asyncio
    async def test_enable_cot_true_raises(self) -> None:
        bench = LCBCodeGenerationBenchmark(run=_make_run())
        with pytest.raises(
            NotImplementedError, match=r"^lcb_codegeneration: .*--accuracy-enable-cot"
        ):
            await bench.load_problems(tasks=None, n_shots=0, enable_cot=True)


class TestLoadProblemsCore:
    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(question_id=f"q{i}") for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 3
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_metadata_carries_lcb_fields(self) -> None:
        rows = [
            _make_row(
                question_id="q42",
                title="Title42",
                platform="codeforces",
                difficulty="MEDIUM",
            )
        ]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        meta = problems[0].metadata
        assert meta["question_id"] == "q42"
        assert meta["question_title"] == "Title42"
        assert meta["platform"] == "codeforces"
        # Difficulty is lowercased for stable downstream filtering.
        assert meta["difficulty"] == "medium"
        assert meta["generation_size"] == DEFAULT_GENERATION_SIZE
        assert DEFAULT_GENERATION_SIZE == 32768

    @pytest.mark.asyncio
    async def test_task_name_is_constant(self) -> None:
        rows = [
            _make_row(difficulty="easy"),
            _make_row(difficulty="hard"),
        ]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert all(p.task == TASK_NAME for p in problems)


class TestPinnedDatasetLoad:
    """``_load_pinned_dataset`` passes
    ``version_tag=Environment.ACCURACY.LCB_RELEASE_TAG`` (default
    ``"release_v1"``, overridable via
    ``AIPERF_ACCURACY_LCB_RELEASE_TAG``) so accuracy numbers are
    reproducible across runs, and remaps any underlying failure into
    a ``RuntimeError`` with an actionable hint."""

    @pytest.mark.asyncio
    async def test_load_dataset_called_with_default_release(self) -> None:
        from aiperf.accuracy.benchmarks.lcb_codegeneration import DATASET_NAME
        from aiperf.common.environment import Environment

        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ) as mock_load:
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        # Single call with the pinned config name as the positional
        # ``name`` arg (matches lighteval's ``hf_subset=`` reference
        # usage). A future accidental swap to the script-only
        # ``version_tag=`` kwarg, or to no pin at all, fails the test.
        mock_load.assert_called_once()
        args, kwargs = mock_load.call_args
        assert args == (DATASET_NAME, Environment.ACCURACY.LCB_RELEASE_TAG)
        # ``trust_remote_code=True`` mirrors the lighteval reference's
        # opt-in; required on ``datasets`` v4+ where remote-code
        # execution is no longer the default.
        assert kwargs == {"split": "test", "trust_remote_code": True}

    @pytest.mark.asyncio
    async def test_env_override_changes_release_tag(self, monkeypatch) -> None:
        """The env var ``AIPERF_ACCURACY_LCB_RELEASE_TAG`` overrides the
        default config-name pin without requiring a source edit. The
        loader reads ``Environment.ACCURACY.LCB_RELEASE_TAG`` at call
        time, so monkeypatching the attribute on the singleton is the
        cleanest way to exercise the override path in tests (the
        ``BaseSettings`` itself is read once at module import)."""
        from aiperf.accuracy.benchmarks.lcb_codegeneration import DATASET_NAME
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.ACCURACY, "LCB_RELEASE_TAG", "v6")
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ) as mock_load:
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        args, _ = mock_load.call_args
        assert args == (DATASET_NAME, "v6")

    @pytest.mark.asyncio
    async def test_load_failure_remapped_to_runtime_error(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            side_effect=ValueError("simulated v4 script-loader removal"),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            with pytest.raises(
                RuntimeError, match=r"^lcb_codegeneration: failed to load"
            ) as exc:
                await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        # Original exception preserved via ``__cause__`` so debuggers
        # can still see what actually broke.
        assert isinstance(exc.value.__cause__, ValueError)
        msg = str(exc.value)
        # Surface both likely root causes so users have a next step.
        assert "datasets<4" in msg
        assert "AIPERF_ACCURACY_LCB_RELEASE_TAG" in msg

    @pytest.mark.asyncio
    async def test_v4_datasets_adds_version_hint(self, monkeypatch) -> None:
        """When ``datasets >= 4`` is installed, the load-failure remap
        prepends a version-aware diagnostic naming the installed version
        so operators don't have to guess whether they're hitting the
        v4 script-loader removal."""
        import datasets

        # Force the version check into the v4+ branch regardless of
        # what's actually installed in the test sandbox.
        monkeypatch.setattr(datasets, "__version__", "4.8.4")
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            side_effect=RuntimeError("simulated script-loader removal"),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            with pytest.raises(RuntimeError) as exc:
                await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        msg = str(exc.value)
        assert "Detected datasets==4.8.4" in msg
        assert "dropped script-based dataset support" in msg

    @pytest.mark.asyncio
    async def test_unparseable_datasets_version_omits_hint(self, monkeypatch) -> None:
        """If ``datasets.__version__`` can't be parsed (malformed string,
        attribute missing, etc.) the hint helper degrades to an empty
        string rather than masking the original error with a parser
        crash. Pins the defensive ``except Exception`` behavior."""
        import datasets

        monkeypatch.setattr(datasets, "__version__", "not-a-version")
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            side_effect=RuntimeError("simulated dataset error"),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            with pytest.raises(RuntimeError) as exc:
                await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        msg = str(exc.value)
        assert "Detected datasets==" not in msg
        assert "datasets<4" in msg  # generic suffix still emitted

    @pytest.mark.asyncio
    async def test_v3_datasets_omits_version_hint(self, monkeypatch) -> None:
        """When ``datasets < 4`` is installed, the v4-specific hint is
        absent — the generic remap message still appears so users get
        the actionable next steps."""
        import datasets

        monkeypatch.setattr(datasets, "__version__", "3.6.0")
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            side_effect=RuntimeError("simulated dataset error"),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            with pytest.raises(RuntimeError) as exc:
                await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
        msg = str(exc.value)
        assert "Detected datasets==" not in msg
        # Generic guidance is still surfaced.
        assert "datasets<4" in msg


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_missing_difficulty_treated_as_empty_string(self) -> None:
        row = _make_row()
        row["difficulty"] = None
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset([row]),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].metadata["difficulty"] == ""

    @pytest.mark.asyncio
    async def test_unicode_in_problem_content_preserved(self) -> None:
        rows = [_make_row(content="Compute ∑ᵢ aᵢ in O(n)")]
        with patch(
            "aiperf.accuracy.benchmarks.lcb_codegeneration.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = LCBCodeGenerationBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∑ᵢ aᵢ" in problems[0].prompt
