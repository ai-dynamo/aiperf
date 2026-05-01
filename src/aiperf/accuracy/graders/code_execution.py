# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-execution grader for LiveCodeBench, backed by lighteval's CodegenMetric.

In the trt-llm benchmark recipe, ``lcb:codegeneration`` is graded by
lighteval's ``lcb_codegen_metric``
(``lighteval/tasks/tasks/lcb/main.py``). That metric extracts code
from the model response, executes it inside a sandboxed subprocess
against the public + private test cases, and reports pass@1.

Wiring into aiperf:
    Aiperf's LCB loader serializes ``starter_code`` /
    ``public_test_cases`` / ``private_test_cases`` / upstream
    ``metadata`` into the ``BenchmarkProblem.ground_truth`` field as
    an orjson payload (see ``benchmarks/lcb_codegeneration.py``). This
    grader parses that payload, builds the ``Doc.specific`` shape
    lighteval's ``CodegenMetric`` expects, and forwards to
    ``CodegenMetric.compute`` — which then forks subprocesses to run
    the generated code in a sandbox.

Process model:
    ``codegen_metrics`` uses a ``ProcessPoolExecutor`` with
    ``num_process_evaluate=8`` workers under the hood. Those forks
    happen inside the record-processor service, not in aiperf's HTTP
    worker pool. Per-grade fork latency is ~50–200 ms on Linux plus
    the actual code-execution time (capped by ``timeout=6``).

Reference: ``lighteval/tasks/tasks/lcb/main.py:CodegenMetric``
"""

from __future__ import annotations

import json
import logging
from typing import Any

import orjson

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig

_log = logging.getLogger(__name__)

try:
    from lighteval.tasks.tasks.lcb.codegen_metrics import (
        codegen_metrics,
        extract_code,
    )

    _HAS_LIGHTEVAL_LCB = True
except ImportError:  # pragma: no cover
    _HAS_LIGHTEVAL_LCB = False
    codegen_metrics = None  # type: ignore[assignment]
    extract_code = None  # type: ignore[assignment]


_MISSING_LIGHTEVAL_HINT = (
    "lighteval is not installed; the code_execution grader cannot "
    "run. Install with: uv pip install 'aiperf[accuracy]'."
)

# Lighteval's CodegenMetric defaults — reproduced here so the grader
# behaves identically to the recipe.
_LCB_PASS_AT_K = (1,)
_LCB_NUM_PROCESSES = 8


class CodeExecutionGrader(BaseGrader):
    """Grades code-generation responses by sandboxed execution via lighteval.

    Pairs with ``LCBCodeGenerationBenchmark``. Expects
    ``ground_truth`` to be the orjson payload produced by that loader's
    ``_build_ground_truth``: a JSON object with ``starter_code``,
    ``public_test_cases``, ``private_test_cases``, and ``metadata``
    fields. We rebuild the lighteval-shaped sample
    (``inputs``/``outputs``/``fn_name``) from the payload and call
    lighteval's ``codegen_metrics`` with a 6-second per-test timeout.

    Returns ``correct=True`` when pass@1 == 1.0 (every test case
    passed), else ``correct=False``. ``unparsed=True`` when the
    grader couldn't extract code from the response or lighteval
    raised an exception during execution.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)
        if not _HAS_LIGHTEVAL_LCB:
            raise RuntimeError(_MISSING_LIGHTEVAL_HINT)

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Return the code block lighteval would extract from the response.

        ``extract_code`` looks for the last ```python ... ``` block in
        the response and returns its body, or empty string when no
        block is present.
        """
        try:
            return str(extract_code(response_text) or "")
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("extract_code raised: %s", exc, exc_info=True)
            return ""

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        try:
            payload = orjson.loads(ground_truth)
        except orjson.JSONDecodeError as exc:
            _log.debug("LCB ground_truth payload not JSON: %s", exc)
            return _grading_failure(
                response_text, ground_truth, "ground_truth not orjson"
            )

        try:
            inputs, outputs, fn_name = _payload_to_test_cases(payload)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            _log.debug("LCB test-case payload malformed: %s", exc)
            return _grading_failure(
                response_text, ground_truth, f"malformed test cases: {exc}"
            )

        # Build lighteval's expected sample shape (one sample = one
        # problem with all its test cases bundled into a JSON string).
        evaluation_sample = [
            {
                "input_output": json.dumps(
                    {
                        "inputs": inputs,
                        "outputs": outputs,
                        "fn_name": fn_name,
                    }
                )
            }
        ]
        # ``generations`` is a list-of-lists: one sample, with one
        # candidate generation. Wrap the response in extract_code so
        # we send only the code block to the sandbox.
        snippet = self.extract_answer(response_text)
        generated_code = [[snippet]]

        try:
            metrics, _ = codegen_metrics(
                evaluation_sample,
                generated_code,
                k_list=list(_LCB_PASS_AT_K),
                num_process_evaluate=_LCB_NUM_PROCESSES,
            )
        except Exception as exc:
            _log.debug("lighteval codegen_metrics raised: %s", exc, exc_info=True)
            return _grading_failure(
                response_text, ground_truth, f"sandboxed exec failed: {exc}"
            )

        pass_at_1 = float(metrics.get("pass@1", [0.0])[0])
        correct = pass_at_1 >= 1.0
        return GradingResult(
            correct=correct,
            unparsed=not snippet,
            confidence=pass_at_1,
            reasoning=(
                f"lighteval codegen_metrics pass@1={pass_at_1:.3f} "
                f"(snippet length={len(snippet)})"
                + ("" if snippet else " (no code block extracted)")
            ),
            extracted_answer=snippet,
            ground_truth="<lcb test cases>",
        )


def _payload_to_test_cases(
    payload: dict[str, Any],
) -> tuple[list[Any], list[Any], str | None]:
    """Convert aiperf's stored LCB ground_truth payload to lighteval inputs.

    The recipe's ``lcb_codegeneration_prompt_fn`` (in lighteval) does
    almost the same transform: parse public + private test cases out
    of the row, concatenate ``inputs`` and ``outputs`` lists, pull
    ``fn_name`` from ``metadata.func_name``. We replicate that here.
    """
    public_raw = payload.get("public_test_cases", "")
    private_raw = payload.get("private_test_cases", "")
    metadata_raw = payload.get("metadata", "")

    public_cases = json.loads(public_raw) if public_raw else []
    private_cases = json.loads(private_raw) if private_raw else []
    all_cases = public_cases + private_cases
    inputs = [tc["input"] for tc in all_cases]
    outputs = [tc["output"] for tc in all_cases]

    fn_name: str | None = None
    if metadata_raw:
        meta = (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        )
        if isinstance(meta, dict):
            fn_name = meta.get("func_name")

    return inputs, outputs, fn_name


def _grading_failure(
    response_text: str, ground_truth: str, reason: str
) -> GradingResult:
    """Build a failure ``GradingResult`` with the given reason."""
    return GradingResult(
        correct=False,
        unparsed=True,
        confidence=0.0,
        reasoning=f"LCB grader failed: {reason}",
        extracted_answer=response_text.strip()[:200],
        ground_truth="<lcb test cases>",
    )
