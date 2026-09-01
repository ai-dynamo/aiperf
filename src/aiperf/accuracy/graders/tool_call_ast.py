# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BFCL tool-call AST grader (Prompt mode).

In BFCL's Prompt mode the tool schemas are injected into the system prompt and
the model replies in plain text with a call list like
``[get_weather(city='SF')]``. That lands in ``message.content``, so this grader
reads the ordinary answer channel and needs no tool-call capture path.

Wiring into aiperf:
    ``BFCLASTBenchmark`` serializes each entry's ``ground_truth`` (BFCL's
    ``possible_answer``), its function docs, its language and its test category
    into ``BenchmarkProblem.ground_truth`` as an orjson payload (see
    ``benchmarks/bfcl_ast.py``). This grader parses that payload, decodes the
    response into BFCL's canonical ``[{"func": {"param": val}}]`` shape, and
    forwards both to bfcl-eval's deterministic ``ast_checker`` - the same
    checker the leaderboard runs, reached through ``_bfcl_compat``.

Two signals, deliberately kept apart:
    ``correct`` answers "was the call right?" - function name, required vs
    optional parameters, strict types, accepted values, with parallel calls
    compared order-independently.

    ``unparsed`` answers "was there a call at all?" - it is set only when the
    response cannot be decoded into a call list. A decoded-but-wrong call is
    ``correct=False, unparsed=False``. That makes the existing per-task
    ``Unparsed`` column a model format-adherence rate in Prompt mode (and, once
    FC mode lands, a serving-layer tool-call-parser reliability rate).

``confidence`` is always 1.0: the AST checker is deterministic, so a verdict
carries no uncertainty. Run-to-run variance in a BFCL run lives in the model's
sampling, not in the grading.

Reference: ``bfcl_eval/eval_checker/ast_eval/ast_checker.py``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.accuracy.graders import _bfcl_compat
from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_log = logging.getLogger(__name__)

#: Categories whose correct behavior is to emit NO call. BFCL calls this
#: hallucination measurement: the offered tools cannot answer the question, so
#: any call is a fabrication.
ABSTAIN_CATEGORIES = frozenset({"irrelevance", "live_irrelevance"})

#: Categories whose correct behavior is to emit SOME call. The mirror image of
#: the abstain set: a relevant tool is available, so refusing is the failure.
RELEVANCE_CATEGORIES = frozenset({"live_relevance"})

#: Model key handed to upstream's checker. This MUST be a key registered in
#: bfcl-eval's ``MODEL_CONFIG_MAPPING``: ``convert_func_name``
#: (``ast_eval/ast_checker.py``) indexes that dict with a bare subscript — not
#: ``.get`` — whenever a gold function name contains a dot, and it is called
#: unconditionally before the name match. An unregistered value therefore
#: raises ``KeyError`` on every dotted-name entry, which is roughly a third of
#: the gradeable dataset (e.g. ``math.factorial`` in ``simple_python_1``).
#: There is no "neutral name" that opts out of model-specific handling.
#:
#: ``gorilla-openfunctions-v2`` is a prompt-mode entry with
#: ``underscore_to_dot=False``, so ``convert_func_name`` returns the gold name
#: unchanged. That identity transform is what aiperf wants: BFCL Prompt mode
#: shows the model the tool schemas verbatim and asks it to echo those names
#: back, dots included, so rewriting them would break the comparison.
#:
#: Validated against the installed wheel in ``check_available`` so an upstream
#: registry change fails in preflight rather than degrading grades silently.
CHECKER_MODEL_NAME = "gorilla-openfunctions-v2"

# Operator-facing buckets, normalized from upstream's "<family>:<detail>"
# error_type strings. Upstream's inventory as of bfcl-eval 2026.3.23:
#   type_error:{simple,nested,java,js}
#   value_error:{others,string,list/tuple,dict_key,dict_value,list_dict_count}
#   simple_function_checker:{wrong_func_name,missing_required,missing_optional,
#                            unexpected_param,wrong_count,unclear}
#   multiple_function_checker:wrong_count
#   parallel_function_checker_enforce_order:wrong_count
#   parallel_function_checker_no_order:{wrong_count,cannot_find_match}
#   dict_checker:unclear / list_dict_checker:unclear
WRONG_TOOL = "wrong_tool"
PARAM_TYPE_ERROR = "param_type_error"
PARAM_VALUE_ERROR = "param_value_error"
SHOULD_NOT_HAVE_CALLED = "should_not_have_called"
SHOULD_HAVE_CALLED = "should_have_called"
UNPARSED = "unparsed"
UNCLASSIFIED = "unclassified"
GRADER_ERROR = "grader_error"

# Buckets keyed on the family (the part before the first colon).
_FAMILY_BUCKETS = {
    "type_error": PARAM_TYPE_ERROR,
    "value_error": PARAM_VALUE_ERROR,
}

# Buckets keyed on the detail (the part after the first colon), for the
# "<something>_checker:<detail>" families where the family names the checker
# that ran rather than the kind of failure.
_DETAIL_BUCKETS = {
    "wrong_func_name": WRONG_TOOL,
    "wrong_count": WRONG_TOOL,
    "cannot_find_match": WRONG_TOOL,
    "missing_required": PARAM_VALUE_ERROR,
    "missing_optional": PARAM_VALUE_ERROR,
    "unexpected_param": PARAM_VALUE_ERROR,
}


def classify_error(error_type: str) -> str:
    """Normalize an upstream ``error_type`` into an operator-facing bucket.

    Production data shows parameter mismatches (right tool, wrong arguments)
    dominate tool-calling failures at scale, so separating ``param_*`` from
    ``wrong_tool`` is what makes the export actionable.

    An unrecognized ``error_type`` maps to ``unclassified`` rather than being
    folded into a neighbouring bucket, so an upstream addition shows up as
    something to look at instead of quietly inflating a real bucket.
    """
    if not error_type:
        return UNCLASSIFIED
    family, _, detail = error_type.partition(":")
    if family in _FAMILY_BUCKETS:
        return _FAMILY_BUCKETS[family]
    if detail in _DETAIL_BUCKETS:
        return _DETAIL_BUCKETS[detail]
    return UNCLASSIFIED


def _describe(bucket: str, result: dict[str, Any]) -> str:
    """Render ``"<bucket>: <error_type>: <errors>"`` for the export."""
    error_type = str(result.get("error_type", ""))
    errors = result.get("error") or []
    if isinstance(errors, list):
        detail = "; ".join(str(e) for e in errors)
    else:
        detail = str(errors)
    return f"{bucket}: {error_type}: {detail}" if detail else f"{bucket}: {error_type}"


def _dumps(value: Any) -> str:
    """Compact orjson rendering for the export's string fields."""
    try:
        return orjson.dumps(value).decode("utf-8")
    except TypeError:  # pragma: no cover - defensive
        return str(value)


def _grader_error(response_text: str, possible_answer: Any) -> GradingResult:
    """Result for a response the checker could not evaluate.

    Deliberately **not** flagged ``unparsed``. The model may well have
    formatted its call perfectly; the failure is on aiperf's side of the
    boundary. Routing it into ``unparsed`` would corrupt the one column this
    benchmark advertises as a model format-adherence rate, pointing operators
    at their model when the fault is in the integration.

    The verdict is still ``correct=False`` — the answer was never verified —
    but it carries its own bucket, and ``_safe_check`` logs at warning level
    so the run itself says something went wrong.
    """
    return GradingResult(
        correct=False,
        unparsed=False,
        confidence=1.0,
        reasoning=(
            f"{GRADER_ERROR}: the AST checker raised on this record, so it "
            f"could not be graded. This is an integration failure, not a model "
            f"failure; see the warning log for the exception."
        ),
        extracted_answer=response_text[:500],
        ground_truth=_dumps(possible_answer)[:500],
    )


def _grading_failure(
    response_text: str, ground_truth: str, reason: str
) -> GradingResult:
    """Result for a response that could not be graded at all.

    Flagged ``unparsed`` because nothing gradeable was recovered - either the
    ground-truth payload was malformed or the response held no call list.
    """
    return GradingResult(
        correct=False,
        unparsed=True,
        confidence=1.0,
        reasoning=f"{UNPARSED}: {reason}",
        extracted_answer=response_text[:500],
        ground_truth=ground_truth[:500],
    )


class ToolCallASTGrader(BaseGrader):
    """Grades BFCL Prompt-mode tool calls with bfcl-eval's AST checker.

    Pairs with ``BFCLASTBenchmark``. Expects ``ground_truth`` to be the orjson
    payload that loader's ``_build_ground_truth`` produces: an object with
    ``possible_answer``, ``function``, ``language`` and ``test_category``.

    Stateless and re-entrant - everything a verdict needs arrives in
    ``(response_text, ground_truth)``, so concurrent records never interact.
    """

    @classmethod
    def check_available(cls) -> None:
        """Raise if bfcl-eval is missing or unusable for grading.

        Also validates ``CHECKER_MODEL_NAME`` against upstream's registry: an
        unregistered key raises inside the checker on every dotted gold
        function name, and the crash guard would turn that into a plausible
        run full of failed records. Preflight is the only place it can still
        be reported as what it is (see ``BaseGrader.check_available``).
        """
        _bfcl_compat.require_bfcl()
        _bfcl_compat.check_checker_model_key(CHECKER_MODEL_NAME)

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(run=run, **kwargs)
        self.check_available()

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Return the decoded call list as compact JSON, or ``""``.

        Never raises: an undecodable response is exactly the case this grader
        reports as ``unparsed``, not an error.
        """
        language = str(kwargs.get("language", "python"))
        try:
            return _dumps(_bfcl_compat.decode_calls(response_text, language))
        except _bfcl_compat.BFCLDecodeError as exc:
            _log.debug("BFCL response not decodable: %s", exc)
            return ""

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        try:
            payload = orjson.loads(ground_truth)
        except orjson.JSONDecodeError as exc:
            _log.debug("BFCL ground_truth payload not JSON: %s", exc)
            return _grading_failure(
                response_text, ground_truth, "ground_truth not orjson"
            )

        try:
            test_category = str(payload["test_category"])
            language = str(payload.get("language", "python"))
            possible_answer = payload.get("possible_answer")
            function_docs = payload["function"]
        except (KeyError, TypeError) as exc:
            _log.debug("BFCL ground_truth payload malformed: %s", exc)
            return _grading_failure(
                response_text, ground_truth, f"malformed ground_truth: {exc}"
            )

        try:
            decoded = _bfcl_compat.decode_calls(response_text, language)
        except _bfcl_compat.BFCLDecodeError as exc:
            decoded = None
            decode_error: str | None = str(exc)
        else:
            decode_error = None

        # Hallucination categories carry no ground truth to match against: the
        # verdict is about whether a call was emitted at all, so they must be
        # answered before the checker is consulted.
        if test_category in ABSTAIN_CATEGORIES or test_category in RELEVANCE_CATEGORIES:
            return self._grade_abstention(
                response_text=response_text,
                test_category=test_category,
                decoded=decoded,
            )

        if decoded is None:
            return _grading_failure(
                response_text, _dumps(possible_answer), f"undecodable: {decode_error}"
            )

        result = self._safe_check(
            function_docs=function_docs,
            decoded=decoded,
            possible_answer=possible_answer,
            language=language,
            test_category=test_category,
        )
        if result is None:
            return _grader_error(response_text, possible_answer)
        correct = bool(result.get("valid"))
        reasoning = (
            "correct"
            if correct
            else _describe(classify_error(str(result.get("error_type", ""))), result)
        )
        return GradingResult(
            correct=correct,
            unparsed=False,
            confidence=1.0,
            reasoning=reasoning,
            extracted_answer=_dumps(decoded),
            ground_truth=_dumps(possible_answer),
        )

    @staticmethod
    def _safe_check(
        *,
        function_docs: Any,
        decoded: list[dict[str, Any]],
        possible_answer: Any,
        language: str,
        test_category: str,
    ) -> dict[str, Any] | None:
        """Run the AST checker with crash-safety.

        Grading happens inside the daemon record processor, where an unhandled
        exception takes down the whole run: the parent then waits forever on
        records that never arrive. Upstream's checker is not defensive about
        malformed input (a missing ``possible_answer`` entry indexes straight
        into ``None``), and it is an optional third-party dependency whose
        internals move between releases, so a raise here is treated the same
        way the lighteval graders treat theirs - report the record as unparsed
        and keep the run alive.

        Returns:
            The checker's verdict, or ``None`` when it raised.
        """
        try:
            return _bfcl_compat.ast_check(
                func_description=function_docs,
                model_output=decoded,
                possible_answer=possible_answer,
                language=language,
                test_category=test_category,
                model_name=CHECKER_MODEL_NAME,
            )
        except Exception as exc:
            # Warning, not debug: this is aiperf's bug to fix, and the graded
            # record alone cannot say so loudly enough.
            _log.warning(
                "bfcl ast_checker raised on category %s (record graded as "
                "%s, not counted against the model's format-adherence rate): %s",
                test_category,
                GRADER_ERROR,
                exc,
                exc_info=True,
            )
            return None

    @staticmethod
    def _grade_abstention(
        *,
        response_text: str,
        test_category: str,
        decoded: list[dict[str, Any]] | None,
    ) -> GradingResult:
        """Grade the hallucination categories, where the verdict is call-or-not.

        For ``irrelevance``/``live_irrelevance`` the offered tools cannot answer
        the question, so the correct response is prose and emitting a call is a
        fabrication. ``live_relevance`` is the mirror image: a usable tool
        exists, so refusing to call is the failure.

        For a response that actually says something, ``unparsed`` stays False in
        both directions: a prose refusal is a real answer here, not a formatting
        failure, and conflating the two would corrupt the format-adherence rate
        the ``Unparsed`` column reports.

        An **empty** answer channel is the exception. It is not an abstention -
        the model said nothing at all, usually because the generation was cut
        off or the content arrived on a reasoning channel - and scoring it
        correct would silently inflate the 240-problem ``irrelevance`` category
        exactly when a run is misconfigured. It is reported as unparsed, the
        same as anywhere else.
        """
        if not response_text or not response_text.strip():
            return _grading_failure(
                response_text,
                "[]" if test_category in ABSTAIN_CATEGORIES else "<any relevant call>",
                "empty answer channel: the model returned nothing, which is not "
                "an abstention. Usually the generation was cut off (max_tokens "
                "too low) or the content arrived on a reasoning channel.",
            )
        emitted_call = bool(decoded)
        should_call = test_category in RELEVANCE_CATEGORIES
        correct = emitted_call is should_call
        if correct:
            reasoning = "correct"
        elif should_call:
            reasoning = f"{SHOULD_HAVE_CALLED}: a relevant function was available but the response emitted no call"
        else:
            reasoning = f"{SHOULD_NOT_HAVE_CALLED}: none of the offered functions can answer this question, but the response emitted a call"
        return GradingResult(
            correct=correct,
            unparsed=False,
            confidence=1.0,
            reasoning=reasoning,
            extracted_answer=_dumps(decoded) if emitted_call else response_text[:500],
            ground_truth="[]" if not should_call else "<any relevant call>",
        )
