# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Berkeley Function Calling Leaderboard (BFCL) loader, Prompt mode.

Loads BFCL's single-turn tool-calling entries and emits them as ordinary chat
problems. Prompt mode means the tool schemas go into the system prompt and the
model answers in plain text with a call list, so the answer arrives in
``message.content`` and grading needs no tool-call capture path - the whole
benchmark is a dataset plus a grader.

Loader and grader pipeline:

- Dataset = the question files bundled inside the installed ``bfcl-eval`` wheel
  (``bfcl_eval/data/BFCL_v4_<category>.json``, JSON-lines) joined by ``id`` to
  their ground truth in ``bfcl_eval/data/possible_answer/``. There is no
  download and no HuggingFace dataset, so the data cannot drift away from the
  checker that scores it - both ship in the same wheel, pinned together by
  ``AIPERF_ACCURACY_BFCL_VERSION_PIN``.
- Prompt = upstream's own ``system_prompt_pre_processing_chat_model``, called
  through ``_bfcl_compat``. BFCL v4 assembles the system prompt per entry from
  a style table rather than keeping one constant, and its format-sensitivity
  results show models are highly sensitive to that exact wording, so the
  template is never reproduced locally.
- Ground truth = an orjson payload of the entry's ``possible_answer``, function
  docs, language and category - everything ``ToolCallASTGrader`` needs at grade
  time without re-reading the dataset.
- ``task`` is the BFCL category, not the benchmark name, so the existing
  per-task console table and ``accuracy_results.csv`` produce the per-category
  accuracy and per-category unparsed breakdown that makes the run actionable.

Scope: the stateless single-turn categories only. ``exec_*`` needs a live API
sandbox, ``multi_turn_*`` is graded by comparing backend state across turns,
and the v4 agentic categories need live web search and persistent memory - none
of which fit a stateless per-record grader. Those raise a clear error rather
than being silently dropped.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.accuracy.graders import _bfcl_compat
from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.path_safety import safe_read_template_path

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.resolution.plan import BenchmarkRun

TASK_NAME = "bfcl_ast"

# Tool-call responses are short, but reasoning models emit a preamble before the
# call list; 4096 leaves room for that without inviting a runaway generation.
DEFAULT_GENERATION_SIZE = 4096

# Mirrors upstream ``NON_LIVE_CATEGORY`` / ``LIVE_CATEGORY``
# (bfcl_eval/constants/category_mapping.py). Duplicated here so the module
# imports - and ``--accuracy-tasks`` validates with a useful message - without
# bfcl-eval installed; ``test_bfcl_ast_parity`` asserts the two stay equal.
NON_LIVE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
LIVE_CATEGORIES = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)

#: Evaluated when ``--accuracy-tasks`` is omitted: the non-live AST categories
#: plus Java/JavaScript plus hallucination measurement. The ``live_*``
#: categories (user-contributed schemas, far larger) are opt-in.
DEFAULT_CATEGORIES = NON_LIVE_CATEGORIES
SUPPORTED_CATEGORIES = NON_LIVE_CATEGORIES + LIVE_CATEGORIES

#: Categories with no ``possible_answer`` file: the verdict is whether a call
#: was emitted at all, so there is no gold call list to load.
_NO_GROUND_TRUTH_CATEGORIES = frozenset(
    {"irrelevance", "live_irrelevance", "live_relevance"}
)

#: Category families upstream ships that aiperf deliberately does not grade,
#: mapped to the reason, so ``--accuracy-tasks multi_turn_base`` explains itself
#: instead of reporting an unknown name.
_OUT_OF_SCOPE_PREFIXES = {
    "exec_": "executable categories require running generated code against live "
    "external APIs (and at least four API keys); aiperf has no execution sandbox "
    "on the accuracy path",
    "multi_turn_": "multi-turn categories are graded by comparing backend system "
    "state across turns; aiperf's grader is stateless and per-record",
    "web_search": "the v4 web-search categories require live web search (SerpAPI) "
    "and multi-run trajectories",
    "memory": "the v4 memory categories require persistent state across runs",
    "format_sensitivity": "format_sensitivity is a non-scoring category upstream "
    "(it varies the prompt template rather than measuring correctness)",
}

# Language is not a field on BFCL entries - it is implied by the category.
_JAVA_CATEGORIES = frozenset({"simple_java"})
_JAVASCRIPT_CATEGORIES = frozenset({"simple_javascript"})


def _language_for(category: str) -> str:
    """Return the AST language a category is checked in."""
    if category in _JAVA_CATEGORIES:
        return "java"
    if category in _JAVASCRIPT_CATEGORIES:
        return "javascript"
    return "python"


def _read_jsonl(path: Path, what: str) -> list[dict[str, Any]]:
    """Parse one of BFCL's bundled JSON-lines files.

    Args:
        path: The bundled file.
        what: Human-readable description used in the error message.

    Raises:
        RuntimeError: when the file is unreadable or a line is not valid JSON.
            Both mean the installed wheel is not what we expect, which must fail
            loudly rather than silently yield a short problem set.
    """
    # Canonicalize before the safety read. ``safe_read_template_path`` refuses
    # any path with a symlinked component - the right rule for a user-supplied
    # template path, which must not traverse a symlink somewhere unexpected. But
    # this path is derived from the installed package's own location, and
    # site-packages is routinely reached through a symlink (pyenv, conda, a
    # symlinked venv, /tmp on macOS). Resolving first is the canonicalization
    # that check is trying to obtain, and it keeps the remaining guarantees
    # (existing regular file, explicit utf-8 decode) intact.
    try:
        canonical = path.resolve()
    except (OSError, RuntimeError, ValueError) as e:
        raise RuntimeError(
            f"{TASK_NAME}: cannot resolve {what} at {path}. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    contents = safe_read_template_path(str(canonical))
    if contents is None:
        raise RuntimeError(
            f"{TASK_NAME}: cannot read {what} at {canonical}. It ships inside "
            f"the bfcl-eval wheel, so this usually means a broken or partial "
            f"install - reinstall with ``uv pip install 'aiperf[bfcl]'``."
        )
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(contents.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(orjson.loads(stripped))
        except orjson.JSONDecodeError as e:
            raise RuntimeError(
                f"{TASK_NAME}: {what} at {path} is not valid JSON-lines "
                f"(line {line_no}). The installed bfcl-eval may not match "
                f"AIPERF_ACCURACY_BFCL_VERSION_PIN. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    return rows


class BFCLASTBenchmark(AIPerfLoggerMixin):
    """BFCL single-turn tool-call correctness loader (Prompt mode).

    Reads the question and ``possible_answer`` files bundled in the installed
    ``bfcl-eval`` wheel, builds each entry's chat messages through upstream's
    own prompt builder, and serializes the ground truth for
    ``ToolCallASTGrader``.
    """

    @classmethod
    def check_available(cls) -> None:
        """Raise if bfcl-eval is missing or is not the pinned version.

        Called by the main-process preflight so both failures surface as a clean
        ConfigurationError before any service spawns, rather than raising deep
        in the dataset-manager loader after a full bootstrap.

        The version pin is checked here and not only at load time because a
        mismatch invalidates the entire run's numbers - the wheel carries both
        the questions and the checker - so it is worth the same up-front
        treatment as a missing install.
        """
        _bfcl_compat.require_bfcl()
        _bfcl_compat.check_version_pin()

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.check_available()
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load BFCL problems in Prompt mode.

        Args:
            tasks: BFCL categories to evaluate, or ``None`` for
                ``DEFAULT_CATEGORIES``. Names are validated against
                ``SUPPORTED_CATEGORIES``.
            n_shots: Must be ``0``. BFCL Prompt mode is zero-shot - the tool
                schemas in the system prompt are the whole context.
            enable_cot: Must be ``False``. BFCL's template instructs the model
                to return only the call list, so a CoT preamble breaks decoding
                and diverges from the reference.

        Returns:
            One ``BenchmarkProblem`` per entry, grouped by category in the
            order the categories were requested. ``task`` is the BFCL category
            and ``ground_truth`` is the orjson payload ``ToolCallASTGrader``
            consumes.

        Raises:
            NotImplementedError: for unsupported flag values or categories,
                prefixed with ``"<TASK_NAME>: "`` per aiperf's validator-gate
                convention.
        """
        if n_shots != 0:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-n-shots != 0 is not supported; BFCL's "
                "Prompt mode is zero-shot - the tool schemas injected into the "
                "system prompt are the entire context the reference provides."
            )
        if enable_cot:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-enable-cot is not supported; BFCL's "
                "Prompt-mode template instructs the model to return only the "
                "function calls, so a chain-of-thought preamble both breaks "
                "call-list decoding and diverges from the reference prompt."
            )
        self._reject_system_prompt_override()
        categories = self._resolve_categories(tasks)

        rows = await asyncio.to_thread(self._load_categories, categories)
        return await asyncio.to_thread(self._build_problems, rows)

    def _reject_system_prompt_override(self) -> None:
        """Reject ``--accuracy-system-prompt`` for this benchmark.

        Every other benchmark takes one global system prompt, but BFCL's is
        built per entry from that entry's own tool schemas. A global override
        would replace it, stripping the tool definitions the model is being
        asked to call - scoring near zero for a reason nothing in the output
        would explain.
        """
        acc_cfg = self.run.cfg.accuracy
        if acc_cfg is not None and acc_cfg.system_prompt is not None:
            raise NotImplementedError(
                f"{TASK_NAME}: --accuracy-system-prompt is not supported; BFCL "
                "builds a per-problem system prompt from that problem's tool "
                "schemas, and a global override would replace it - removing the "
                "tool definitions the model is being asked to call. Unset the "
                "flag to run this benchmark."
            )

    @staticmethod
    def _resolve_categories(tasks: list[str] | None) -> tuple[str, ...]:
        """Validate ``--accuracy-tasks`` against the gradeable categories."""
        if tasks is None:
            return DEFAULT_CATEGORIES
        unknown = [t for t in tasks if t not in SUPPORTED_CATEGORIES]
        if not unknown:
            return tuple(tasks)
        reasons = [
            f"'{name}' is intentionally out of scope: {reason}"
            for name in unknown
            for prefix, reason in _OUT_OF_SCOPE_PREFIXES.items()
            if name.startswith(prefix)
        ]
        detail = (" " + " ".join(reasons)) if reasons else ""
        raise NotImplementedError(
            f"{TASK_NAME}: --accuracy-tasks {unknown} is not supported.{detail} "
            f"Supported categories: {', '.join(SUPPORTED_CATEGORIES)}."
        )

    def _load_categories(
        self, categories: tuple[str, ...]
    ) -> list[tuple[str, dict[str, Any], Any]]:
        """Read every requested category and pair entries with their gold.

        Returns:
            ``(category, entry, possible_answer)`` triples in category order.
            ``possible_answer`` is ``None`` for the hallucination categories,
            which ship no ground-truth file.
        """
        _bfcl_compat.check_version_pin()
        prefix = _bfcl_compat.version_prefix()
        data_dir = _bfcl_compat.data_dir()
        answer_dir = _bfcl_compat.possible_answer_dir()

        paired: list[tuple[str, dict[str, Any], Any]] = []
        for category in categories:
            file_name = f"{prefix}_{category}.json"
            entries = _read_jsonl(
                data_dir / file_name, f"the {category!r} question file"
            )
            answers = self._load_answers(answer_dir / file_name, category)
            expects_gold = category not in _NO_GROUND_TRUTH_CATEGORIES
            for entry in entries:
                entry_id = entry.get("id")
                if expects_gold and entry_id not in answers:
                    raise RuntimeError(
                        f"{TASK_NAME}: question {entry_id!r} in category "
                        f"{category!r} has no matching entry in "
                        f"{answer_dir / file_name}. The question and "
                        f"possible_answer files are joined on 'id' and are "
                        f"expected to cover the same set; a gap means the "
                        f"installed bfcl-eval does not match "
                        f"AIPERF_ACCURACY_BFCL_VERSION_PIN, or its data layout "
                        f"changed. Grading with no ground truth would report a "
                        f"silent 0% for this category rather than an error."
                    )
                paired.append((category, entry, answers.get(entry_id)))
            count = len(entries)
            self.debug(
                lambda category=category, count=count: (
                    f"{TASK_NAME}: loaded {count} entries for category {category!r}"
                )
            )
        return paired

    @staticmethod
    def _load_answers(path: Path, category: str) -> dict[str, Any]:
        """Load one category's ``possible_answer`` file, keyed by entry id.

        The hallucination categories ship no such file - their verdict is
        whether a call was emitted at all - so they return empty rather than
        failing on the missing path.
        """
        if category in _NO_GROUND_TRUTH_CATEGORIES:
            return {}
        rows = _read_jsonl(path, f"the {category!r} possible_answer file")
        return {row.get("id"): row.get("ground_truth") for row in rows}

    def _build_problems(
        self, rows: list[tuple[str, dict[str, Any], Any]]
    ) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for category, entry, possible_answer in rows:
            problems.append(self._build_problem(category, entry, possible_answer))
        return problems

    def _build_problem(
        self, category: str, entry: dict[str, Any], possible_answer: Any
    ) -> BenchmarkProblem:
        entry_id = str(entry.get("id", ""))
        function_docs = entry.get("function", [])
        # BFCL nests each entry's turns one level deep (``question[0]`` is the
        # first - and for single-turn categories, only - turn).
        question_turns = entry.get("question") or [[]]
        messages = _bfcl_compat.build_chat_messages(
            question_turns[0], function_docs, entry_id
        )
        raw_messages: list[AccuracyChatMessage] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        prompt = "\n\n".join(m["content"] for m in raw_messages)
        return BenchmarkProblem(
            prompt=prompt,
            ground_truth=self._build_ground_truth(
                entry_id, category, function_docs, possible_answer
            ),
            task=category,
            metadata={
                "bfcl_id": entry_id,
                "language": _language_for(category),
                "test_category": category,
                "generation_size": DEFAULT_GENERATION_SIZE,
            },
            raw_messages=raw_messages,
        )

    @staticmethod
    def _build_ground_truth(
        entry_id: str,
        category: str,
        function_docs: Any,
        possible_answer: Any,
    ) -> str:
        """Serialize everything ``ToolCallASTGrader`` needs at grade time.

        ``possible_answer`` is passed through verbatim: its shape (per argument,
        a list of accepted values, with ``""`` standing in for an omitted
        optional) is owned by upstream's checker, not by this loader.
        """
        payload = {
            "id": entry_id,
            "test_category": category,
            "language": _language_for(category),
            "function": function_docs,
            "possible_answer": possible_answer,
        }
        return orjson.dumps(payload).decode("utf-8")
