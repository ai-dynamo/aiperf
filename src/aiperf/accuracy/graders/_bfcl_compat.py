# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy compatibility shim for the optional ``bfcl-eval`` dependency.

Every call aiperf makes into ``bfcl_eval`` goes through this module. Two
reasons it exists at all:

1. **Import cost.** Plugin discovery imports every registered benchmark and
   grader class eagerly, and importing anything under ``bfcl_eval`` transitively
   pulls its full model-handler stack (anthropic, cohere, boto3, faiss-cpu,
   sentence-transformers, qwen-agent, soundfile). Paying that on every aiperf
   invocation - including plain perf runs that never touch accuracy - is not
   acceptable, so ``bfcl_eval`` is imported lazily inside the functions here and
   never at module scope.

2. **API stability.** BFCL's checker, decoder and prompt builder are internals
   of an eval harness, not a versioned public API. Resolving them through
   ordered candidate lists in one place means an upstream move surfaces as a
   single clear error naming what was tried, and ``ast_checker`` is always
   invoked with **keyword** binding so a parameter reorder cannot silently
   mis-grade a whole run by shifting arguments.

API recorded against ``bfcl-eval==2026.3.23``::

    bfcl_eval.eval_checker.ast_eval.ast_checker:
        ast_checker(func_description, model_output, possible_answer,
                    language: Language, test_category: str, model_name: str)
            -> {"valid": bool, "error": [...], "error_type": str}

    bfcl_eval.model_handler.utils:
        ast_parse(input_str, language: ReturnFormat = ReturnFormat.PYTHON,
                  has_tool_call_tag: bool = False) -> list[dict]
        system_prompt_pre_processing_chat_model(
            prompts: list[dict], function_docs: list[dict], test_entry_id: str
        ) -> list[dict]

Note the two enums are NOT interchangeable: the checker takes ``Language``
(python/java/javascript) while the decoder takes ``ReturnFormat`` (which also
covers json/xml return styles). :func:`ast_check` and :func:`decode_calls` each
convert from our plain lowercase language string.

The version aiperf is written against is pinned by
``AIPERF_ACCURACY_BFCL_VERSION_PIN`` and enforced by :func:`check_version_pin`,
because BFCL ships its dataset and its checker in the same wheel: changing the
version changes both the questions asked and the scores returned.

Mirrors ``aiperf.accuracy.benchmarks._datasets_compat`` (the same lazy-import
posture for the ``datasets`` package).
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import math
import operator
from copy import deepcopy
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

DISTRIBUTION_NAME = "bfcl-eval"
PACKAGE_NAME = "bfcl_eval"

#: Sentinel accepted by ``AIPERF_ACCURACY_BFCL_VERSION_PIN`` to disable the
#: installed-version check entirely.
ANY_VERSION = "any"

MISSING_BFCL_HINT = (
    "bfcl-eval is not installed; the 'bfcl_ast' benchmark and the "
    "'tool_call_ast' grader cannot run. Install it with: "
    "uv pip install 'aiperf[bfcl]'."
)

# Ordered (module, attribute) candidates for each upstream symbol. The first
# that resolves wins; when none does, the raised error names every path tried,
# so recovering from an upstream move is a one-line edit here rather than a
# debugging session.
_CHECKER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.eval_checker.ast_eval.ast_checker", "ast_checker"),
)
_AST_PARSE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.model_handler.utils", "ast_parse"),
    ("bfcl_eval.model_handler.parser.ast_parser", "ast_parse"),
)
_PROMPT_BUILDER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.model_handler.utils", "system_prompt_pre_processing_chat_model"),
)
_FUNC_DOC_PREPROCESSOR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.utils", "_func_doc_language_specific_pre_processing"),
)
_LANGUAGE_ENUM_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.enums", "Language"),
)
_RETURN_FORMAT_ENUM_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.enums", "ReturnFormat"),
)
_NON_LIVE_CATEGORY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.category_mapping", "NON_LIVE_CATEGORY"),
)
_LIVE_CATEGORY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.category_mapping", "LIVE_CATEGORY"),
)
_VERSION_PREFIX_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.category_mapping", "VERSION_PREFIX"),
)
_MODEL_CONFIG_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("bfcl_eval.constants.model_config", "MODEL_CONFIG_MAPPING"),
)

# Our canonical ``ast_check`` keyword -> the parameter names upstream has used
# for it, tried in order against the live signature. The first entry of each
# tuple is what ``bfcl-eval==2026.3.23`` declares.
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "func_description": ("func_description", "func_doc", "function", "functions"),
    "model_output": ("model_output", "model_result", "model_result_decoded"),
    "possible_answer": ("possible_answer", "possible_answers", "answer"),
    "language": ("language", "test_language"),
    "test_category": ("test_category", "category"),
    "model_name": ("model_name", "model", "model_id"),
}


class BFCLDecodeError(Exception):
    """The model response could not be decoded into a BFCL call list.

    Raised by :func:`decode_calls` when the response is not parseable as a
    ``[func_name(param=value)]`` call list (or its Java/JavaScript equivalent).
    The grader turns this into ``GradingResult.unparsed=True`` - a model
    format-adherence failure, which is a different signal from a wrong answer.
    """


def bfcl_available() -> bool:
    """Whether ``bfcl_eval`` is importable, without importing it."""
    try:
        return importlib.util.find_spec(PACKAGE_NAME) is not None
    except (ImportError, ValueError):
        return False


def require_bfcl() -> None:
    """Raise when ``bfcl-eval`` is not installed.

    Called from ``check_available()`` on both the ``bfcl_ast`` benchmark loader
    and the ``tool_call_ast`` grader, so a missing extra surfaces from the
    main-process preflight as a clean ``ConfigurationError`` before any service
    is spawned - instead of crashing the daemon record processor mid-run.

    Raises:
        RuntimeError: carrying the ``uv pip install 'aiperf[bfcl]'`` recovery step.
    """
    if not bfcl_available():
        raise RuntimeError(MISSING_BFCL_HINT)


def installed_version() -> str:
    """Version of the installed ``bfcl-eval`` distribution.

    Raises:
        RuntimeError: when the distribution metadata is absent (e.g. the package
            was dropped onto ``sys.path`` rather than installed), since the
            version pin then cannot be verified.
    """
    try:
        return importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError as e:
        raise RuntimeError(
            f"{DISTRIBUTION_NAME} is importable but has no distribution "
            f"metadata, so its version cannot be verified against "
            f"AIPERF_ACCURACY_BFCL_VERSION_PIN. Install it normally with "
            f"``uv pip install 'aiperf[bfcl]'``, or set "
            f"``AIPERF_ACCURACY_BFCL_VERSION_PIN={ANY_VERSION}`` to skip the "
            f"check (scores then carry no version guarantee). "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def _version_for_message() -> str:
    """Installed version for use inside an error message, never raising.

    :func:`installed_version` raises when distribution metadata is missing. That
    is the right behavior for the version-pin check, but inside the message of
    *another* error it would replace the real cause with an unrelated one - so
    these call sites degrade to a placeholder instead.
    """
    try:
        return installed_version()
    except RuntimeError:
        return "<version unavailable>"


def check_version_pin() -> None:
    """Enforce ``AIPERF_ACCURACY_BFCL_VERSION_PIN`` against the install.

    BFCL ships its dataset and its AST checker in one wheel, so the package
    version determines both which questions are asked and how answers are
    scored. Two runs on different versions are not comparable, and the drift is
    silent - hence a hard check rather than a warning.

    No-op when the pin is ``any``.

    Raises:
        RuntimeError: on mismatch, naming both versions and the exact
            ``uv pip install`` command that reconciles them.
    """
    from aiperf.common.environment import Environment

    pin = Environment.ACCURACY.BFCL_VERSION_PIN
    if pin == ANY_VERSION:
        return
    found = installed_version()
    if found == pin:
        return
    raise RuntimeError(
        f"bfcl_ast: installed {DISTRIBUTION_NAME} is {found!r} but "
        f"AIPERF_ACCURACY_BFCL_VERSION_PIN is {pin!r}. BFCL bundles its dataset "
        f"and its AST checker in the same wheel, so these two versions ask "
        f"different questions AND score them differently - the run would not be "
        f"comparable to the pinned baseline. Either install the pinned version "
        f"(``uv pip install '{DISTRIBUTION_NAME}=={pin}'``) or set "
        f"``AIPERF_ACCURACY_BFCL_VERSION_PIN={found}`` to rebaseline against "
        f"what is installed (``={ANY_VERSION}`` disables the check entirely)."
    )


def package_root() -> Path:
    """Filesystem root of the installed ``bfcl_eval`` package."""
    require_bfcl()
    spec = importlib.util.find_spec(PACKAGE_NAME)
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations:
        raise RuntimeError(
            f"{PACKAGE_NAME} is importable but exposes no package directory, so "
            f"its bundled data files cannot be located. Reinstall with "
            f"``uv pip install 'aiperf[bfcl]'``."
        )
    return Path(locations[0])


def data_dir() -> Path:
    """Directory holding BFCL's bundled question files.

    BFCL vendors its dataset inside the wheel (gorilla PR #504), so there is no
    download step and no HuggingFace dataset that can drift away from the
    pinned checker.
    """
    return package_root() / "data"


def possible_answer_dir() -> Path:
    """Directory holding BFCL's bundled ``possible_answer`` ground truth."""
    return data_dir() / "possible_answer"


def _resolve(candidates: tuple[tuple[str, str], ...], what: str) -> Any:
    """Return the first resolvable ``(module, attribute)`` candidate.

    Args:
        candidates: Ordered ``(module_path, attribute_name)`` pairs.
        what: Human-readable name of the symbol, used in the error message.

    Raises:
        RuntimeError: when no candidate resolves, listing every path tried and
            the installed version.
    """
    from aiperf.common.environment import Environment

    require_bfcl()
    tried: list[str] = []
    for module_path, attr in candidates:
        tried.append(f"{module_path}:{attr}")
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:  # pragma: no cover - upstream layout drift
            _log.debug("bfcl candidate %s not importable: %s", module_path, e)
            continue
        resolved = getattr(module, attr, None)
        if resolved is not None:
            return resolved
    raise RuntimeError(
        f"bfcl_ast: cannot locate {what} in the installed {DISTRIBUTION_NAME} "
        f"{_version_for_message()}. Tried: {', '.join(tried)}. BFCL's internals "
        f"are not a versioned public API; either install the pinned version "
        f"(``uv pip install '{DISTRIBUTION_NAME}=="
        f"{Environment.ACCURACY.BFCL_VERSION_PIN}'``) or add the new path to "
        f"the candidate list in aiperf/accuracy/graders/_bfcl_compat.py."
    )


def version_prefix() -> str:
    """BFCL's bundled-data filename prefix (e.g. ``BFCL_v4``)."""
    return str(_resolve(_VERSION_PREFIX_CANDIDATES, "the data-file version prefix"))


def single_turn_categories() -> tuple[str, ...]:
    """Upstream's stateless single-turn categories (non-live + live).

    This is exactly the set aiperf can grade: every entry is a one-shot
    question scored by the AST checker or by abstention, with no backend state
    to carry across turns.
    """
    non_live = _resolve(_NON_LIVE_CATEGORY_CANDIDATES, "the non-live category list")
    live = _resolve(_LIVE_CATEGORY_CANDIDATES, "the live category list")
    return tuple(non_live) + tuple(live)


def check_checker_model_key(model_name: str) -> None:
    """Verify ``model_name`` is a key upstream's checker will accept.

    ``convert_func_name`` indexes ``MODEL_CONFIG_MAPPING`` with a bare
    subscript whenever a gold function name contains a dot, and it runs
    unconditionally before the function-name match. An unregistered key
    therefore raises ``KeyError`` on roughly a third of the gradeable dataset
    — and because grading is crash-guarded, that would surface as a pile of
    failed records rather than as the integration error it is.

    Checking it in preflight turns a silent, plausible-looking score into an
    immediate ``ConfigurationError`` naming the exact cause.

    Raises:
        RuntimeError: when the key is absent from the installed registry.
    """
    mapping = _resolve(_MODEL_CONFIG_CANDIDATES, "the model-config registry")
    if model_name in mapping:
        return
    raise RuntimeError(
        f"bfcl_ast: the grader's checker model key {model_name!r} is not "
        f"registered in the installed {DISTRIBUTION_NAME} "
        f"{_version_for_message()} (MODEL_CONFIG_MAPPING has "
        f"{len(mapping)} keys). Upstream's convert_func_name looks this key up "
        f"with a bare dict subscript for every dotted gold function name, so "
        f"grading would raise on roughly a third of the dataset. Pick another "
        f"registered prompt-mode key whose config has underscore_to_dot=False "
        f"and set CHECKER_MODEL_NAME in "
        f"aiperf/accuracy/graders/tool_call_ast.py, or install the pinned "
        f"version with ``uv pip install 'aiperf[bfcl]'``."
    )


def _language_enum(language: str) -> Any:
    """Convert a lowercase language string to upstream's ``Language`` member."""
    enum_cls = _resolve(_LANGUAGE_ENUM_CANDIDATES, "the Language enum")
    return enum_cls(language.lower())


def _return_format_enum(language: str) -> Any:
    """Convert a lowercase language string to upstream's ``ReturnFormat`` member.

    ``ReturnFormat`` is a different enum from ``Language`` - it also covers the
    json/xml return styles BFCL's format-sensitivity work uses - but its
    python/java/javascript members carry the same values, which is what the
    Prompt-mode decoder dispatches on.
    """
    enum_cls = _resolve(_RETURN_FORMAT_ENUM_CANDIDATES, "the ReturnFormat enum")
    return enum_cls(language.lower())


def build_chat_messages(
    question_messages: list[dict[str, Any]],
    function_docs: list[dict[str, Any]],
    test_entry_id: str,
) -> list[dict[str, Any]]:
    """Compose the Prompt-mode chat messages exactly as BFCL does.

    Delegates to upstream's ``system_prompt_pre_processing_chat_model`` rather
    than reproducing the template locally. BFCL v4 no longer keeps a single
    system-prompt constant: the prompt is assembled per entry from a style
    table, an output-format table and the entry's own format spec. Building it
    ourselves would be a standing parity risk, and BFCL v4's format-sensitivity
    results show models are highly sensitive to exactly this template - some
    tool-trained models drop to near-zero on small wording changes.

    Upstream mutates and returns the list it is given, so a copy is passed in.

    Args:
        question_messages: The entry's ``question`` turn (role/content dicts).
        function_docs: The entry's ``function`` tool schemas.
        test_entry_id: The entry's ``id``; upstream derives the prompt format
            from it.

    Returns:
        Messages with BFCL's system prompt at index 0.
    """
    build = _resolve(_PROMPT_BUILDER_CANDIDATES, "the system-prompt builder")
    return list(
        build([dict(m) for m in question_messages], function_docs, test_entry_id)
    )


def preprocess_function_docs(
    function_docs: list[dict[str, Any]], test_category: str
) -> list[dict[str, Any]]:
    """Apply BFCL's language-specific preprocessing to a *copy* of the tool schemas.

    Upstream only ever builds a Prompt-mode prompt from preprocessed schemas:
    ``load_dataset_entry`` runs every entry through this step before
    generation (``add_language_specific_hint_to_function_doc``, which calls
    the ``_func_doc_language_specific_pre_processing`` resolved here, both in
    ``bfcl_eval/utils.py``). It appends the language hint to each description
    and, for Java/JavaScript, rewrites ``type``/``properties`` into the
    "string representation" form the prompt instructs the model to use.

    This must run on a **copy**: the function mutates its input in place
    (appending to ``description`` and, for Java/JavaScript, rewriting
    ``type``/``properties``), and the AST checker needs the original,
    unmodified schema to grade against - the Java/JavaScript rewrite in
    particular replaces real parameter types with ``"string"``, which would
    silently break type checking if it ever reached the grader. Callers must
    keep using the original ``function_docs`` for ground truth and pass only
    this function's return value to :func:`build_chat_messages`.

    Args:
        function_docs: The entry's function documentation (tool schemas),
            never mutated by this call.
        test_category: BFCL category; selects the Java/JavaScript/Python hint
            and parameter rewrite.

    Returns:
        A deep copy of ``function_docs`` with upstream's language-specific
        hints and type rewrites applied.
    """
    preprocess = _resolve(
        _FUNC_DOC_PREPROCESSOR_CANDIDATES, "the function-doc language preprocessor"
    )
    copied = deepcopy(function_docs)
    return list(preprocess(copied, test_category))


def _bind_checker_kwargs(checker: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map our canonical keyword names onto the live ``ast_checker`` signature.

    Returns:
        ``kwargs`` rekeyed to the parameter names the installed checker declares.

    Raises:
        RuntimeError: when an argument matches no parameter under any known
            alias - i.e. upstream renamed something, and calling positionally
            would have mis-graded the run in silence.
    """
    parameters = inspect.signature(checker).parameters
    bound: dict[str, Any] = {}
    for canonical, value in kwargs.items():
        for alias in _ARG_ALIASES[canonical]:
            if alias in parameters:
                bound[alias] = value
                break
        else:
            raise RuntimeError(
                f"bfcl_ast: the installed {DISTRIBUTION_NAME} "
                f"{_version_for_message()} ast_checker has no parameter for "
                f"{canonical!r} (tried aliases {list(_ARG_ALIASES[canonical])}; "
                f"it declares {list(parameters)}). Refusing to call it rather "
                f"than risk silently mis-grading the run. Install the pinned "
                f"version, or extend _ARG_ALIASES in "
                f"aiperf/accuracy/graders/_bfcl_compat.py."
            )
    return bound


def ast_check(
    *,
    func_description: Any,
    model_output: Any,
    possible_answer: Any,
    language: str,
    test_category: str,
    model_name: str,
) -> dict[str, Any]:
    """Run BFCL's deterministic AST checker over one decoded response.

    Args:
        func_description: The entry's function documentation (tool schemas).
        model_output: Decoded call list, BFCL's ``[{"func": {"param": val}}]``.
        possible_answer: The entry's ``ground_truth`` list, verbatim.
        language: ``"python"``, ``"java"`` or ``"javascript"``.
        test_category: BFCL category, which selects the per-category checker
            (parallel categories are compared order-independently).
        model_name: Forwarded to upstream, which consults it only for a few
            model-specific leniencies.

    Returns:
        The checker's verdict: ``{"valid": bool, "error": [...],
        "error_type": str}``.
    """
    checker = _resolve(_CHECKER_CANDIDATES, "the AST checker")
    bound = _bind_checker_kwargs(
        checker,
        {
            "func_description": func_description,
            "model_output": model_output,
            "possible_answer": possible_answer,
            "language": _language_enum(language),
            "test_category": test_category,
            "model_name": model_name,
        },
    )
    result = checker(**bound)
    if not isinstance(result, dict):  # pragma: no cover - upstream drift
        raise RuntimeError(
            f"bfcl_ast: ast_checker returned {type(result).__name__}, expected a "
            f"dict with 'valid'/'error'/'error_type'. The installed "
            f"{DISTRIBUTION_NAME} {_version_for_message()} is incompatible with "
            f"this integration."
        )
    return result


# ---------------------------------------------------------------------------
# Non-executing Python-call decoder.
#
# Upstream's own ``resolve_ast_by_type`` (bfcl_eval.model_handler.utils) calls
# ``eval(ast.unparse(node))`` for ``ast.BinOp`` and ``ast.Lambda`` argument
# values. The AST-node check only restricts which *shape* of expression is
# allowed to reach ``eval`` - the re-serialized source text it hands to
# ``eval`` is then re-parsed and executed by CPython with no further
# restriction. Confirmed directly against the pinned wheel: a response whose
# decoded call contained a ``BinOp``-shaped argument built from
# ``__import__('builtins').print('PROBE')`` executed in-process through the
# real grader - i.e. inference-server output (fully attacker-controlled, in
# the threat-model sense that aiperf does not trust the model under test) can
# run arbitrary Python with aiperf's own credentials and filesystem access.
#
# This reimplements the value-resolution step BFCL's Python-language
# Prompt-mode decoder needs, mirroring every non-executing branch of
# ``resolve_ast_by_type``/``resolve_ast_call`` one-for-one, and replacing the
# two executing branches with a bounded, non-executing arithmetic evaluator
# (``BinOp``) or an outright refusal (``Lambda`` - already unreachable
# upstream in practice, since its own body indexes ``Lambda.body[0]`` against
# a single AST node rather than a list and raises on any real lambda, so
# refusing it here costs no decoding capability). ``ast.parse`` itself only
# builds a syntax tree and never executes code, so that step is unchanged
# from upstream.
#
# Java and JavaScript still route through upstream's tree-sitter-based
# ``parse_java_function_call``/``parse_javascript_function_call`` below
# (verified against the installed wheel: no ``eval``/``exec`` anywhere under
# ``bfcl_eval/model_handler/parser/``), so only the Python path needs this.
# ---------------------------------------------------------------------------

#: Binary operators safe to apply once both operands are already confirmed
#: numeric. Deliberately excludes anything that could reach a non-numeric
#: type (string concatenation/repetition, matrix multiply, etc.) - operands
#: are validated through :func:`_safe_numeric` before any operator here runs.
_SAFE_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

#: Upper bound on ``**`` exponent magnitude. Unbounded exponentiation on
#: attacker-controlled operands is a CPU/memory exhaustion vector even
#: without ``eval`` (nested powers grow arbitrarily large integers); no
#: legitimate BFCL tool-call argument needs an exponent this large.
_MAX_POW_EXPONENT = 1024

#: Upper bound on the bit length of any integer operand or result inside an
#: argument expression. Decoding runs synchronously on the record-processor
#: event loop, so ``**``, ``<<`` and ``*`` are size-checked *before* they run:
#: a bounded exponent alone still admits ``((3**1024)**1024)**64`` and
#: ``1<<100000000``. 256 bits is far beyond any real tool-call argument and
#: keeps every result printable and serializable.
_MAX_INT_BITS = 256


def _safe_numeric(node: ast.AST) -> int | float:
    """Resolve a numeric literal/unary/binary expression without executing it.

    Raises:
        BFCLDecodeError: the node is not built entirely from numeric
            constants, unary +/-, and the operators in :data:`_SAFE_BINOPS` -
            i.e. it is not provably safe to evaluate.
    """
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_numeric(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        return _safe_binop(node)
    raise BFCLDecodeError(
        f"unsupported value in numeric expression: {ast.dump(node, annotate_fields=False)}"
    )


def _safe_binop(node: ast.BinOp) -> int | float:
    """Apply one binary operator to two recursively-validated numeric operands."""
    op_fn = _SAFE_BINOPS.get(type(node.op))
    if op_fn is None:
        raise BFCLDecodeError(
            f"unsupported operator in numeric expression: {type(node.op).__name__}"
        )
    left = _check_magnitude(_safe_numeric(node.left))
    right = _check_magnitude(_safe_numeric(node.right))
    if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
        raise BFCLDecodeError(
            f"exponent {right} exceeds the {_MAX_POW_EXPONENT} bound for a "
            f"tool-call argument expression"
        )
    if _estimated_result_bits(node.op, left, right) > _MAX_INT_BITS:
        raise BFCLDecodeError(
            f"{type(node.op).__name__} result would exceed {_MAX_INT_BITS} bits "
            f"for a tool-call argument expression"
        )
    try:
        result = op_fn(left, right)
    except (ZeroDivisionError, OverflowError, ValueError) as e:
        raise BFCLDecodeError(f"could not evaluate numeric expression: {e}") from e
    return _check_magnitude(result)


def _estimated_result_bits(op: ast.operator, left: float, right: float) -> int:
    """Lower bound on the result's bit length for the integer ops that can grow.

    Computed from the operands alone, so an oversized ``**``/``<<``/``*`` is
    refused without ever being evaluated. Every other operator yields at most
    one bit more than its bounded operands, which :func:`_check_magnitude`
    catches afterwards.
    """
    if not (isinstance(left, int) and isinstance(right, int)):
        return 0
    if isinstance(op, ast.Pow) and right > 0:
        return (abs(left).bit_length() - 1) * right
    if isinstance(op, ast.LShift) and right > 0:
        return abs(left).bit_length() + right
    if isinstance(op, ast.Mult):
        return abs(left).bit_length() + abs(right).bit_length() - 1
    return 0


def _check_magnitude(value: int | float) -> int | float:
    """Refuse integers wider than :data:`_MAX_INT_BITS` and non-finite floats."""
    if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
        raise BFCLDecodeError(
            f"integer of {value.bit_length()} bits exceeds the {_MAX_INT_BITS}-bit "
            f"bound for a tool-call argument expression"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise BFCLDecodeError(
            f"numeric expression evaluated to non-finite {value} in a tool-call argument"
        )
    return value


def _resolve_constant(value: ast.Constant) -> Any:
    return "..." if value.value is Ellipsis else value.value


def _resolve_list(value: ast.List) -> list[Any]:
    return [_resolve_value(v) for v in value.elts]


def _resolve_dict(value: ast.Dict) -> dict[Any, Any]:
    return {
        _resolve_value(k): _resolve_value(v)
        for k, v in zip(value.keys, value.values, strict=True)
    }


def _resolve_name(value: ast.Name) -> str:
    return value.id


def _resolve_call_value(value: ast.Call) -> Any:
    """A call with no keyword arguments is stringified, never evaluated.

    Mirrors upstream's own behavior for this shape: a bare
    ``ast.unparse(value)`` reproduces the call's source text (e.g. for a
    default-value sentinel like ``some_enum.MEMBER``), it does not execute
    anything. A call WITH keywords is a nested BFCL function call, resolved
    through :func:`_resolve_call` the same as the top level.
    """
    if len(value.keywords) == 0:
        return ast.unparse(value)
    return _resolve_call(value)


def _resolve_tuple(value: ast.Tuple) -> tuple[Any, ...]:
    return tuple(_resolve_value(v) for v in value.elts)


def _resolve_lambda(value: ast.Lambda) -> Any:  # noqa: ARG001
    raise BFCLDecodeError(
        "lambda expressions are not supported as a tool-call argument "
        "value (refused rather than evaluated - see the security note "
        "above _SAFE_BINOPS)"
    )


def _resolve_subscript(value: ast.Subscript) -> str:
    try:
        return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    except Exception as e:  # pragma: no cover - mirrors upstream's bare except
        raise BFCLDecodeError(f"unsupported subscript expression: {e}") from e


#: Dispatch table for :func:`_resolve_value`, keyed by exact AST node type
#: (not a subclass check - every node type BFCL's grammar can produce is
#: listed explicitly, so an unhandled type falls through to the function's
#: own refusal rather than silently matching the wrong handler).
_VALUE_RESOLVERS: dict[type, Any] = {
    ast.Constant: _resolve_constant,
    ast.UnaryOp: _safe_numeric,
    ast.List: _resolve_list,
    ast.Dict: _resolve_dict,
    ast.BinOp: _safe_numeric,
    ast.Name: _resolve_name,
    ast.Call: _resolve_call_value,
    ast.Tuple: _resolve_tuple,
    ast.Lambda: _resolve_lambda,
    ast.Subscript: _resolve_subscript,
}


def _resolve_value(value: ast.AST) -> Any:
    """Non-executing equivalent of upstream's ``resolve_ast_by_type``.

    Mirrors every branch of the upstream function except ``BinOp`` (routed
    through the bounded :func:`_safe_numeric` evaluator instead of ``eval``)
    and ``Lambda`` (refused outright instead of ``eval`` - see the module
    note above :data:`_SAFE_BINOPS`). Dispatches by exact node type through
    :data:`_VALUE_RESOLVERS` rather than an if/elif chain, so each node kind
    is its own small, independently testable function.
    """
    resolver = _VALUE_RESOLVERS.get(type(value))
    if resolver is None:
        raise BFCLDecodeError(f"unsupported AST node type: {type(value).__name__}")
    return resolver(value)


def _resolve_call(elem: ast.Call) -> dict[str, Any]:
    """Non-executing equivalent of upstream's ``resolve_ast_call``."""
    func_parts: list[str] = []
    func_part: ast.AST = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {arg.arg: _resolve_value(arg.value) for arg in elem.keywords}
    return {func_name: args_dict}


def _decode_python_calls(input_str: str) -> list[dict[str, Any]]:
    """Non-executing equivalent of upstream's ``ast_parse`` Python branch.

    ``ast.parse`` only builds a syntax tree - it never executes code - so
    this step is unchanged from upstream. The value resolution that follows
    is reimplemented through :func:`_resolve_call`/:func:`_resolve_value`
    instead of calling into ``bfcl_eval``, to keep model-controlled text out
    of ``eval``/``exec`` entirely.
    """
    cleaned = input_str.strip().strip("'")
    parsed = ast.parse(cleaned, mode="eval")
    body = parsed.body
    if isinstance(body, ast.Call):
        return [_resolve_call(body)]
    elements = getattr(body, "elts", None)
    if elements is None:
        raise BFCLDecodeError(
            f"expected a call or a list of calls, got {type(body).__name__}"
        )
    calls: list[dict[str, Any]] = []
    for elem in elements:
        if not isinstance(elem, ast.Call):
            raise BFCLDecodeError(
                f"expected every list element to be a call, got {type(elem).__name__}"
            )
        calls.append(_resolve_call(elem))
    return calls


def _normalize_prompt_response(response_text: str) -> str:
    """Upstream's Prompt-mode strip-and-wrap, applied before any language branch.

    Mirrors ``default_decode_ast_prompting`` (``bfcl_eval/model_handler/
    utils.py``): strip backticks, newlines and spaces, then bracket the text if
    it is not already. Skipping it changes verdicts - a fenced call list is
    undecodable without it (so ``irrelevance`` would score it a correct
    abstention), and upstream's Java/JavaScript parsers drop the first and
    last character on the assumption that this wrapping already happened.
    """
    result = response_text.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    return result


def decode_calls(response_text: str, language: str) -> list[dict[str, Any]]:
    """Decode a Prompt-mode response into BFCL's canonical call list.

    In Prompt mode the model answers in plain text with a Python-style call list
    (``[get_weather(city='SF')]``), decoded into ``[{"get_weather": {"city":
    "SF"}}]``. The text first goes through upstream's strip-and-wrap
    normalization (:func:`_normalize_prompt_response`) for every language.

    Security: the Python-language path never calls into ``bfcl_eval``'s own
    decoder. That decoder calls ``eval()`` on re-serialized source text for
    ``BinOp``/``Lambda`` argument values - a confirmed arbitrary-code-execution
    path through the grader process (see the module note above
    :data:`_SAFE_BINOPS`). Instead this calls :func:`_decode_python_calls`, a
    non-executing reimplementation that resolves every value node itself.
    Java/JavaScript still delegate to upstream's tree-sitter-based parsers,
    which do not call ``eval``/``exec`` (verified against the installed wheel).

    Args:
        response_text: The model's answer channel.
        language: ``"python"``, ``"java"`` or ``"javascript"``.

    Returns:
        One dict per call, mapping the function name to its arguments.

    Raises:
        BFCLDecodeError: when the response is not a parseable call list. This is
            the ``unparsed`` signal, not a grading verdict.
    """
    if not response_text or not response_text.strip():
        raise BFCLDecodeError(
            "cannot decode a BFCL call list: the model returned an empty "
            "answer channel. Usually the generation was cut off (max_tokens "
            "too low), or the model emitted only a reasoning channel."
        )
    normalized = _normalize_prompt_response(response_text)
    if language == "python":
        try:
            decoded = _decode_python_calls(normalized)
        except BFCLDecodeError:
            raise
        except Exception as e:
            # ast.parse raises SyntaxError/ValueError depending on how the
            # response is malformed; all of them mean the same thing here.
            raise BFCLDecodeError(f"{type(e).__name__}: {e}") from e
    else:
        parse = _resolve(_AST_PARSE_CANDIDATES, "the response decoder (ast_parse)")
        # Resolved OUTSIDE the try below: a failure here means upstream drift
        # (no ReturnFormat member for this language), which must surface as a
        # loud RuntimeError. Folded into the decode failure it would instead
        # mark every problem in the affected language `unparsed` - reading as
        # a model that never emits a parseable call rather than as a broken
        # integration.
        return_format = _return_format_enum(language)
        try:
            decoded = parse(normalized, return_format)
        except Exception as e:
            raise BFCLDecodeError(f"{type(e).__name__}: {e}") from e
    if not isinstance(decoded, list):  # pragma: no cover - upstream drift
        raise BFCLDecodeError(
            f"decoder returned {type(decoded).__name__}, expected a list of calls"
        )
    return decoded
