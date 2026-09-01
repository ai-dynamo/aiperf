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

import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
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


def decode_calls(response_text: str, language: str) -> list[dict[str, Any]]:
    """Decode a Prompt-mode response into BFCL's canonical call list.

    In Prompt mode the model answers in plain text with a Python-style call list
    (``[get_weather(city='SF')]``), which upstream's ``ast_parse`` turns into
    ``[{"get_weather": {"city": "SF"}}]``. Java and JavaScript go through the
    same entry point, which dispatches to its tree-sitter grammars.

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
    parse = _resolve(_AST_PARSE_CANDIDATES, "the response decoder (ast_parse)")
    # Resolved OUTSIDE the try below: a failure here means upstream drift (no
    # ReturnFormat member for this language), which must surface as a loud
    # RuntimeError. Folded into the decode failure it would instead mark every
    # problem in the affected language `unparsed` - reading as a model that
    # never emits a parseable call rather than as a broken integration.
    return_format = _return_format_enum(language)
    try:
        decoded = parse(response_text.strip(), return_format)
    except Exception as e:
        # Upstream raises SyntaxError/ValueError/AssertionError depending on how
        # the response is malformed; all of them mean the same thing here.
        raise BFCLDecodeError(f"{type(e).__name__}: {e}") from e
    if not isinstance(decoded, list):  # pragma: no cover - upstream drift
        raise BFCLDecodeError(
            f"decoder returned {type(decoded).__name__}, expected a list of calls"
        )
    return decoded
