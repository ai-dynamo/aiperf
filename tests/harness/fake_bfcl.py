# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal stand-in for the ``bfcl-eval`` surface aiperf uses.

``bfcl-eval`` pulls anthropic, cohere, boto3, faiss-cpu, sentence-transformers
and qwen-agent as *base* dependencies (upstream ships no checker-only extra),
and it pins ``numpy==1.26.4``, which conflicts with the ``[accuracy]`` extra's
``numpy>=2``. The default unit-test job therefore runs without it, and every
test pinning a real-bfcl contract is opt-in via ``@pytest.mark.requires_bfcl``.

This module re-creates just enough behavior for the ``bfcl_ast`` loader and
``tool_call_ast`` grader tests to exercise their own logic:

- :func:`ast_parse` decodes the Python call-list format into BFCL's canonical
  ``[{"func": {"param": value}}]`` shape, raising on anything that is not a
  call list (which is what drives the ``unparsed`` path).
- :func:`ast_checker` re-implements only the *verdict shape* - name match,
  required params, one-level type check, accepted-value match, unordered
  parallel comparison - and returns upstream's ``{"valid", "error",
  "error_type"}`` dict using its real ``error_type`` vocabulary, so bucket
  classification is exercised faithfully. It is emphatically **not** a
  reimplementation of upstream's semantics; parity against the real checker is
  asserted in ``test_bfcl_ast_parity.py``.
- :func:`build_chat_messages` returns a synthetic system prompt. It is
  deliberately *not* byte-equal to upstream's; the byte-equality contract is
  pinned in ``test_bfcl_prompt_template.py`` against the real install.
- :data:`DATA_ROWS` / :data:`POSSIBLE_ANSWER_ROWS` mirror the on-disk JSON-lines
  shapes so loader tests can write fixture files.

Wiring: ``tests/unit/accuracy/conftest.py`` patches the resolver functions in
``aiperf.accuracy.graders._bfcl_compat`` with these fakes per-test (autouse,
function scope) when the real ``bfcl_eval`` is not importable, and no-ops when
it is - so a ``[bfcl]`` environment exercises the real path end to end. Nothing
is injected into ``sys.modules``.
"""

from __future__ import annotations

import ast
from typing import Any

import orjson

#: Synthetic system prompt. Structurally similar to upstream's (persona, task,
#: return-format instruction, tool list) so loader assertions about ordering and
#: tool inclusion are meaningful, but deliberately not byte-equal.
SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert in composing functions."
    "You are given a question and a set of possible functions.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the "
    "format of [func_name1(params_name1=params_value1)]\n\n"
    "Here is a list of functions in JSON format that you can invoke.\n"
    "{functions}\n"
)

#: Question-file rows, keyed by category, in the bundled JSON-lines shape.
DATA_ROWS: dict[str, list[dict[str, Any]]] = {
    "simple_python": [
        {
            "id": "simple_python_0",
            "question": [[{"role": "user", "content": "What is the weather in SF?"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "city": {"type": "string", "description": "City name."},
                            "days": {"type": "integer", "description": "Horizon."},
                        },
                        "required": ["city"],
                    },
                }
            ],
        },
    ],
    "parallel": [
        {
            "id": "parallel_0",
            "question": [[{"role": "user", "content": "Weather in SF and LA?"}]],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "city": {"type": "string", "description": "City name."}
                        },
                        "required": ["city"],
                    },
                }
            ],
        },
    ],
    "irrelevance": [
        {
            "id": "irrelevance_0",
            "question": [
                [{"role": "user", "content": "What is the capital of France?"}]
            ],
            "function": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "city": {"type": "string", "description": "City name."}
                        },
                        "required": ["city"],
                    },
                }
            ],
        },
    ],
}

#: ``possible_answer`` rows, keyed by category, in the bundled JSON-lines shape.
POSSIBLE_ANSWER_ROWS: dict[str, list[dict[str, Any]]] = {
    "simple_python": [
        {
            "id": "simple_python_0",
            "ground_truth": [{"get_weather": {"city": ["SF"], "days": [1, ""]}}],
        },
    ],
    "parallel": [
        {
            "id": "parallel_0",
            "ground_truth": [
                {"get_weather": {"city": ["SF"]}},
                {"get_weather": {"city": ["LA"]}},
            ],
        },
    ],
}

VERSION_PREFIX = "BFCL_v4"

# Mirrors upstream's category lists so the loader's drift assertion has
# something to compare against without the real install.
NON_LIVE_CATEGORY = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]
LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]

# Upstream's JSON-schema type names -> the Python types the checker accepts.
# ``integer`` deliberately excludes ``bool`` (a Python bool is an int) and
# ``float`` accepts ``int``, matching BFCL's documented int->float leniency.
_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "float": (float, int),
    "boolean": (bool,),
    "array": (list,),
    "dict": (dict,),
    "tuple": (tuple, list),
}


def ast_parse(
    input_str: str, language: Any = None, has_tool_call_tag: bool = False
) -> list[dict[str, Any]]:
    """Decode a Python-style call list into BFCL's canonical shape.

    Raises:
        SyntaxError: when the text is not parseable as a Python expression.
        ValueError: when it parses but is not a call (or list of calls).
    """
    parsed = ast.parse(input_str.strip().strip("'"), mode="eval")
    body = parsed.body
    calls = [body] if isinstance(body, ast.Call) else getattr(body, "elts", None)
    if calls is None:
        raise ValueError(f"not a call list: {input_str!r}")
    decoded: list[dict[str, Any]] = []
    for element in calls:
        if not isinstance(element, ast.Call):
            raise ValueError(f"list element is not a call: {input_str!r}")
        decoded.append({_call_name(element): _call_kwargs(element)})
    return decoded


def _call_name(node: ast.Call) -> str:
    """Render a call's (possibly dotted) function name."""
    parts: list[str] = []
    func: Any = node.func
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _call_kwargs(node: ast.Call) -> dict[str, Any]:
    """Evaluate a call's keyword arguments to literal Python values."""
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}


#: The subset of upstream's MODEL_CONFIG_MAPPING keys these tests use. The real
#: registry has 175; only membership matters for the dotted-name lookup.
REGISTERED_MODEL_NAMES = frozenset({"gorilla-openfunctions-v2"})


def _reject_unregistered_model_name(model_name: str, possible_answer: Any) -> None:
    """Reproduce upstream's ``KeyError`` on an unregistered model key."""
    if model_name in REGISTERED_MODEL_NAMES:
        return
    for gold in possible_answer or []:
        if any("." in name for name in gold):
            raise KeyError(model_name)


def _result(valid: bool, error: str = "", error_type: str = "") -> dict[str, Any]:
    return {
        "valid": valid,
        "error": [error] if error else [],
        "error_type": error_type or "simple_function_checker:unclear",
    }


def _check_one(
    call: dict[str, Any], gold: dict[str, Any], func_description: Any
) -> dict[str, Any]:
    """Verdict for a single call against a single gold entry."""
    ((gold_name, gold_args),) = gold.items()
    ((name, args),) = call.items()
    if name != gold_name:
        return _result(
            False,
            f"Function name '{gold_name}' not found in model output.",
            "simple_function_checker:wrong_func_name",
        )
    properties = _properties_for(func_description, name)
    for param, accepted in gold_args.items():
        if param not in args:
            # "" in the accepted list marks an optional the model may omit.
            if "" in accepted:
                continue
            return _result(
                False,
                f"Missing required parameter: '{param}'.",
                "simple_function_checker:missing_required",
            )
        value = args[param]
        expected_type = properties.get(param, {}).get("type")
        allowed = _TYPE_CHECKS.get(expected_type or "")
        if allowed and (isinstance(value, bool) is not (expected_type == "boolean")):
            return _result(
                False,
                f"Incorrect type for parameter '{param}'. Expected type "
                f"{expected_type}, got {type(value).__name__}.",
                "type_error:simple",
            )
        if allowed and not isinstance(value, allowed):
            return _result(
                False,
                f"Incorrect type for parameter '{param}'. Expected type "
                f"{expected_type}, got {type(value).__name__}.",
                "type_error:simple",
            )
        if value not in accepted:
            return _result(
                False,
                f"Invalid value for parameter '{param}': {value}. "
                f"Expected one of {accepted}.",
                "value_error:others",
            )
    for param in args:
        if param not in gold_args:
            return _result(
                False,
                f"Unexpected parameter: '{param}'.",
                "simple_function_checker:unexpected_param",
            )
    return _result(True)


def _properties_for(func_description: Any, name: str) -> dict[str, Any]:
    """Parameter schema for ``name`` out of the entry's function docs."""
    for doc in func_description or []:
        if doc.get("name") == name:
            return doc.get("parameters", {}).get("properties", {})
    return {}


def ast_checker(
    func_description: Any,
    model_output: Any,
    possible_answer: Any,
    language: Any = None,
    test_category: str = "simple_python",
    model_name: str = "gorilla-openfunctions-v2",
) -> dict[str, Any]:
    """Verdict in upstream's shape for a decoded call list.

    Parallel categories are compared without regard to order, matching
    upstream's ``parallel_function_checker_no_order``.

    Raises:
        KeyError: when ``model_name`` is not a registered key and a gold
            function name contains a dot. Upstream's ``convert_func_name``
            indexes ``MODEL_CONFIG_MAPPING`` with a bare subscript in exactly
            that case, so an unregistered key raises there. The fake models it
            because the alternative — accepting any name — is what let an
            unregistered key ship: the real oracle only runs in a ``[bfcl]``
            environment, which CI does not build.
    """
    _reject_unregistered_model_name(model_name, possible_answer)
    gold_calls = list(possible_answer or [])
    calls = list(model_output or [])
    if len(calls) != len(gold_calls):
        return _result(
            False,
            f"Wrong number of functions: expected {len(gold_calls)}, got {len(calls)}.",
            f"{_checker_name(test_category)}:wrong_count",
        )
    remaining = list(gold_calls)
    for call in calls:
        for index, gold in enumerate(remaining):
            if _check_one(call, gold, func_description)["valid"]:
                remaining.pop(index)
                break
        else:
            # Report the first gold entry's specific failure, which is what a
            # single-call category cares about; multi-call categories fall back
            # to upstream's "no match" error_type.
            failure = _check_one(call, remaining[0], func_description)
            if len(gold_calls) > 1:
                return _result(
                    False,
                    str(failure["error"][0]) if failure["error"] else "",
                    "parallel_function_checker_no_order:cannot_find_match",
                )
            return failure
    return _result(True)


def _checker_name(test_category: str) -> str:
    """Upstream's per-category checker name, used in ``error_type``."""
    if test_category.endswith("parallel") or "parallel" in test_category:
        return "parallel_function_checker_no_order"
    if "multiple" in test_category:
        return "multiple_function_checker"
    return "simple_function_checker"


def build_chat_messages(
    question_messages: list[dict[str, Any]],
    function_docs: list[dict[str, Any]],
    test_entry_id: str,
) -> list[dict[str, Any]]:
    """Prepend a synthetic BFCL-shaped system prompt to the entry's turn."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        functions=orjson.dumps(function_docs).decode("utf-8")
    )
    return [{"role": "system", "content": system_prompt}, *question_messages]


def decode_calls(response_text: str, language: str) -> list[dict[str, Any]]:
    """Stand-in for ``_bfcl_compat.decode_calls``, raising its error type.

    Lives here rather than in each conftest so the empty-response rule and the
    ``BFCLDecodeError`` wrapping are defined once - both the unit-test fixture
    and the component-integration test patch ``_bfcl_compat.decode_calls`` with
    this, and they must agree on the contract for their results to mean the
    same thing.
    """
    from aiperf.accuracy.graders._bfcl_compat import BFCLDecodeError

    if not response_text or not response_text.strip():
        raise BFCLDecodeError("empty answer channel")
    try:
        return ast_parse(response_text, language)
    except Exception as e:
        raise BFCLDecodeError(f"{type(e).__name__}: {e}") from e
