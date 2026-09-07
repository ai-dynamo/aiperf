# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accept ``KEY=VALUE`` and JSON on the ``aiperf kube`` mapping flags.

``KubeOptions`` exposes four ``dict[str, str]`` flags -- ``--annotations``,
``--labels``, ``--env-vars`` and ``--env-from-secrets``. cyclopts binds mapping
fields by dot-notation (``--labels.tier gold``) and has no code path for a bare
token, so ``--labels tier=gold`` reaches ``Argument._json`` carrying an empty
``keys`` tuple and dies on ``token.keys[-1]`` with ``IndexError: tuple index out
of range``. The defect is upstream and unfixed in both cyclopts 4.23.2 (the
pinned release) and the 5.0.0b1 prerelease, so it cannot be resolved by a bump.

Registering this module's hook as a cyclopts ``config`` callable places it after
token parsing and before conversion -- the one point where the tokens are
reachable and ``Argument._json`` has not run yet. Bare ``KEY=VALUE`` and
JSON-object tokens are rewritten into the keyed tokens cyclopts expects, so both
spellings work; anything else raises a usage error naming every accepted
spelling instead of crashing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, get_origin

import orjson
from cyclopts.exceptions import CoercionError
from cyclopts.token import Token

if TYPE_CHECKING:
    from cyclopts import App
    from cyclopts.argument import Argument, ArgumentCollection


def _binds_keys_by_dot_notation(argument: Argument) -> bool:
    """Is this a mapping flag whose keys cyclopts binds via dot-notation?"""
    if not argument.parse or argument.parameter.accepts_keys is False:
        return False
    origin = get_origin(argument.hint) or argument.hint
    return isinstance(origin, type) and issubclass(origin, Mapping)


def _usage_error(argument: Argument, token: Token) -> CoercionError:
    """Build a usage error naming every spelling the flag accepts."""
    flag = token.keyword or argument.name
    return CoercionError(
        msg=(
            f"{token.value!r} is not a KEY=VALUE pair. Supply "
            f"{flag} KEY=VALUE, {flag}.KEY VALUE, or "
            f'{flag} \'{{"KEY": "VALUE"}}\'.'
        ),
        token=token,
        argument=argument,
    )


def _json_value_as_str(value: Any) -> str | None:
    """Render a JSON scalar as the string a ``dict[str, str]`` field needs."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return None


def _rekey_json_object(argument: Argument, token: Token) -> list[Token]:
    """Split a JSON-object token into one keyed token per entry."""
    try:
        parsed = orjson.loads(token.value)
    except orjson.JSONDecodeError as exc:
        raise _usage_error(argument, token) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise _usage_error(argument, token)

    tokens = []
    for key, value in parsed.items():
        rendered = _json_value_as_str(value)
        if not isinstance(key, str) or rendered is None:
            raise CoercionError(
                msg=(
                    f"{token.keyword or argument.name} JSON must map string keys "
                    f"to string values; got {key!r}: {value!r}."
                ),
                token=token,
                argument=argument,
            )
        tokens.append(token.evolve(keys=(*token.keys, key), value=rendered))
    return tokens


def _rekey(argument: Argument, token: Token) -> list[Token]:
    """Rewrite one bare mapping token into the keyed tokens cyclopts expects."""
    raw = token.value.strip()
    if raw.startswith("{"):
        return _rekey_json_object(argument, token.evolve(value=raw))
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise _usage_error(argument, token)
    return [token.evolve(keys=(*token.keys, key), value=value)]


def normalize_mapping_flag_tokens(
    app: App,
    commands: tuple[str, ...],
    arguments: ArgumentCollection,
) -> None:
    """Re-key bare tokens on dot-notation mapping flags before conversion.

    Registered via ``App(config=...)``, which cyclopts invokes once per command
    dispatch between token parsing and conversion.

    Args:
        app: Command app being dispatched. Unused; part of the cyclopts config
            callable signature.
        commands: Resolved command chain. Unused; part of the same signature.
        arguments: Parsed arguments, mutated in place.

    Raises:
        CoercionError: A bare token is neither ``KEY=VALUE`` nor a JSON object.
    """
    for argument in arguments:
        if not _binds_keys_by_dot_notation(argument):
            continue
        rewritten: list[Token] = []
        rekeyed_any = False
        for token in argument.tokens:
            # An explicitly-supplied empty mapping carries no keys by design.
            if token.keys or isinstance(token.implicit_value, dict):
                rewritten.append(token)
                continue
            rekeyed_any = True
            rewritten.extend(_rekey(argument, token))
        if rekeyed_any:
            argument.tokens = rewritten
