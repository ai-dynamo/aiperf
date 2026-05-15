# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.utils import load_json_str
from aiperf.plugin.enums import ServiceType

"""
This module provides utility functions for validating and parsing configuration inputs.
"""


# Non-HTTP URI schemes that must never be silently rewritten to ``http://`` —
# doing so would either smuggle a foreign scheme past the validator (e.g.
# ``javascript:1234`` becoming ``http://javascript:1234``) or corrupt it into
# an "invalid port" message. The ``://`` check below already handles forms
# like ``ftp://host`` and ``file:///path``; this list catches the
# ``scheme:opaque`` forms that lack the authority separator.
_FOREIGN_URI_SCHEMES = frozenset(
    {
        "data",
        "file",
        "ftp",
        "ftps",
        "gopher",
        "javascript",
        "ldap",
        "ldaps",
        "mailto",
        "sftp",
        "ssh",
        "tel",
        "vbscript",
        "ws",
        "wss",
    }
)


def normalize_http_url(url: str) -> str:
    """Prepend ``http://`` to a URL that has no scheme component.

    aiohttp rejects URLs without a recognized scheme with NonHttpUrlClientError.
    Users commonly pass ``--url localhost:8000`` expecting an implicit scheme;
    this normalization accepts that form.

    A URL is considered to "already have a scheme" if it contains ``://`` —
    the structural separator between scheme and authority. This is preferred
    over ``urlsplit(url).scheme`` because urlsplit parses ``localhost:8000``
    as scheme=``localhost``, which would defeat the normalization. The
    ``://`` check is also case-insensitive (matches ``http://``, ``HTTPS://``,
    ``ftp://`` etc.) and preserves the original scheme; URLs with non-HTTP
    schemes are passed through unmodified rather than corrupted.

    A bare ``scheme:opaque`` form (no ``://``) like ``javascript:alert(1)``
    or ``data:text/plain;...`` is left alone when the prefix is a recognized
    foreign URI scheme, so the downstream validator surfaces "missing scheme
    or host" instead of silently rewriting the input to ``http://javascript:...``.
    """
    if "://" in url:
        return url
    prefix, sep, _ = url.partition(":")
    if sep and prefix.lower() in _FOREIGN_URI_SCHEMES:
        return url
    return f"http://{url}"


def normalize_http_urls(urls: list[str]) -> list[str]:
    """Apply :func:`normalize_http_url` to each URL in the list."""
    return [normalize_http_url(u) for u in urls]


def parse_str_or_list(input: Any) -> list[Any] | None:
    """
    Parses the input to ensure it is either a string, a list, or None. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it is returned as-is. If the input
    is None, it is returned as-is. If the input is none of these, a ValueError is raised.
    Args:
        input (Any): The input to be parsed. Expected to be a string, a list, or None.
    Returns:
        list | None: A list of strings derived from the input, or None if input is None.
    Raises:
        ValueError: If the input is neither a string, a list, nor None.
    """
    if input is None:
        return None
    elif isinstance(input, str):
        output = [item.strip() for item in input.split(",")]
    elif isinstance(input, list):
        # TODO: When using cyclopts, the values are already lists, so we have to split them by commas.
        output = []
        for item in input:
            if isinstance(item, str):
                output.extend([token.strip() for token in item.split(",")])
            else:
                output.append(item)
    else:
        raise ValueError(f"User Config: {input} - must be a string, list, or None")

    return output


def parse_str_or_csv_list(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is either a string or a list. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it will split each item by commas
    and trim any whitespace around each element, returning the combined result as a list.
    If the input is neither a string nor a list, a ValueError is raised.

    [1, 2, 3] -> [1, 2, 3]
    "1,2,3" -> ["1", "2", "3"]
    ["1,2,3", "4,5,6"] -> ["1", "2", "3", "4", "5", "6"]
    ["1,2,3", 4, 5] -> ["1", "2", "3", 4, 5]
    """
    if isinstance(input, str):
        output = [item.strip() for item in input.split(",")]
    elif isinstance(input, list):
        output = []
        for item in input:
            if isinstance(item, str):
                output.extend([token.strip() for token in item.split(",")])
            else:
                output.append(item)
    else:
        raise ValueError(f"User Config: {input} - must be a string or list")

    return output


def parse_service_types(input: Any | None) -> set[ServiceType] | None:
    """Parses the input to ensure it is a set of service types.
    Will replace hyphens with underscores for user convenience."""
    if input is None:
        return None

    return {
        ServiceType(service_type.replace("-", "_"))
        for service_type in parse_str_or_csv_list(input)
    }


def coerce_value(value: Any) -> Any:
    """Coerce the value to the correct type."""
    if not isinstance(value, str):
        return value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("none", "null"):
        return None
    if value.isdigit() and (not value.startswith("0") or value == "0"):
        return int(value)
    if (
        value.startswith("-")
        and value[1:].isdigit()
        and (not value.startswith("-0") or value == "-0")
    ):
        return int(value)
    if value.count(".") == 1 and (
        value.replace(".", "").isdigit()
        or (value.startswith("-") and value[1:].replace(".", "").isdigit())
    ):
        return float(value)
    return value


def _parse_sequence_as_tuple_list(input: Any) -> list[tuple[str, Any]]:
    """Parse a list/tuple/set into a list of (key, value) tuples."""
    output: list[tuple[str, Any]] = []
    for item in input:
        # If item is already a 2-element sequence (key-value pair), convert directly to tuple
        if isinstance(item, (list, tuple)) and len(item) == 2:
            key, value = item
            output.append((str(key), coerce_value(value)))
        else:
            res = parse_str_or_dict_as_tuple_list(item)
            if res is not None:
                output.extend(res)
    return output


def _parse_str_as_tuple_list(input: str) -> list[tuple[str, Any]]:
    """Parse a string (JSON object or comma-separated key:value pairs) into a list of tuples."""
    if input.startswith("{"):
        try:
            return list(load_json_str(input).items())
        except orjson.JSONDecodeError as e:
            raise ValueError(
                f"User Config: {input} - must be a valid JSON string"
            ) from e

    result: list[tuple[str, Any]] = []
    for item in input.split(","):
        parts = item.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"User Config: {input} - each item must be in 'key:value' format"
            )
        key, value = parts
        result.append((key.strip(), coerce_value(value.strip())))
    return result


def parse_str_or_dict_as_tuple_list(input: Any | None) -> list[tuple[str, Any]] | None:
    """
    Parses the input to ensure it is a list of tuples. (key, value) pairs.

    - If the input is a string:
        - If the string starts with a '{', it is parsed as a JSON string.
        - Otherwise, it splits the string by commas and then for each item, it splits the item by colons
        into key and value, trims any whitespace, and coerces the value to the correct type.
    - If the input is a dictionary, it is converted to a list of tuples by key and value pairs.
    - If the input is a list, it recursively calls this function on each item, and aggregates the results.
        - If the item is already a 2-element sequence (key-value pair), it is converted directly to a tuple.
    - Otherwise, a ValueError is raised.

    Args:
        input (Any): The input to be parsed. Expected to be a string, list, or dictionary.
    Returns:
        list[tuple[str, Any]]: A list of tuples derived from the input.
    Raises:
        ValueError: If the input is neither a string, list, nor dictionary, or if the parsing fails.
    """
    if input is None:
        return None
    if isinstance(input, (list, tuple, set)):
        return _parse_sequence_as_tuple_list(input)
    if isinstance(input, dict):
        return [(key, coerce_value(value)) for key, value in input.items()]
    if isinstance(input, str):
        return _parse_str_as_tuple_list(input)

    raise ValueError(f"User Config: {input} - must be a valid string, list, or dict")


def print_str_or_list(input: Any) -> str:
    """Convert a list, Enum, or scalar to a display string.

    Lists become comma-separated strings; Enums return their lowercased value;
    other types are returned unchanged. Used for rendering config values in
    user-facing CLI output.
    """
    if isinstance(input, list):
        return ", ".join(map(str, input))
    elif isinstance(input, Enum):
        return str(input.value).lower()
    return input


def parse_str_or_list_of_positive_values(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is a list of positive integers or floats.
    This function first converts the input into a list using `parse_str_or_list`.
    It then validates that each value in the list is either an integer or a float
    and that all values are strictly greater than zero. If any value fails this
    validation, a `ValueError` is raised.
    Args:
        input (Any): The input to be parsed. It can be a string or a list.
    Returns:
        List[Any]: A list of positive integers or floats.
    Raises:
        ValueError: If any value in the parsed list is not a positive integer or float,
                    or if the input is None.
    """
    # Guard against None before calling parse_str_or_list to provide clear error
    if input is None:
        raise ValueError("input must be a string or list of strings, not None")

    output = parse_str_or_list(input)

    # Additional safety check (should not be reached due to above check, but defensive)
    if output is None:
        raise ValueError("input must be a string or list of strings, not None")

    try:
        output = [
            float(x) if "." in str(x) or "e" in str(x).lower() else int(x)
            for x in output
        ]
    except ValueError as e:
        raise ValueError(f"User Config: {output} - all values must be numeric") from e

    if not all(isinstance(x, ((int, float))) and x > 0 for x in output):
        raise ValueError(f"User Config: {output} - all values must be positive numbers")

    return output


def parse_file(value: str | None) -> Path | None:
    """Parse an existing file/directory path from a CLI value.

    Args:
        value: Path string from CLI/config input. ``None`` or an empty string
            disables the path and returns ``None``.

    Returns:
        A ``Path`` pointing to an existing file or directory, or ``None`` when no
        value was provided.

    Raises:
        ValueError: If ``value`` is not a string, or if the path does not exist as
            a file or directory.
    """

    if not value:
        return None
    elif not isinstance(value, str):
        raise ValueError(f"Expected a string, but got {type(value).__name__}")
    else:
        path = Path(value)
        if path.is_file() or path.is_dir():
            return path
        else:
            raise ValueError(f"'{value}' is not a valid file or directory")


def validate_sequence_distribution(v: str | None) -> str | None:
    """Validate sequence distribution format, returns original value if valid."""
    # Only validate the format, don't create the full distribution yet.
    # This avoids requiring RNG initialization during config validation.
    if v is not None:
        from aiperf.common.models.sequence_distribution import DistributionParser

        DistributionParser.validate(v)
    return v


def _parse_numeric_dict_item(item: str) -> tuple[str, float]:
    """Parse a single 'key:value' token into (key, float(value))."""
    if not item or ":" not in item:
        raise ValueError(f"User Config: '{item}' is not in 'key:value' format")
    key, val = item.split(":", 1)
    key, val = key.strip(), val.strip()
    if not key:
        raise ValueError(f"User Config: '{item}' has an empty key")
    if not val:
        raise ValueError(f"User Config: '{item}' has an empty value")
    try:
        return key, float(val)
    except ValueError as e:
        raise ValueError(
            f"User Config: value for '{key}' must be numeric, got '{val}'"
        ) from e


def parse_str_as_numeric_dict(
    input_string: str | dict | None,
) -> dict[str, float] | None:
    """
    Parse a string of key:value pairs such as 'k:v x:y' into {k: v, x: y}.
    """
    if input_string is None:
        return None
    if isinstance(input_string, dict):
        try:
            return {k: float(v) for k, v in input_string.items()}
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"User Config: goodput dict values must be numeric, got {input_string!r}"
            ) from e
    if not isinstance(input_string, str):
        raise ValueError(
            f"User Config: expected a string of space-separated 'key:value' pairs, got {type(input_string).__name__}"
        )

    input_string = input_string.strip()
    if not input_string:
        raise ValueError(
            "User Config: expected space-separated 'key:value' pairs (e.g., 'k:v x:y'), got empty string"
        )

    return dict(_parse_numeric_dict_item(item) for item in input_string.split())


def parse_int_or_int_list(input: Any) -> int | list[int] | None:
    """Parse a CLI value into ``int`` or ``list[int]`` (or None if absent).

    Accepts ``None``, ``int``, comma-separated strings (collapsed to a single
    int when one element survives stripping), and explicit lists. Rejects
    ``bool`` (which is an ``int`` subclass) so flags can't sneak in as 0/1.
    """
    if input is None:
        return None
    if isinstance(input, bool):
        raise TypeError(f"User Config: got bool {input!r}; expected int or list[int]")
    if isinstance(input, int):
        return input
    if isinstance(input, str):
        s = input.strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1:
                return int(parts[0])
            return [int(p) for p in parts]
        return int(s)
    if isinstance(input, (list, tuple)):
        items = [int(v) for v in input]
        return items[0] if len(items) == 1 else items
    raise TypeError(f"cannot parse {type(input).__name__} as int / list[int]")


def parse_float_or_float_list(input: Any) -> float | list[float] | None:
    """Parse a CLI value into ``float`` or ``list[float]`` (or None if absent).

    Mirror of :func:`parse_int_or_int_list` for ``--request-rate`` and any
    other float-valued magic-list flag. Accepts ``None``, ``int``/``float``
    scalars, comma-separated strings (collapsed to a single float when one
    element survives stripping), and explicit lists. Rejects ``bool``
    (which is an ``int`` subclass) so flags can't sneak in as 0.0/1.0.
    """
    if input is None:
        return None
    if isinstance(input, bool):
        raise TypeError(
            f"User Config: got bool {input!r}; expected float or list[float]"
        )
    if isinstance(input, (int, float)):
        return float(input)
    if isinstance(input, str):
        s = input.strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1:
                return float(parts[0])
            return [float(p) for p in parts]
        return float(s)
    if isinstance(input, (list, tuple)):
        items = [float(v) for v in input]
        return items[0] if len(items) == 1 else items
    raise TypeError(f"cannot parse {type(input).__name__} as float / list[float]")


def require_turn_mean_at_least_one(value: Any) -> Any:
    """AfterValidator: every scalar or list element must be >= 1.

    The CLI flag's BeforeValidator (parse_int_or_int_list) yields int or
    list[int], so per-element checking here catches `--conversation-turn-mean 0`
    AND `--conversation-turn-mean 1,0,4` (the latter would otherwise slip
    through because the placeholder lands as the first element, leaving 0
    only to materialize at sweep-expansion time on individual cells).
    """
    if value is None:
        return value
    elements = value if isinstance(value, list) else [value]
    for v in elements:
        if v < 1:
            raise ValueError(
                f"--conversation-turn-mean must be >= 1 for every value (got {v}); "
                "use --conversation-turn-mean 1 or omit it for single-turn conversations."
            )
    return value
