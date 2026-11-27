# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import ValidationInfo

from aiperf.common.utils import load_json_str
from aiperf.plugin.enums import ServiceType

if TYPE_CHECKING:
    from aiperf.config.dataset.remote_ref import RemoteRef

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


MAX_URL_HEADER_INDEX = 1023
"""Upper bound for a per-URL header index when the URL count is not known."""


def _parse_url_index(raw: Any, source: Any) -> int:
    """Parse a per-URL header index (the 0-based position in the URL list)."""
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    raise ValueError(
        f"User Config: {source!r} - URL index must be a non-negative integer"
    )


def url_index_out_of_range_message(index: int, url_count: int) -> str:
    """Error text for a per-URL header index past the configured URLs."""
    return (
        f"url_headers: URL index {index} is out of range; {url_count} URL(s) "
        f"are configured, so valid indices are 0..{url_count - 1}."
    )


def _set_url_headers(
    output: dict[int, dict[str, str]],
    index: int,
    headers: Any,
    source: Any,
    *,
    url_count: int | None,
) -> None:
    if url_count is not None and index >= url_count:
        raise ValueError(url_index_out_of_range_message(index, url_count=url_count))
    if url_count is None and index > MAX_URL_HEADER_INDEX:
        raise ValueError(
            f"User Config: {source!r} - URL index {index} is above the "
            f"maximum of {MAX_URL_HEADER_INDEX}"
        )
    if not isinstance(headers, dict):
        raise ValueError(
            f"User Config: {source!r} - headers for URL {index} must be a "
            "mapping of header name to value"
        )
    target = output.setdefault(index, {})
    for name, value in headers.items():
        name = str(name).strip()
        if not name:
            raise ValueError(
                f"User Config: {source!r} - header name for URL {index} "
                "must not be empty"
            )
        # Header names are case-insensitive: a later assignment replaces an
        # earlier one regardless of case instead of sending both.
        for existing in [k for k in target if k.lower() == name.lower()]:
            del target[existing]
        target[name] = str(value)


def _merge_url_headers_item(
    output: dict[int, dict[str, str]], item: Any, url_count: int | None
) -> None:
    if isinstance(item, str) and item.lstrip().startswith(("{", "[")):
        try:
            item = load_json_str(item)
        except orjson.JSONDecodeError as e:
            raise ValueError(
                f"User Config: {item} - must be a valid JSON string"
            ) from e
        if isinstance(item, list):
            for index, headers in enumerate(item):
                _set_url_headers(output, index, headers, item, url_count=url_count)
            return
    if isinstance(item, dict):
        for raw_index, headers in item.items():
            _set_url_headers(
                output,
                _parse_url_index(raw_index, item),
                headers,
                item,
                url_count=url_count,
            )
    elif isinstance(item, str):
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                f"User Config: {item!r} - must be in 'INDEX:Header:Value' "
                "format, e.g. '0:X-data-parallel-rank:0'"
            )
        index = _parse_url_index(parts[0], item)
        _set_url_headers(
            output, index, {parts[1]: parts[2].strip()}, item, url_count=url_count
        )
    else:
        raise ValueError(
            f"User Config: {item!r} - must be an 'INDEX:Header:Value' string, "
            "a list of header mappings, or a mapping of URL index to headers"
        )


def parse_url_headers(
    input: Any, url_count: int | None = None
) -> list[dict[str, str]] | None:
    """Parse per-URL headers into a list aligned with the URL list.

    Entry ``i`` holds the headers for the ``i``-th URL; URLs without headers
    get ``{}``. Accepts a list of header mappings (the YAML form), the
    ``INDEX:Header:Value`` CLI form (the value may contain colons), a mapping
    of URL index to headers, a JSON string of either, or a list of CLI
    strings. Later entries for the same header win. Values are kept as
    strings: unlike ``--header`` they are never coerced, so
    ``0:X-data-parallel-rank:0`` sends ``"0"``. Header names are matched
    case-insensitively, so a later ``x-route`` replaces an earlier ``X-Route``.

    Indices are checked against ``url_count`` (when known) before any padding
    is allocated; without it they are capped at ``MAX_URL_HEADER_INDEX``.
    """
    if input is None:
        return None
    output: dict[int, dict[str, str]] = {}
    if isinstance(input, (list, tuple)) and all(isinstance(i, dict) for i in input):
        for index, headers in enumerate(input):
            _set_url_headers(output, index, headers, input, url_count=url_count)
    else:
        for item in input if isinstance(input, (list, tuple)) else [input]:
            _merge_url_headers_item(output, item, url_count)
    size = max(output) + 1 if output else 0
    return [output.get(index, {}) for index in range(size)]


def parse_url_headers_for_urls(input: Any, info: ValidationInfo) -> Any:
    """``parse_url_headers`` bound to the already-validated ``urls`` field."""
    urls = info.data.get("urls")
    url_count = len(urls) if isinstance(urls, (list, tuple)) else None
    return parse_url_headers(input, url_count)


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


def parse_file(value: "str | Path | RemoteRef | None") -> "Path | RemoteRef | None":
    """Parse an existing file/directory path, or a remote reference, from a CLI value.

    Args:
        value: Path string from CLI/config input, or a ``Path`` or ``RemoteRef``
            this function already returned -- accepted as a pass-through so the
            validator is idempotent when a caller round-trips an
            already-parsed field value back through it (e.g. reconstructing
            a model from another instance's field values). ``None`` or an
            empty string disables the path and returns ``None``.

    Returns:
        A ``Path`` pointing to an existing file or directory, a ``RemoteRef`` for
        ``hf://``, ``s3://``, ``https://`` values, or ``None`` when no value was
        provided.

    Raises:
        ValueError: If ``value`` is not a string or Path, or if the path does not
            exist as a file or directory.
    """
    # Imported here: aiperf.common.environment imports this module at load time, and
    # remote_ref pulls in aiperf.config, which would re-enter this module mid-import.
    from aiperf.config.dataset.remote_ref import RemoteRef, is_remote_ref

    if not value:
        return None
    if isinstance(value, RemoteRef):
        return value
    if is_remote_ref(value):
        return RemoteRef(ref=value)
    if isinstance(value, Path):
        path = value
    elif not isinstance(value, str):
        raise ValueError(f"Expected a string, but got {type(value).__name__}")
    else:
        path = Path(value)
    if path.is_file() or path.is_dir():
        return path
    raise ValueError(f"'{value}' is not a valid file or directory")


class RejectedInputFile:
    """An ``--input-file`` value that failed to parse, carried only in redacted form.

    ``raise_rejected_input_file`` raises its reason, so Pydantic echoes this object
    as the error's input, never the raw value, which may hold a presigned signature
    or URL userinfo. Other fields keep their values in validation errors.
    """

    __slots__ = ("reason", "redacted")

    def __init__(self, redacted: str, reason: str) -> None:
        self.redacted = redacted
        self.reason = reason

    def __repr__(self) -> str:
        return repr(self.redacted)


def parse_input_file(value: Any) -> "Path | RemoteRef | RejectedInputFile | None":
    """``parse_file``, turning a failure into a redacted ``RejectedInputFile``."""
    from pydantic import ValidationError

    from aiperf.common.redact import redact_url

    try:
        return parse_file(value)
    except ValueError as e:
        raw = str(value)
        redacted = redact_url(raw)
        if isinstance(e, ValidationError):
            reason = "; ".join(
                err["msg"].removeprefix("Value error, ") for err in e.errors()
            )
        else:
            reason = str(e)
        return RejectedInputFile(redacted, reason.replace(raw, redacted))


def raise_rejected_input_file(value: Any) -> Any:
    """Raise the reason ``parse_input_file`` recorded; pass anything else through."""
    if isinstance(value, RejectedInputFile):
        raise ValueError(value.reason)
    return value


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


# SPEED-Bench category and entropy-tier selectors moved from
# --custom-dataset-type to --public-dataset, where AIPerf resolves the dataset
# itself instead of requiring a separately prepared file. The names were the
# only documented way to run those subsets, so a bare enum error would strand
# anyone with an existing script.
_MOVED_TO_PUBLIC_DATASET = frozenset(
    [
        f"speed_bench_{c}"
        for c in (
            "coding",
            "humanities",
            "math",
            "multilingual",
            "qa",
            "rag",
            "reasoning",
            "roleplay",
            "stem",
            "summarization",
            "writing",
        )
    ]
    + [
        f"speed_bench_throughput_{isl}_{tier}"
        for isl in ("1k", "2k", "8k", "16k", "32k")
        for tier in ("low_entropy", "mixed", "high_entropy")
    ]
)


def reject_moved_custom_dataset_type(input: Any) -> Any:
    """Point relocated dataset-format values at their new flag.

    Runs before enum coercion so the message names the replacement command
    rather than listing every remaining valid value. Shared by the
    ``--custom-dataset-type`` flag and the YAML ``format:`` key, which are two
    spellings of the same choice and must migrate identically.

    Raises:
        ValueError: If the value moved to ``--public-dataset``.
    """
    if isinstance(input, str) and input in _MOVED_TO_PUBLIC_DATASET:
        raise ValueError(
            f"'{input}' is no longer a custom dataset format "
            f"(--custom-dataset-type / YAML 'format:'). SPEED-Bench "
            f"subsets are now selected with --public-dataset, and AIPerf "
            f"resolves the dataset for you, so --input-file is not needed:\n"
            f"    aiperf profile --public-dataset {input} ...\n"
            f"See docs/tutorials/speed-bench.md."
        )
    return input
