# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import orjson
from cyclopts.token import Token

from aiperf.common.enums import ServiceType
from aiperf.common.utils import load_json_str

"""
This module provides utility functions for validating and parsing configuration inputs.
"""


def parse_str_or_list(input: Any) -> list[Any]:
    """
    Parses the input to ensure it is either a string or a list. If the input is a string,
    it splits the string by commas and trims any whitespace around each element, returning
    the result as a list. If the input is already a list, it is returned as-is. If the input
    is neither a string nor a list, a ValueError is raised.
    Args:
        input (Any): The input to be parsed. Expected to be a string or a list.
    Returns:
        list: A list of strings derived from the input.
    Raises:
        ValueError: If the input is neither a string nor a list.
    """
    if isinstance(input, str):
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
        raise ValueError(f"User Config: {input} - must be a string or list")

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
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    if value.count(".") == 1 and (
        value.replace(".", "").isdigit()
        or (value.startswith("-") and value[1:].replace(".", "").isdigit())
    ):
        return float(value)
    return value


def parse_str_or_dict_as_tuple_list(input: Any | None) -> list[tuple[str, Any]] | None:
    """
    Parses the input to ensure it is a list of tuples. (key, value) pairs.

    - If the input is a string:
        - If the string starts with a '{', it is parsed as a JSON string.
        - Otherwise, it splits the string by commas and then for each item, it splits the item by colons
        into key and value, trims any whitespace, and coerces the value to the correct type.
    - If the input is a dictionary, it is converted to a list of tuples by key and value pairs.
    - If the input is a list, it recursively calls this function on each item, and aggregates the results.
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

    if isinstance(input, list | tuple | set):
        output = []
        for item in input:
            res = parse_str_or_dict_as_tuple_list(item)
            if res is not None:
                output.extend(res)
        return output

    if isinstance(input, dict):
        return [(key, coerce_value(value)) for key, value in input.items()]

    if isinstance(input, str):
        if input.startswith("{"):
            try:
                return [(key, value) for key, value in load_json_str(input).items()]
            except orjson.JSONDecodeError as e:
                raise ValueError(
                    f"User Config: {input} - must be a valid JSON string"
                ) from e
        else:
            return [
                (key.strip(), coerce_value(value.strip()))
                for item in input.split(",")
                for key, value in [item.split(":")]
            ]

    raise ValueError(f"User Config: {input} - must be a valid string, list, or dict")


def print_str_or_list(input: Any) -> str:
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
        ValueError: If any value in the parsed list is not a positive integer or float.
    """

    output = parse_str_or_list(input)

    try:
        output = [
            float(x) if "." in str(x) or "e" in str(x).lower() else int(x)
            for x in output
        ]
    except ValueError as e:
        raise ValueError(f"User Config: {output} - all values must be numeric") from e

    if not all(isinstance(x, (int | float)) and x > 0 for x in output):
        raise ValueError(f"User Config: {output} - all values must be positive numbers")

    return output


def parse_file(value: str | None) -> Path | None:
    """
    Parses the given string value and returns a Path object if the value represents
    a valid file or directory. Returns None if the input value is empty.
    Args:
        value (str): The string value to parse.
    Returns:
        Optional[Path]: A Path object if the value is valid, or None if the value is empty.
    Raises:
        ValueError: If the value is not a valid file or directory.
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


def custom_enum_converter(type_: Any, value: Sequence[Token]) -> Any:
    """This is a custom converter for cyclopts that allows us to use our custom enum types"""
    if len(value) != 1:
        raise ValueError(f"Expected 1 value, but got {len(value)}")
    return type_(value[0].value)
