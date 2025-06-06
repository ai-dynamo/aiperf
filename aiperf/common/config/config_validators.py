#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from enum import Enum
from pathlib import Path
from typing import Any

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

    if type(input) is str:
        output = [input.strip() for input in input.split(",")]
    elif type(input) is list:
        output = input
    else:
        raise ValueError(f"User Config: {input} - must be a string or list")

    return output


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

    for value in output:
        if not isinstance(value, (int | float)) or value <= 0:
            raise ValueError(
                f"User Config: {output} - all values {value} must be a positive integer or float"
            )

    return output


def parse_goodput(goodputs: dict[str, Any]) -> dict[str, float]:
    """
    Parses and validates a dictionary of goodput values, ensuring that all values
    are non-negative integers or floats, and converts them to floats.
    Args:
        goodputs (Dict[str, Any]): A dictionary where keys are target metric names
            (strings) and values are the corresponding goodput values.
    Returns:
        Dict[str, float]: A dictionary with the same keys as the input, but with
            all values converted to floats.
    Raises:
        ValueError: If any value in the input dictionary is not an integer or float,
            or if any value is negative.
    """

    constraints = {}
    for target_metric, target_value in goodputs.items():
        if isinstance(target_value, (int | float)):
            if target_value < 0:
                raise ValueError(
                    f"User Config: Goodput values must be non-negative ({target_metric}: {target_value})"
                )

            constraints[target_metric] = float(target_value)
        else:
            raise ValueError("User Config: Goodput values must be integers or floats")

    return constraints


def parse_file(value: str | None) -> Path | None:
    """
    Parses the given string value and returns a Path object if the value represents
    a valid file, directory, or a specific synthetic/payload format. Returns None if
    the input value is empty.
    Args:
        value (str): The string value to parse.
    Returns:
        Optional[Path]: A Path object if the value is valid, or None if the value is empty.
    Raises:
        ValueError: If the value is not a valid file or directory and does not match
                    the synthetic/payload format.
    """

    if not value:
        return None
    elif not isinstance(value, str):
        raise ValueError(f"Expected a string, but got {type(value).__name__}")
    elif value.startswith("synthetic:") or value.startswith("payload"):
        return Path(value)
    else:
        path = Path(value)
        if path.is_file() or path.is_dir():
            return path
        else:
            raise ValueError(f"'{value}' is not a valid file or directory")
