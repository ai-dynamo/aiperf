# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
import os
import traceback
from collections.abc import Callable
from typing import Any

import orjson

from aiperf.common import aiperf_logger
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import AIPerfMultiError

_logger = AIPerfLogger(__name__)


async def call_all_functions_self(
    self_: object, funcs: list[Callable], *args, **kwargs
) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(self_, *args, **kwargs)
            else:
                func(self_, *args, **kwargs)
        except Exception as e:
            # TODO: error handling, logging
            traceback.print_exc()
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


async def call_all_functions(funcs: list[Callable], *args, **kwargs) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        except Exception as e:
            # TODO: error handling, logging
            traceback.print_exc()
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


def load_json_str(json_str: str, func: Callable = lambda x: x) -> dict[str, Any]:
    """
    Deserializes JSON encoded string into Python object.

    Args:
      - json_str: string
          JSON encoded string
      - func: callable
          A function that takes deserialized JSON object. This can be used to
          run validation checks on the object. Defaults to identity function.
    """
    try:
        # Note: orjson may not parse JSON the same way as Python's standard json library,
        # notably being stricter on UTF-8 conformance.
        # Refer to https://github.com/ijl/orjson?tab=readme-ov-file#str for details.
        return func(orjson.loads(json_str))
    except orjson.JSONDecodeError:
        snippet = json_str[:200] + ("..." if len(json_str) > 200 else "")
        _logger.error(f"Failed to parse JSON string: '{snippet}'")
        raise


async def yield_to_event_loop() -> None:
    """Yield to the event loop. This forces the current coroutine to yield and allow
    other coroutines to run, preventing starvation. Use this when you do not want to
    delay your coroutine via sleep, but still want to allow other coroutines to run if
    there is a potential for an infinite loop.
    """
    await asyncio.sleep(0)


def close_enough(a: Any, b: Any, epsilon: float = 1e-9) -> bool:
    """Check if two objects are close enough to each other to be considered equal."""
    # If they are both numerical, compare them with a small epsilon
    if isinstance(a, float | int) and isinstance(b, float | int):
        return abs(a - b) < epsilon

    a_is_list = isinstance(a, list | tuple)
    b_is_list = isinstance(b, list | tuple)

    # If they are both lists, compare each element pairwise
    if a_is_list and b_is_list:
        if len(a) != len(b):
            raise ValueError(f"Lists must have the same length to compare: {a} and {b}")
        return all(close_enough(a, b, epsilon) for a, b in zip(a, b, strict=True))

    # If one is a list, compare each element of the list with the other object
    if a_is_list:
        return all(close_enough(a, b, epsilon) for a in a)
    if b_is_list:
        return all(close_enough(a, b, epsilon) for b in b)

    # Otherwise, try and compare them as objects
    return a == b


# This is used to identify the source file of the call_all_functions function
# in the AIPerfLogger class to skip it when determining the caller.
# NOTE: Using similar logic to logging._srcfile
_srcfile = os.path.normcase(call_all_functions.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)
