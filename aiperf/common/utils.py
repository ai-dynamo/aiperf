# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import logging
import traceback
from collections.abc import Callable
from typing import Any

import orjson

from aiperf.common.exceptions import AIPerfMultiError

logger = logging.getLogger(__name__)


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
        logger.error("Failed to parse JSON string: '%s'", snippet)
        raise
