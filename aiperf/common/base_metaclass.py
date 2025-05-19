#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from aiperf.common.decorators import AIPerfHooks
from aiperf.common.exceptions.base_exceptions import AIPerfMetaclassError


class BaseMetaclass(ABCMeta):
    """Base metaclass for all AIPerf metaclasses that support hooks.

    This metaclass is used to collect the hooks for a service. All of the logic
    for the hooks is implemented here, and other metaclasses should inherit from
    this one, and specify the supported hook types via the `register_metaclass`
    decorator.
    """

    _supported_hook_types: list[AIPerfHooks] = []
    _strict_hook_types: bool = True

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        _aiperf_hooks: dict[str, list[Callable]] = defaultdict(list)

        # Add hooks from the current class by looking through each element
        # in the namespace and checking if it has a hook type attribute.
        for _, attr_value in namespace.items():
            hook_type = getattr(attr_value, AIPerfHooks.HOOK_TYPE, None)
            if hook_type is None:
                continue

            # If strict hook types are enabled, and the hook type is not in the
            # list of supported hook types, raise an error. This is to prevent
            # issues that arise from using a hook type that is not supported,
            # and wondering why it is not working.
            if mcs._strict_hook_types and hook_type not in mcs._supported_hook_types:
                raise AIPerfMetaclassError(
                    f"Invalid hook type: {hook_type} for {name} ({attr_value})"
                )

            _aiperf_hooks[hook_type].append(attr_value)

        # Add hooks from all base classes. This is done to allow for hooks to be
        # defined in base classes and still be inherited by derived classes.
        for base in bases:
            if hasattr(base, "_aiperf_hooks"):
                for hook_type, hooks in base._aiperf_hooks.items():
                    _aiperf_hooks[hook_type].extend(hooks)

        namespace["_aiperf_hooks"] = _aiperf_hooks

        cls = super().__new__(mcs, name, bases, namespace)
        return cls


def register_metaclass(
    *supported_hook_types: AIPerfHooks,
    strict: bool = True,
) -> Callable[[type], type]:
    """Decorator to register a metaclass with the AIPerf framework.

    It will add the supported hook types to the metaclass and set whether the
    hook types are strict or not.

    Args:
        supported_hook_types: The hook types that the metaclass supports.
        strict: Whether the hook types are strict or not. If strict is True
            (default: True), then the metaclass will raise an error if an invalid
            hook type is used. This is to prevent issues that arise from using
            a hook type that is not supported, and wondering why it is not
            working.

    Returns:
        The decorated metaclass.
    """

    def decorator(cls: type[BaseMetaclass]) -> type[BaseMetaclass]:
        cls._supported_hook_types = list(supported_hook_types)
        cls._strict_hook_types = strict
        return cls

    return decorator
