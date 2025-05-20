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
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.enums.base_enums import StrEnum
from aiperf.common.enums.service_enums import ServiceState


def on_cleanup(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during
    cleanup.

    It will be called during the service shutdown to allow the derived service to
    clean up any resources specific to that service.
    """

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.CLEANUP)
    return func


def on_init(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during
    initialization.

    It will be called during the service initialization to allow the derived class
    to set up any resources specific to that subclass.
    """

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.INIT)
    return func


def on_comms_init(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during
    communication initialization.

    It will be called after the communication is initialized to allow the derived
    classes to set up any resources specific to that subclass.
    """

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.COMMS_INIT)
    return func


def on_stop(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during stop.

    It will be called during the service shutdown to allow the derived service to
    clean up any resources specific to that service.
    """

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.STOP)
    return func


def on_start(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during start.

    It will be called during the service start to allow the derived service to
    start any processes or components specific to that service.
    """

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.START)
    return func


def on_configure(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during the service
    configuration.

    It will be called during the service configuration to allow the derived service
    to set up any resources specific to that service.
    """
    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.CONFIGURE)
    return func


def on_run(func: Callable) -> Callable:
    """
    Decorator to specify that the function should be called during run.

    It will be called during the service run to allow the derived service to
    run any processes or components specific to that service.
    """
    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.RUN)
    return func


def on_set_state(
    func: Callable[[Any, ServiceState], Coroutine[Any, Any, None]],
) -> Callable[[Any, ServiceState], Coroutine[Any, Any, None]]:
    """
    Decorator to specify that the function should be called when the service state
    is set.

    It will be called when the service state is set to allow the derived service to
    take any action based on the new state. The new state is passed as an argument
    to the function.
    """
    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.SET_STATE)
    return func


def aiperf_task(func: Callable, interval: float | None = None) -> Callable:
    """
    Decorator to indicate that the function is a task function. It will be
    started automatically and stopped when the service shuts down.

    It allows a derived class to run any tasks specific to that class. An
    interval can be provided to indicate the frequency at which the task should
    be run, otherwise it will run once.

    Args:
        func: The function to decorate.
        interval: The interval at which the task should be run. If the interval is
            None, the task will be run once.
    """
    # TODO: task intervals are not supported yet

    setattr(func, AIPerfHooks.HOOK_TYPE, AIPerfHooks.TASK)
    setattr(func, AIPerfHooks.TASK_INTERVAL, interval)
    return func


class AIPerfHooks(StrEnum):
    """
    Internal constants for the various AIPerf decorators.

    Note: If you add a new hook, you must also add it to the @register_metaclass
    decorators in the metaclasses you wish to use the hook in.
    """

    HOOK_TYPE = "__aiperf_hook_type__"

    CLEANUP = "__aiperf_on_cleanup__"
    INIT = "__aiperf_on_init__"
    COMMS_INIT = "__aiperf_on_comms_init__"
    STOP = "__aiperf_on_stop__"
    START = "__aiperf_on_start__"
    CONFIGURE = "__aiperf_on_configure__"
    RUN = "__aiperf_on_run__"
    TASK = "__aiperf_task__"
    TASK_INTERVAL = "__aiperf_task_interval__"
    SET_STATE = "__aiperf_on_set_state__"
