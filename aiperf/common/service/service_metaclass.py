# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.base_metaclass import BaseMetaclass, register_metaclass
from aiperf.common.decorators import AIPerfHooks


@register_metaclass(
    AIPerfHooks.INIT,
    AIPerfHooks.START,
    AIPerfHooks.STOP,
    AIPerfHooks.CLEANUP,
    AIPerfHooks.RUN,
    AIPerfHooks.CONFIGURE,
    AIPerfHooks.TASK,
    AIPerfHooks.SET_STATE,
)
class ServiceMetaclass(BaseMetaclass):
    """Meta class for services.

    This meta class is used to collect the hooks for a service. All of the logic
    for the hooks is implemented in the BaseMetaclass.
    """

    pass
