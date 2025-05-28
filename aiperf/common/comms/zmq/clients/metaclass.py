# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.base_metaclass import BaseMetaclass, register_metaclass
from aiperf.common.decorators import AIPerfHooks


@register_metaclass(AIPerfHooks.INIT, AIPerfHooks.CLEANUP, AIPerfHooks.TASK)
class ZMQClientMetaclass(BaseMetaclass):
    """Meta class for ZMQ clients.

    This meta class is used to specify the supported hooks for ZMQ clients.
    """

    pass
