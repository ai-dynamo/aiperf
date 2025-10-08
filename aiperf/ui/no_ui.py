# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.decorators import implements_protocol
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.protocols import AIPerfUIProtocol


@implements_protocol(AIPerfUIProtocol)
class NoUI(AIPerfLifecycleMixin):
    """
    A UI that does nothing.

    Implements the :class:`AIPerfUIProtocol` to allow it to be used as a UI, but provides no functionality.

    NOTE: Not inheriting from :class:`BaseAIPerfUI` because it does not need to track progress or workers.
    """
