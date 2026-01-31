# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin


class NoUI(AIPerfLifecycleMixin):
    """
    A UI that does nothing.

    Implements the :class:`AIPerfUIProtocol` to allow it to be used as a UI, but provides no functionality.

    NOTE: Not inheriting from :class:`BaseAIPerfUI` because it does not need to track progress or workers.
    """
