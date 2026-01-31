# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from aiperf.common.protocols import AIPerfLifecycleProtocol


@runtime_checkable
class AIPerfUIProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol interface definition for AIPerf UI implementations.

    Basically a UI can be any class that implements the AIPerfLifecycleProtocol. However, in order to provide
    progress tracking and worker tracking, the simplest way would be to inherit from the :class:`aiperf.ui.base_ui.BaseAIPerfUI`.
    """
