# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.common.protocols import CommunicationProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class CommunicationMixin(AIPerfLifecycleMixin, ABC):
    """Mixin to provide access to a CommunicationProtocol instance. This mixin should be inherited
    by any mixin that needs access to the communication layer to create Communication clients.
    """

    def __init__(self, run: BenchmarkRun, **kwargs) -> None:
        super().__init__(run=run, **kwargs)
        self.run = run
        comm_cfg = run.resolved.comm_config or run.cfg.comm_config
        CommClass = plugins.get_class(PluginType.COMMUNICATION, comm_cfg.comm_backend)
        self.comms: CommunicationProtocol = CommClass(config=comm_cfg)
        self.attach_child_lifecycle(self.comms)
