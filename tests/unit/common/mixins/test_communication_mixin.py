# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.config.comm.ipc import ZMQIPCConfig


class UsesCommunicationMixin(CommunicationMixin):
    pass


class TestCommunicationMixinCommConfig:
    def test_prefers_resolved_comm_config(self, benchmark_run) -> None:
        cfg_comm = ZMQIPCConfig()
        resolved_comm = ZMQIPCConfig()
        object.__setattr__(benchmark_run.cfg, "_comm_config_cache", cfg_comm)
        benchmark_run.resolved.comm_config = resolved_comm

        comm_instance = MagicMock()
        comm_class = MagicMock(return_value=comm_instance)

        with patch(
            "aiperf.common.mixins.communication_mixin.plugins.get_class",
            return_value=comm_class,
        ):
            component = UsesCommunicationMixin(run=benchmark_run)

        assert component.comms is comm_instance
        comm_class.assert_called_once_with(config=resolved_comm)
