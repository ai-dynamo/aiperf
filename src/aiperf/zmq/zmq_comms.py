# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ-based implementation of the CommunicationProtocol."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import zmq.asyncio

from aiperf.common.base_comms import BaseCommunication
from aiperf.common.config import ZMQDualBindConfig, ZMQIPCConfig, ZMQTCPConfig
from aiperf.common.enums import CommAddress, LifecycleState
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.singleton import Singleton
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.common.config import BaseZMQCommunicationConfig
    from aiperf.common.protocols import CommunicationClientProtocol
    from aiperf.common.types import CommAddressType
    from aiperf.plugin.enums import CommClientType


class BaseZMQCommunication(BaseCommunication, AIPerfLoggerMixin, ABC, Singleton):
    """ZeroMQ-based implementation of the CommunicationProtocol.

    Uses ZeroMQ for publish/subscribe, request/reply, and pull/push patterns to
    facilitate communication between AIPerf components.
    """

    def __init__(
        self,
        config: BaseZMQCommunicationConfig,
    ) -> None:
        super().__init__()
        self.config = config

        self.context = zmq.asyncio.Context.instance()
        self._clients_cache: dict[
            tuple[CommClientType, CommAddressType, bool], CommunicationClientProtocol
        ] = {}

        self._ensure_ipc_directory()
        self.debug(f"ZMQ communication using protocol: {type(self.config).__name__}")

    def _ensure_ipc_directory(self) -> None:
        """Create IPC socket directory if the config specifies one."""
        if self.config.ipc_path and not self.config.ipc_path.exists():
            self.config.ipc_path.mkdir(parents=True, exist_ok=True)

    def get_address(self, address_type: CommAddressType) -> str:
        """Get the actual address based on the address type from the config."""
        if isinstance(address_type, CommAddress):
            return self.config.get_address(address_type)
        return address_type

    def create_client(
        self,
        client_type: CommClientType,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> CommunicationClientProtocol:
        """Create a communication client for a given client type and address.

        Args:
            client_type: The type of client to create.
            address: The type of address to use when looking up in the communication config, or the address itself.
            bind: Whether to bind or connect the socket.
            socket_ops: Additional socket options to set.
            max_pull_concurrency: The maximum number of concurrent pull requests to allow. (Only used for pull clients)
            additional_bind_address: Optional second address to bind to for dual-bind mode (e.g., IPC + TCP).
        """
        if (client_type, address, bind) in self._clients_cache:
            return self._clients_cache[(client_type, address, bind)]

        if self.state != LifecycleState.CREATED:
            # We require the clients to be created before the communication class is initialized.
            # This is because this class manages the lifecycle of the clients of as well.
            raise InvalidStateError(
                f"Communication clients must be created before the {self.__class__.__name__} "
                f"class is initialized: {self.state!r}"
            )

        ClientClass = plugins.get_class(PluginType.COMMUNICATION_CLIENT, client_type)
        client = ClientClass(
            address=self.get_address(address),
            bind=bind,
            socket_ops=socket_ops,
            max_pull_concurrency=max_pull_concurrency,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )

        self._clients_cache[(client_type, address, bind)] = client
        self.attach_child_lifecycle(client)
        return client


class ZMQTCPCommunication(BaseZMQCommunication, Singleton):
    """ZeroMQ-based implementation of the Communication interface using TCP transport."""

    def __init__(self, config: ZMQTCPConfig | None = None) -> None:
        """Initialize ZMQ TCP communication.

        Args:
            config: ZMQTCPTransportConfig object with configuration parameters
        """
        super().__init__(config or ZMQTCPConfig())


class ZMQDualBindCommunication(BaseZMQCommunication, Singleton):
    """ZeroMQ-based implementation of the Communication interface using dual-bind transport."""

    def __init__(self, config: ZMQDualBindConfig | None = None) -> None:
        """Initialize ZMQ dual-bind communication.

        Args:
            config: ZMQDualBindConfig object with configuration parameters
        """
        super().__init__(config or ZMQDualBindConfig())


class ZMQIPCCommunication(BaseZMQCommunication, Singleton):
    """ZeroMQ-based implementation of the Communication interface using IPC transport."""

    def __init__(self, config: ZMQIPCConfig | None = None) -> None:
        """Initialize ZMQ IPC communication.

        Args:
            config: ZMQIPCConfig object with configuration parameters
        """
        super().__init__(config or ZMQIPCConfig())
