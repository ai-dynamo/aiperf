# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, cast

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommClientType
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.protocols import (
    CommunicationClientProtocol,
    CommunicationProtocol,
    PubClientProtocol,
    PullClientProtocol,
    PushClientProtocol,
    ReplyClientProtocol,
    RequestClientProtocol,
    StreamingDealerClientProtocol,
    StreamingRouterClientProtocol,
    SubClientProtocol,
)
from aiperf.common.types import CommAddressType


@implements_protocol(CommunicationProtocol)
class BaseCommunication(AIPerfLifecycleMixin, ABC):
    """Base class for specifying the base communication layer for AIPerf components."""

    @abstractmethod
    def get_address(self, address_type: CommAddressType) -> str:
        """Get the address for a given address type.

        Args:
            address_type: The type of address to get the address for, or the address itself.

        Returns:
            The address for the given address type, or the address itself if it is a string.
        """

    @abstractmethod
    def create_client(
        self,
        client_type: CommClientType,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        **kwargs: Any,
    ) -> CommunicationClientProtocol:
        """Create a communication client for a given client type and address.

        Args:
            client_type: The type of client to create.
            address: The type of address to use when looking up in the communication config, or the address itself.
            bind: Whether to bind or connect the socket.
            socket_ops: Additional socket options to set.
            max_pull_concurrency: The maximum number of concurrent pull requests to allow. (Only used for pull clients)
            **kwargs: Additional keyword arguments passed to specific client types (e.g., identity for DEALER).
        """

    def create_pub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PubClientProtocol:
        return cast(
            PubClientProtocol,
            self.create_client(CommClientType.PUB, address, bind, socket_ops),
        )

    def create_sub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> SubClientProtocol:
        return cast(
            SubClientProtocol,
            self.create_client(CommClientType.SUB, address, bind, socket_ops),
        )

    def create_push_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PushClientProtocol:
        return cast(
            PushClientProtocol,
            self.create_client(CommClientType.PUSH, address, bind, socket_ops),
        )

    def create_pull_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
    ) -> PullClientProtocol:
        return cast(
            PullClientProtocol,
            self.create_client(
                CommClientType.PULL,
                address,
                bind,
                socket_ops,
                max_pull_concurrency=max_pull_concurrency,
            ),
        )

    def create_request_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> RequestClientProtocol:
        return cast(
            RequestClientProtocol,
            self.create_client(CommClientType.REQUEST, address, bind, socket_ops),
        )

    def create_reply_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> ReplyClientProtocol:
        return cast(
            ReplyClientProtocol,
            self.create_client(CommClientType.REPLY, address, bind, socket_ops),
        )

    def create_streaming_router_client(
        self,
        address: CommAddressType,
        bind: bool = True,
        socket_ops: dict | None = None,
    ) -> StreamingRouterClientProtocol:
        return cast(
            StreamingRouterClientProtocol,
            self.create_client(
                CommClientType.STREAMING_ROUTER, address, bind, socket_ops
            ),
        )

    def create_streaming_dealer_client(
        self,
        address: CommAddressType,
        identity: str,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> StreamingDealerClientProtocol:
        # Identity must be passed through client_kwargs since it's specific to DEALER
        return cast(
            StreamingDealerClientProtocol,
            self.create_client(
                CommClientType.STREAMING_DEALER,
                address,
                bind,
                socket_ops,
                identity=identity,
            ),
        )
