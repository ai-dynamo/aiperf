# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, Field

from aiperf.common.enums import CommAddress
from aiperf.plugin.enums import CommunicationBackend


class BaseZMQProxyConfig(BaseModel, ABC):
    """Configuration Protocol for ZMQ Proxy."""

    @property
    @abstractmethod
    def frontend_address(self) -> str:
        """Get the frontend address based on protocol configuration."""

    @property
    @abstractmethod
    def backend_address(self) -> str:
        """Get the backend address based on protocol configuration."""

    @property
    @abstractmethod
    def control_address(self) -> str | None:
        """Get the control address based on protocol configuration."""

    @property
    @abstractmethod
    def capture_address(self) -> str | None:
        """Get the capture address based on protocol configuration."""

    @property
    def additional_frontend_bind_address(self) -> str | None:
        """Additional frontend address for dual-bind proxies. None by default."""
        return None

    @property
    def additional_backend_bind_address(self) -> str | None:
        """Additional backend address for dual-bind proxies. None by default."""
        return None

    def resolve_frontend(self, remote_host: str | None = None) -> str:
        """Resolve the frontend address. Subclasses may use remote_host for dual-bind."""
        return self.frontend_address

    def resolve_backend(self, remote_host: str | None = None) -> str:
        """Resolve the backend address. Subclasses may use remote_host for dual-bind."""
        return self.backend_address


class BaseZMQCommunicationConfig(BaseModel, ABC):
    """Configuration for ZMQ communication."""

    comm_backend: ClassVar[CommunicationBackend]

    # Proxy config options to be overridden by subclasses
    event_bus_proxy_config: ClassVar[BaseZMQProxyConfig]
    dataset_manager_proxy_config: ClassVar[BaseZMQProxyConfig]
    raw_inference_proxy_config: ClassVar[BaseZMQProxyConfig]

    ipc_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="IPC socket directory path. None for non-IPC transports (e.g., TCP).",
        ),
    ]

    @property
    def _remote_host(self) -> str | None:
        """Remote host for address resolution. None means use local addresses."""
        return None

    @property
    @abstractmethod
    def records_push_pull_address(self) -> str:
        """Get the inference push/pull address based on protocol configuration."""

    @property
    @abstractmethod
    def credit_router_address(self) -> str:
        """Get the credit router address for bidirectional ROUTER-DEALER credit routing."""

    @property
    @abstractmethod
    def credit_return_router_address(self) -> str:
        """Get the credit return router address for dedicated ROUTER-DEALER credit return channel."""

    @property
    @abstractmethod
    def control_address(self) -> str:
        """Get the control channel address for ROUTER-DEALER control plane."""

    @property
    @abstractmethod
    def group_lifecycle_address(self) -> str:
        """Get the group-local lifecycle channel address for WorkerGroupManager coordination."""

    def get_address(self, address_type: CommAddress) -> str:
        """Resolve a concrete ZMQ address for a ``CommAddress`` selector.

        Dispatches through ``_ADDRESS_RESOLVERS`` so subclasses only implement the
        transport-specific address properties.

        Raises:
            ValueError: If ``address_type`` is not a supported ``CommAddress`` key.
        """
        resolver = _ADDRESS_RESOLVERS.get(address_type)
        if resolver is None:
            raise ValueError(f"Invalid address type: {address_type}")
        return resolver(self)


# Dispatch table for get_address. Keeping this as a module-level mapping from
# CommAddress to a resolver callable keeps BaseZMQCommunicationConfig.get_address
# flat (no match ladder) and makes the mapping easy to extend.
_ADDRESS_RESOLVERS: dict[CommAddress, Callable[[BaseZMQCommunicationConfig], str]] = {
    CommAddress.EVENT_BUS_PROXY_FRONTEND: lambda c: c.event_bus_proxy_config.resolve_frontend(
        c._remote_host
    ),
    CommAddress.EVENT_BUS_PROXY_BACKEND: lambda c: c.event_bus_proxy_config.resolve_backend(
        c._remote_host
    ),
    CommAddress.DATASET_MANAGER_PROXY_FRONTEND: lambda c: c.dataset_manager_proxy_config.resolve_frontend(
        c._remote_host
    ),
    CommAddress.DATASET_MANAGER_PROXY_BACKEND: lambda c: c.dataset_manager_proxy_config.resolve_backend(
        c._remote_host
    ),
    # Raw inference proxy is always local (within-pod IPC). Workers and record
    # processors are co-located in the same pod, so remote_host is ignored.
    CommAddress.RAW_INFERENCE_PROXY_FRONTEND: lambda c: c.raw_inference_proxy_config.resolve_frontend(
        None
    ),
    CommAddress.RAW_INFERENCE_PROXY_BACKEND: lambda c: c.raw_inference_proxy_config.resolve_backend(
        None
    ),
    CommAddress.CREDIT_ROUTER: lambda c: c.credit_router_address,
    CommAddress.CREDIT_RETURN_ROUTER: lambda c: c.credit_return_router_address,
    CommAddress.CREDIT_RETURN: lambda c: c.credit_return_push_pull_address,
    CommAddress.RECORDS: lambda c: c.records_push_pull_address,
    CommAddress.CONTROL: lambda c: c.control_address,
    CommAddress.GROUP_LIFECYCLE: lambda c: c.group_lifecycle_address,
}
