#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
from collections.abc import Callable

from aiperf.common.comms.base_communication import BaseCommunication
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import CommunicationBackend
from aiperf.common.exceptions.comm_exceptions import (
    CommunicationCreateError,
    CommunicationTypeAlreadyRegisteredError,
    CommunicationTypeUnknownError,
)
from aiperf.common.models.comm_models import (
    ZMQCommunicationConfig,
    ZMQTCPTransportConfig,
)

logger = logging.getLogger(__name__)


class CommunicationFactory:
    """Factory for creating communication instances. Provides a registry of communication types and
    methods for registering new communication types and creating communication instances from existing
    registered types.
    """

    # Registry of communication types
    _comm_registry: dict[CommunicationBackend | str, type[BaseCommunication]] = {}

    @classmethod
    def register_comm_type(
        cls, comm_type: CommunicationBackend | str, comm_class: type[BaseCommunication]
    ) -> None:
        """Register a new communication type.

        Args:
            comm_type: String representation of the communication type
            comm_class: The class that implements the communication type
        """
        cls._comm_registry[comm_type] = comm_class
        logger.debug("Registered communication type: %s", comm_type)

    @classmethod
    def register(cls, comm_type: CommunicationBackend | str) -> Callable:
        """Register a new communication type.

        Args:
            comm_type: String representation of the communication type

        Returns:
            Decorator for the communication class

        Raises:
            CommunicationTypeAlreadyRegisteredError: If the communication type is already registered
        """

        def decorator(comm_cls: type[BaseCommunication]) -> type[BaseCommunication]:
            if comm_type in cls._comm_registry:
                raise CommunicationTypeAlreadyRegisteredError(
                    f"Communication type {comm_type} already registered"
                )
            cls._comm_registry[comm_type] = comm_cls
            return comm_cls

        return decorator

    @classmethod
    def create_communication(
        cls, service_config: ServiceConfig, **kwargs
    ) -> BaseCommunication:
        """Create a communication instance.

        Args:
            service_config: Service configuration containing the communication type
            **kwargs: Additional arguments for the communication class

        Returns:
            BaseCommunication instance

        Raises:
            CommunicationTypeUnknownError: If the communication type is not registered
            CommunicationCreateError: If there was an error creating the communication instance
        """
        if service_config.comm_backend not in cls._comm_registry:
            logger.error("Unknown communication type: %s", service_config.comm_backend)
            raise CommunicationTypeUnknownError(
                f"Unknown communication type: {service_config.comm_backend}"
            )

        try:
            comm_class = cls._comm_registry[service_config.comm_backend]
            config = kwargs.get("config") or ZMQCommunicationConfig(
                protocol_config=ZMQTCPTransportConfig()
            )
            kwargs["config"] = config

            return comm_class(**kwargs)
        except Exception as e:
            logger.error(
                "Exception creating communication for type %s: %s",
                service_config.comm_backend,
                e,
            )
            raise CommunicationCreateError from e
