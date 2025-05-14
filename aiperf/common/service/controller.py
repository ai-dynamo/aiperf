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
from abc import ABC

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ClientType, PubClientType, ServiceState, SubClientType
from aiperf.common.service.base import BaseService


class ControllerServiceBase(BaseService, ABC):
    """Base class for all controller services.

    This class provides a common interface for all controller services in the AIPerf framework.
    It inherits from the ServiceBase class and implements the required methods for controller
    services.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        # The controller service subscribes to controller messages and publishes to components
        return [PubClientType.CONTROLLER, SubClientType.COMPONENT]

    # TODO: Complete the implementation of the controller service methods
    async def run(self) -> None:
        """Start the service and initialize its components."""
        await self._base_init()

        await self._initialize()

        await self.set_state(ServiceState.READY)

        # Start the service
        await self._start()

        # Wait forever for the stop event to be set
        await self.stop_event.wait()
