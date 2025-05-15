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
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.service.base import BaseService


def bootstrap_and_run_service(
    service_class: type[BaseService], service_config: ServiceConfig | None = None
):
    """Bootstrap the service and run it.

    This function will load the service configuration, create an instance of the service,
    and run it.

    Args:
        service_class: The service class of the service to run
        service_config: The service configuration to use, if not provided, the service configuration
                will be loaded from the config file

    """
    import uvloop

    uvloop.install()

    # Load the service configuration
    if service_config is None:
        from aiperf.common.config.loader import load_service_config

        service_config = load_service_config()

    # Create the service instance and run it
    service = service_class(service_config=service_config)
    uvloop.run(service.run())
