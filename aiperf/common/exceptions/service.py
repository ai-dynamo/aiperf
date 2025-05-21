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

from aiperf.common.exceptions.base import AIPerfError


class ServiceError(AIPerfError):
    """Base class for all exceptions raised by services."""

    # TODO: have the base exception class accept the service information
    #       and add it to the pre-defined messages for each exception
    message: str = "Service error"


class ServiceMetaclassError(AIPerfError):
    """Exception raised for service metaclass errors."""

    message: str = (
        "Service metaclass error. Please check the service definition decorators."
    )


class ServiceInitializationError(ServiceError):
    """Exception raised for service initialization errors."""

    message: str = "Failed to initialize service"


class ServiceStartError(ServiceError):
    """Exception raised for service start errors."""

    message: str = "Failed to start service"


class ServiceStopError(ServiceError):
    """Exception raised for service stop errors."""

    message: str = "Failed to stop service"


class ServiceCleanupError(ServiceError):
    """Exception raised for service cleanup errors."""

    message: str = "Failed to cleanup service"


class ServiceMessageProcessingError(ServiceError):
    """Exception raised for service message processing errors."""

    message: str = "Failed to process message"


class ServiceRegistrationError(ServiceError):
    """Exception raised for service registration errors."""

    message: str = "Failed to register service"


class ServiceStatusError(ServiceError):
    """Exception raised for service status errors."""

    message: str = "Failed to get service status"


class ServiceRunError(ServiceError):
    """Exception raised for service run errors."""

    message: str = "Failed to run service"


class ServiceConfigureError(ServiceError):
    """Exception raised for service configure errors."""

    message: str = "Failed to configure service"


class ServiceHeartbeatError(ServiceError):
    """Exception raised for service heartbeat errors."""

    message: str = "Failed to send heartbeat"
