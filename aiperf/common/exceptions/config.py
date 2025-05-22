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


class ConfigError(AIPerfError):
    """Base class for all exceptions raised by configuration errors."""

    message: str = "Configuration error"


class ConfigLoadError(ConfigError):
    """Exception raised for configuration load errors."""

    message: str = "Failed to load configuration"


class ConfigParseError(ConfigError):
    """Exception raised for configuration parse errors."""

    message: str = "Failed to parse configuration"


class ConfigValidationError(ConfigError):
    """Exception raised for configuration validation errors."""

    message: str = "Failed to validate configuration"
