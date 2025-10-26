# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import ConfigDict

from aiperf.common.models.auto_routed_model import AutoRoutedModel


class AIPerfBaseModel(AutoRoutedModel):
    """Base model for all AIPerf Pydantic models.

    Inherits high-performance auto-routing capabilities from AutoRoutedModel.
    Models can optionally set discriminator_field to enable automatic routing.

    This class is configured to allow arbitrary types to be used as fields
    to allow for more flexible model definitions by end users without breaking
    existing code.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
