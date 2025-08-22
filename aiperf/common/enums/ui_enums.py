# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfUIType(CaseInsensitiveStrEnum):
    """The type of UI to use."""

    DASHBOARD = "dashboard"
    SIMPLE = "simple"
    NONE = "none"
