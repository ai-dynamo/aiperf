# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from aiperf.common.config import UserConfig
from aiperf.common.models import ProfileResults


@dataclass
class ExporterConfig:
    results: ProfileResults
    user_config: UserConfig
