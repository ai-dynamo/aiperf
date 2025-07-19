# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from aiperf.common.config import UserConfig
from aiperf.common.messages import ProfileResultsMessage


@dataclass
class ExporterConfig:
    results: ProfileResultsMessage
    input_config: UserConfig
