# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from aiperf.common.config import UserConfig
from aiperf.data_exporter.record import Record


@dataclass
class ExporterConfig:
    records: list[Record]
    input_config: UserConfig
