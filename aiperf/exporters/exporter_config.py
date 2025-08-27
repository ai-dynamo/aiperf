# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.models import ProfileResults


@dataclass
class ExporterConfig:
    results: ProfileResults
    user_config: UserConfig
    service_config: ServiceConfig


@dataclass
class FileExportInfo:
    export_type: str
    file_path: Path
