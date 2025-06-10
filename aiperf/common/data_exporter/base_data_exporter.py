# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from aiperf.common.data_exporter.record import Record


class BaseDataExporter(Protocol):
    def export(self, records: list[Record]) -> None: ...
