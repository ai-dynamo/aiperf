# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from aiperf.data_exporter.record import Record

################################################################################
# Data Exporter Protocol
################################################################################


class DataExporterProtocol(Protocol):
    def export(self, records: list[Record]) -> None: ...
