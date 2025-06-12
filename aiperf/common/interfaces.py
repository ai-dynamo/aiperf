# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from aiperf.data_exporter.record import Record

################################################################################
# Data Exporter Protocol
################################################################################


class DataExporterProtocol(Protocol):
    """
    Protocol for data exporters.
    Any class implementing this protocol must provide an `export` method
    that takes a list of Record objects and handles exporting them appropriately.
    """

    def export(self, records: list[Record]) -> None: ...


################################################################################
# Post Processor Protocol
################################################################################
class PostProcessorProtocol(Protocol):
    """
    PostProcessorProtocol is a protocol that defines the API for post-processors.
    It requires an `process` method that takes a list of records and returns a result.
    """

    def process(self, records: dict) -> dict:
        """
        Execute the post-processing logic on the given payload.

        :param payload: The input data to be processed.
        :return: The processed data as a dictionary.
        """
        pass
