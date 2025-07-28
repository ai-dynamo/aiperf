# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.models import ParsedResponseRecord, ResponseData

################################################################################
# Data Exporter Protocol
################################################################################


@runtime_checkable
class DataExporterProtocol(Protocol):
    """
    Protocol for data exporters.
    Any class implementing this protocol must provide an `export` method
    that takes a list of Record objects and handles exporting them appropriately.
    """

    async def export(self) -> None:
        """Export the data."""
        ...


################################################################################
# Post Processor Protocol
################################################################################
class PostProcessorProtocol(Protocol):
    """
    PostProcessorProtocol is a protocol that defines the API for post-processors.
    It requires an `process` method that takes a list of records and returns a result.
    """

    def process_record(self, record: "ParsedResponseRecord") -> None:
        """Process a single record."""
        ...

    def post_process(self) -> None:
        """
        Execute the post-processing logic on the records.
        """
        pass

    def get_results(self) -> Any:
        """Get the results of the post-processing."""
        ...


################################################################################
# Response Extractor Protocol
################################################################################


class ResponseExtractor(Protocol):
    """Base class for all response extractors."""

    async def extract_response_data(
        self, record: "ParsedResponseRecord"
    ) -> list["ResponseData"]:
        """Extract the text from a server response message."""
        ...
