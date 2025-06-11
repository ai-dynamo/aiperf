#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field

from aiperf.common.record_models import Record, Transaction


class Records(BaseModel):
    """
    A collection of records, each containing a request and a list of responses.
    """

    records: list[Record] = Field(
        default_factory=list,
        description="A list of records, each containing a request and its responses.",
    )

    def add_record(self, request: Transaction, responses: list[Transaction]) -> None:
        """
        Add a new record with the given request and responses.
        """
        self.records.append(Record(request=request, responses=responses))
