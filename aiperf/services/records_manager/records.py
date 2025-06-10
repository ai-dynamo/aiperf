#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import TypeAlias

from pydantic import BaseModel

from aiperf.services.records_manager.transaction import Transaction

Transactions: TypeAlias = list[Transaction]
Records: TypeAlias = list["Record"]


@dataclass
class Record:
    """
    Represents a record containing a request transaction and its associated response transactions.
    Attributes:
        request: The input transaction for the record.
        responses A list of response transactions corresponding to the request.
    """

    request: Transaction
    responses: Transactions


class Records(BaseModel):
    """
    A collection of records, each containing a request and a list of responses.
    """

    records: Records = []

    def add_record(self, request: Transaction, responses: Transactions) -> None:
        """
        Add a new record with the given request and responses.
        """
        self.records.append(Record(request=request, responses=responses))
