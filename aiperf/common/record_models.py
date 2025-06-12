#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """
    Represents a request/response with a timestamp and associated payload.

    Attributes:
        timestamp: The time at which the transaction was recorded.
        payload: The data or content of the transaction.
    """

    timestamp: int = Field(description="The timestamp of the transaction")
    payload: Any = Field(description="The payload of the transaction")


class Record(BaseModel):
    """
    Represents a record containing a request transaction and its associated response transactions.
    Attributes:
        request: The input transaction for the record.
        responses A list of response transactions corresponding to the request.
    """

    request: Transaction = Field(description="The request transaction for the record")
    responses: list[Transaction] = Field(
        description="A list of response transactions for the record",
    )
