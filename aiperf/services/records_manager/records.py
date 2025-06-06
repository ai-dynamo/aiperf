#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic import BaseModel

Timestamp: TypeAlias = int


@dataclass
class Message:
    """
    Represents a request/response with a timestamp and associated payload.

    Attributes:
        timestamp: The time at which the response was recorded.
        payload: The data or content of the response.
    """

    timestamp: Timestamp
    payload: Any


@dataclass
class Record:
    """
    Data class representing a record of a request and its associated responses.

    Attributes:
        request: The timestamp when the request was made.
        responses: A list of Response objects associated with the request.
    """

    request: Message
    responses: list[Message]


class Records(BaseModel):
    """
    A collection of records, each containing a request timestamp and a list of responses.
    """

    records: list[Record] = []

    def add_record(self, request: Message, responses: list[Message]) -> None:
        """
        Add a new record with the given request timestamp and responses.
        """
        self.records.append(Record(request=request, responses=responses))

    def get_records(self) -> list[Record]:
        """
        Retrieve all records.
        """
        return self.records
