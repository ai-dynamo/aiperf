# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from aiperf.common.enums import CustomDatasetType


class TraceCustomData(BaseModel):
    """Defines the schema of each JSONL line in a trace file.

    Example:
    ```python
    # SUCCESS
    json_string = '{"timestamp": 1000, "input_length": 10, "output_length": 4, "hash_ids": [123, 456]}'
    custom_data = TraceCustomData.model_validate_json(json_string)

    # ERROR: timestamp and session_id (or delay)cannot be set together
    json_string = '{"timestamp": 1000, "session_id": "12345", "delay": 1000, "input_length": 300, "output_length": 40, "hash_ids": [123, 456]}'
    custom_data = TraceCustomData.model_validate_json(json_string)  # ERROR
    ```
    """

    type: Literal[CustomDatasetType.TRACE] = CustomDatasetType.TRACE

    input_length: int = Field(..., description="The input length of a request in trace")
    output_length: int = Field(
        ..., description="The output length of a request in trace"
    )
    hash_ids: list[int] = Field(..., description="The hash ids of a request in trace")
    timestamp: int | None = Field(
        None, description="The timestamp of a request in trace"
    )
    session_id: str | None = Field(
        None, description="The session id of a request in trace"
    )
    delay: int | None = Field(None, description="The delay of a request in trace")

    @model_validator(mode="after")
    def validate_mutually_exclusive_fields(self) -> "TraceCustomData":
        """Ensure timestamp cannot be set with session_id or delay"""
        if self.timestamp is not None:
            if self.session_id is not None:
                raise ValueError("timestamp and session_id cannot both be set")
            if self.delay is not None:
                raise ValueError("timestamp and delay cannot both be set")
        return self


CustomData = Annotated[TraceCustomData, Field(discriminator="type")]
"""A union type of all custom data types."""
