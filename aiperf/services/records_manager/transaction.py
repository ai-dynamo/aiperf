#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, TypeAlias

Timestamp: TypeAlias = int


@dataclass
class Transaction:
    """
    Represents a request/response with a timestamp and associated payload.

    Attributes:
        timestamp: The time at which the transaction was recorded.
        payload: The data or content of the transaction.
    """

    timestamp: Timestamp
    payload: Any
