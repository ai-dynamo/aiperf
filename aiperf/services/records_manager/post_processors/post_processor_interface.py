# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Protocol


class PostProcessorInterface(Protocol):
    """
    PostProcessorInterface is a protocol that defines the interface for post-processors.
    It requires an `process` method that takes a list of records and returns a result.
    """

    def process(self, records: dict) -> dict:
        """
        Execute the post-processing logic on the given payload.

        :param payload: The input data to be processed.
        :return: The processed data as a dictionary.
        """
        pass
