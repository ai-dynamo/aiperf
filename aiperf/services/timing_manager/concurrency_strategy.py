# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.services.timing_manager.credit_issuing_strategy import CreditIssuingStrategy


class ConcurrencyStrategy(CreditIssuingStrategy):
    """
    Class for concurrency credit issuing strategy.
    """

    def __init__(self, config, credit_drop_function):
        raise NotImplementedError()

    async def initialize(self) -> None:
        raise NotImplementedError()

    async def start(self) -> None:
        raise NotImplementedError()
