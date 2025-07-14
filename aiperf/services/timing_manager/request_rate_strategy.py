# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditManagerProtocol,
)


class RequestRateStrategy(CreditIssuingStrategy, AsyncTaskManagerMixin):
    """
    Strategy for issuing credits based on a specified request rate.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)

    async def _execute_phases(self) -> None:
        """Execute all phases sequentially."""
        raise NotImplementedError("Not implemented")
