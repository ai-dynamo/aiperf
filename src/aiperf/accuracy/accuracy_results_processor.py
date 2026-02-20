# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.config import UserConfig
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.common.messages.inference_messages import MetricRecordsData


class AccuracyResultsProcessor(AIPerfLifecycleMixin):
    """Results processor for accuracy benchmarking.

    Aggregates accuracy grading results and computes summary metrics.
    Self-disables when accuracy mode is not enabled.
    """

    def __init__(self, user_config: UserConfig, **kwargs) -> None:
        if not user_config.accuracy.enabled:
            raise PostProcessorDisabled(
                "Accuracy results processor is disabled: accuracy mode is not enabled"
            )

        super().__init__(user_config=user_config, **kwargs)
        self.user_config = user_config

    async def process_result(self, record_data: MetricRecordsData) -> None:
        raise NotImplementedError

    async def summarize(self) -> list[MetricResult]:
        raise NotImplementedError
