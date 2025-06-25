#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.records import Record


class OutputPromptMetric(BaseMetric):
    """
    Post-processor for calculating the output prompts from records.
    """

    tag = "output_prompt"
    unit = MetricTimeType.NONE
    type = MetricType.METRIC_OF_RECORDS
    larger_is_better = True
    header = "Output Prompt"

    def __init__(self):
        self.metric: list[str] = []

    def update_value(
        self, record: Record | None = None, metrics: dict["BaseMetric"] | None = None
    ) -> None:
        """
        Adds a record and calculates the accumulated output prompt.

        """
        # TODO: define the payload output prompt in a constants file once agreed upon
        self._check_record(record)
        output_prompts = [
            response.payload["output_prompt"]
            for response in record.responses
            if response is not None
        ]
        joined_prompt = "".join(output_prompts)
        self.metric.append(joined_prompt)

    def values(self) -> float:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        # Only accumulating the output prompt, no specific checks needed
        pass
