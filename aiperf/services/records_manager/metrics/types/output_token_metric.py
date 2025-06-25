#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MetricTimeType, MetricType
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.records_manager.metrics.base_metric import BaseMetric
from aiperf.services.records_manager.metrics.types.output_prompt_metric import (
    OutputPromptMetric,
)
from aiperf.services.records_manager.records import Record


class OutputTokenMetric(BaseMetric):
    """
    Post-processor for calculating the output tokens.
    """

    tag = "output_token"
    unit = MetricTimeType.NONE
    type = MetricType.METRIC_OF_METRICS
    larger_is_better = True
    header = "Output Token Count"

    def __init__(self):
        self.metric: list[int] = []
        self._tokenizer = Tokenizer()

    def update_value(
        self, record: Record | None = None, metrics: dict["BaseMetric"] | None = None
    ) -> None:
        """
        Calculate the output token count.

        """
        output_prompt_metric = metrics[OutputPromptMetric.tag].values()
        print(f"Output prompts for tokenization: {output_prompt_metric}")
        for prompt in output_prompt_metric:
            if prompt is not None:
                print(f"Tokenizing output prompt: {prompt}")
                # Tokenize the output prompt and count the tokens
                tokenized_output = self._tokenizer.encode(prompt)
                self.metric.append(len(tokenized_output))

    def values(self) -> float:
        """
        Returns the list of Time to First Token (TTFT) metrics.
        """
        return self.metric

    def _check_record(self, record: Record) -> None:
        # Only accumulating the output prompt, no specific checks needed
        pass
