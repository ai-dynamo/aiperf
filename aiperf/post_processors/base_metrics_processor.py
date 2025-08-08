# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums.metric_enums import MetricFlags, MetricType
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_registry import MetricRegistry


class BaseMetricsProcessor(AIPerfLoggerMixin, ABC):
    """Base class for all metrics processors. This class is responsible for filtering the metrics based on the user config."""

    def __init__(self, user_config: UserConfig, **kwargs):
        self.user_config = user_config
        super().__init__(user_config=user_config, **kwargs)

    def get_filters(self) -> tuple[MetricFlags, MetricFlags]:
        """Get the filters for the metrics based on the user config.
        Returns:
            tuple[MetricFlags, MetricFlags]: The required and disallowed flags.
        """
        # Start with no flags (unfiltered)
        required_flags, disallowed_flags = MetricFlags.NONE, MetricFlags.NONE
        # Disable metrics that are not applicable to the endpoint type
        if not self.user_config.endpoint.type.produces_tokens:
            disallowed_flags |= MetricFlags.PRODUCES_TOKENS_ONLY
        if not self.user_config.endpoint.type.supports_audio:
            disallowed_flags |= MetricFlags.SUPPORTS_AUDIO_ONLY
        if not self.user_config.endpoint.type.supports_images:
            disallowed_flags |= MetricFlags.SUPPORTS_IMAGE_ONLY
        if not self.user_config.endpoint.streaming:
            disallowed_flags |= MetricFlags.STREAMING_ONLY
        return required_flags, disallowed_flags

    def _setup_metrics(
        self,
        *metric_types: MetricType,
        error_metrics_only: bool = False,
        exclude_error_metrics: bool = False,
    ) -> list[BaseMetric]:
        """Get an ordered list of metrics that are applicable to the endpoint type and user config.
        The metrics are ordered based on their dependencies, ensuring proper computation order.

        Be sure to compute the metrics sequentially versus in parallel, as some metrics may depend on the results of previous metrics.
        """
        required_flags, disallowed_flags = self.get_filters()
        if error_metrics_only:
            required_flags |= MetricFlags.ERROR_ONLY
        elif exclude_error_metrics:
            disallowed_flags |= MetricFlags.ERROR_ONLY

        metrics: list[BaseMetric] = []
        supported_tags = MetricRegistry.tags_applicable_to(
            required_flags,
            disallowed_flags,
            *metric_types,
        )
        ordered_tags = MetricRegistry.create_dependency_order_for(
            supported_tags,
        )
        for metric_tag in ordered_tags:
            metrics.append(MetricRegistry.get_instance(metric_tag))
        return metrics
