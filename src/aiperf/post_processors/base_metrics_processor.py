# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

from aiperf.common.config import UserConfig
from aiperf.common.constants import GOOD_REQUEST_COUNT_TAG
from aiperf.common.enums import MetricFlags, MetricType
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_registry import MetricRegistry


class BaseMetricsProcessor(AIPerfLifecycleMixin, ABC):
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
        from aiperf.plugin import plugins

        endpoint_metadata = plugins.get_endpoint_metadata(
            self.user_config.endpoint.type
        )
        if not endpoint_metadata.produces_tokens:
            disallowed_flags |= MetricFlags.PRODUCES_TOKENS_ONLY
        if not endpoint_metadata.tokenizes_input:
            disallowed_flags |= MetricFlags.TOKENIZES_INPUT_ONLY
        if not endpoint_metadata.supports_audio:
            disallowed_flags |= MetricFlags.SUPPORTS_AUDIO_ONLY
        if not endpoint_metadata.supports_images:
            disallowed_flags |= MetricFlags.SUPPORTS_IMAGE_ONLY
        if not endpoint_metadata.supports_videos:
            disallowed_flags |= MetricFlags.SUPPORTS_VIDEO_ONLY
        if not endpoint_metadata.produces_videos:
            disallowed_flags |= MetricFlags.PRODUCES_VIDEO_ONLY
        if not self.user_config.endpoint.streaming:
            disallowed_flags |= MetricFlags.STREAMING_ONLY
        if self.user_config.endpoint.use_server_token_count:
            # Disable usage diff metrics if server token counts are used, because
            # these metrics are only applicable when client side tokenization is enabled.
            disallowed_flags |= MetricFlags.USAGE_DIFF_ONLY
        if not Environment.DEV.MODE and not Environment.DEV.SHOW_INTERNAL_METRICS:
            disallowed_flags |= MetricFlags.INTERNAL
        if not Environment.DEV.MODE and not Environment.DEV.SHOW_EXPERIMENTAL_METRICS:
            disallowed_flags |= MetricFlags.EXPERIMENTAL
        return required_flags, disallowed_flags

    def _configure_goodput(self, applicable_tags: set[str]) -> None:
        """
        If --goodput SLOs are provided, wire the SLOs into the GoodRequestCountMetric.
        """
        if not self.user_config.input.goodput:
            return
        if GOOD_REQUEST_COUNT_TAG not in applicable_tags:
            return

        slo_tags = set((self.user_config.input.goodput or {}).keys())
        missing_tags = slo_tags - set(applicable_tags)
        if missing_tags:
            raise RuntimeError(
                "Invalid --goodput: metric(s) "
                + ", ".join(sorted(missing_tags))
                + " are not applicable to the current endpoint/configuration."
            )

        try:
            MetricRegistry.get_class(GOOD_REQUEST_COUNT_TAG).set_slos(
                self.user_config.input.goodput
            )
        except ValueError as e:
            raise RuntimeError(f"Invalid --goodput: {e}") from e

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

        if not self.user_config.input.goodput:
            disallowed_flags |= MetricFlags.GOODPUT

        metrics: list[BaseMetric] = []
        applicable_tags = MetricRegistry.tags_applicable_to(
            required_flags,
            disallowed_flags,
            *metric_types,
        )
        self._configure_goodput(applicable_tags)

        ordered_tags = MetricRegistry.create_dependency_order_for(
            applicable_tags,
        )
        for metric_tag in ordered_tags:
            metric = MetricRegistry.get_instance(metric_tag)
            metrics.append(metric)

        return metrics
