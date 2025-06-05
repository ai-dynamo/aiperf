# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import PostProcessorType
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary


class PostProcessFactory:
    """
    Factory class for creating post-processors based on the provided type.
    """

    POST_PROCESSORS_MAP = {
        PostProcessorType.METRIC_SUMMARY: MetricSummary,  # Assuming METRIC_SUMMARY is defined elsewhere
        # Add other post-processors here as needed
        # TODO: Add dynamic loading of post-processors
    }

    @classmethod
    def create(cls, post_processor_type: PostProcessorType):
        """
        Create a post-processor instance based on the specified type.

        :param post_processor_type: The type of post-processor to create.
        :return: An instance of the specified post-processor.
        """

        return cls.POST_PROCESSORS_MAP[post_processor_type]()
