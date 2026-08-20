# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation tags on DatasetConfiguredNotification.

Worker pods and the API dataset router read ``benchmark_generation`` and
``dataset_generation`` off this notification. Regression: the fields were
absent from the message, so every worker pod raised AttributeError inside the
subscription handler, never downloaded the dataset, and the whole benchmark
died at 'Configure Profiling'.
"""

from pathlib import Path

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models.dataset_models import (
    DatasetMetadata,
    MemoryMapClientMetadata,
)
from aiperf.dataset.dataset_manager import _dataset_generation_of
from aiperf.plugin.enums import DatasetSamplingStrategy


def _client_metadata() -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        format=MemoryMapFormat.PAYLOAD_BYTES,
        data_file_path=Path("/aiperf/datasets/aiperf_mmap_bench-7f2a/dataset.dat"),
        index_file_path=Path("/aiperf/datasets/aiperf_mmap_bench-7f2a/index.bin"),
        conversation_count=4,
        total_size_bytes=1024,
    )


def test_notification_carries_generation_tags() -> None:
    message = DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(sampling_strategy=DatasetSamplingStrategy.RANDOM),
        client_metadata=_client_metadata(),
        benchmark_generation="bench-7f2a",
        dataset_generation="aiperf_mmap_bench-7f2a",
    )
    assert message.benchmark_generation == "bench-7f2a"
    assert message.dataset_generation == "aiperf_mmap_bench-7f2a"


def test_generation_tags_default_to_none() -> None:
    """Non-Kubernetes publishers may omit them; consumers must still read None."""
    message = DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(sampling_strategy=DatasetSamplingStrategy.RANDOM),
        client_metadata=_client_metadata(),
    )
    assert message.benchmark_generation is None
    assert message.dataset_generation is None


def test_dataset_generation_derived_from_mmap_run_directory() -> None:
    assert _dataset_generation_of(_client_metadata()) == "aiperf_mmap_bench-7f2a"
