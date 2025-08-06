# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import ParsedResponseRecord, RequestRecord, ResponseData
from aiperf.metrics.types.output_sequence_length_metric import (
    OutputSequenceLengthMetric,
)


@pytest.fixture
def sample_record():
    return ParsedResponseRecord(
        request=RequestRecord(
            conversation_id="c1",
            turn_index=0,
            model_name="model",
            start_perf_ns=0,
            timestamp_ns=0,
        ),
        responses=[
            ResponseData(
                perf_ns=100,
                raw_text=["hello"],
                parsed_text=["hello"],
                token_count=5,
            )
        ],
        output_token_count=5,
    )


@pytest.skip(reason="TODO: Metric refactor work in progress", allow_module_level=True)
def test_output_token_count_metric(sample_record):
    metric = OutputSequenceLengthMetric()
    metric.update_value(record=sample_record)
    assert metric.values() == [5]
