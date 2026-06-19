# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: full-dataset list-metric collection must not mutate the ragged buffer.

``metric_result_from_array`` sorts its input array in place. For the
full-dataset replay path the input used to be a *view* into ``RaggedSeries``'
backing buffer (``backend.values``), so the sort globally reordered the values
while ``offsets``/``record_indices`` stayed in insertion order -- silently
corrupting every ICL-aware sweep curve that ``compute_sweep_curves``
reconstructs afterward (it runs after ``_compute_results`` inside
``summarize()``). The collection path must hand ``metric_result_from_array`` a
copy, leaving the shared buffer untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_records_message,
)

ICL_TAG = "inter_chunk_latency"


@pytest.mark.asyncio
async def test_summarize_does_not_sort_ragged_buffer_in_place(
    mock_user_config,
) -> None:
    acc = create_accumulator_with_metrics(mock_user_config, InterChunkLatencyMetric)

    # Per-record ICL chosen so a global sort would scramble record grouping:
    # record 0 = [30, 10, 20], record 1 = [5, 15]. Insertion order != sorted.
    await acc.process_record(
        create_metric_records_message(
            session_num=0, results=[{ICL_TAG: [30.0, 10.0, 20.0]}]
        ).to_data()
    )
    await acc.process_record(
        create_metric_records_message(
            session_num=1, results=[{ICL_TAG: [5.0, 15.0]}]
        ).to_data()
    )

    backend = acc.column_store.ragged(ICL_TAG)
    # Guard against a vacuous pass: the tag must actually reach
    # metric_result_from_array (which performs the in-place sort).
    assert ICL_TAG in acc.column_store.ragged_tags()

    expected_values = [30.0, 10.0, 20.0, 5.0, 15.0]
    # Per-request cumulative sum (reset at record boundaries): the foundation of
    # the ICL-aware throughput sweeps.
    expected_grouped_cumsum = [30.0, 40.0, 60.0, 5.0, 20.0]
    np.testing.assert_array_equal(backend.values, expected_values)

    # summarize() runs _compute_results() (which sorts) BEFORE
    # compute_sweep_curves() (which reads the buffer). The buffer must survive.
    await acc.summarize()

    np.testing.assert_array_equal(
        backend.values,
        expected_values,
        err_msg="full-dataset list collection sorted the shared ragged buffer in place",
    )
    np.testing.assert_array_equal(backend.grouped_cumsum(), expected_grouped_cumsum)
