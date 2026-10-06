# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace formats that nest their timing must be classified as having timing.

``fixed_schedule`` is rejected unless the resolver believes the dataset carries
timing. That belief comes from a generic probe of the first record, which only
sees top-level ``timestamp``/``delay`` fields. Formats that nest their timing
have to be named explicitly, or every guide documenting ``--fixed-schedule``
against them describes a combination that cannot run.

Only the ``weka_trace`` row is new here; the other four already pass on main.
They are kept deliberately as a drift guard -- dropping a format from the set
is silent, and the symptom is a documented command that stops working -- but
they are not coverage of this change and should not be counted as such.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.dataset.resolver import _implicit_timing_types
from aiperf.plugin.enums import CustomDatasetType


@pytest.mark.parametrize(
    "dataset_type,where_timing_lives",
    [
        param(CustomDatasetType.WEKA_TRACE, "requests[].t", id="weka_trace"),
        param(CustomDatasetType.TRACELAB, "timing_events[]", id="tracelab"),
        param(
            CustomDatasetType.SAGEMAKER_DATA_CAPTURE,
            "eventMetadata.inferenceTime",
            id="sagemaker",
        ),
        param(CustomDatasetType.BASETEN_TRACE, "parquet columns", id="baseten"),
        param(CustomDatasetType.BURST_GPT_TRACE, "CSV Timestamp column", id="burst_gpt"),
    ],
)  # fmt: skip
def test_nested_timing_formats_are_classified_as_timed(
    dataset_type: CustomDatasetType, where_timing_lives: str
) -> None:
    assert dataset_type in _implicit_timing_types(), (
        f"{dataset_type} keeps its timing in {where_timing_lives}, which the "
        f"generic first-record probe cannot see, so fixed_schedule would be "
        f"rejected for a dataset that does carry timing."
    )
