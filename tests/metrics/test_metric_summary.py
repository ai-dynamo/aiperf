# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.common.models import ParsedResponseRecord, RequestRecord, ResponseData
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary


def build_record(
    start_ns, first_resp_ns, last_resp_ns, final_ns, input_tokens=5, output_tokens=5
):
    return ParsedResponseRecord(
        worker_id="worker-1",
        request=RequestRecord(
            conversation_id="cid",
            turn_index=0,
            model_name="model",
            start_perf_ns=start_ns,
            timestamp_ns=start_ns,
        ),
        responses=[
            ResponseData(
                perf_ns=first_resp_ns,
                token_count=1,
                raw_text=["hi"],
                parsed_text=["hi"],
            ),
            ResponseData(
                perf_ns=last_resp_ns,
                token_count=output_tokens - 1,
                raw_text=["bye"],
                parsed_text=["bye"],
            ),
        ],
        input_token_count=input_tokens,
        output_token_count=output_tokens,
    )


def test_metric_summary_process_with_all_metrics():
    records = [
        build_record(0, 100, 150, 170),
        build_record(10, 120, 160, 180),
        build_record(20, 140, 180, 200),
    ]

    summary = MetricSummary()
    for record in records:
        summary.process_record(record)
    summary.post_process()

    for metric in summary._metrics:
        try:
            value = metric.values()
            assert value is not None
        except Exception:
            pytest.fail(f"Metric {metric.tag} failed to compute")
