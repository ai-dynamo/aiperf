#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from aiperf.services.records_manager.post_processors.metric_summary import MetricSummary
from aiperf.services.records_manager.records import Records, Transaction


def test_metric_summary_process_and_get_metrics():
    ms = MetricSummary()

    # Prepare records
    records = Records()
    req = Transaction(timestamp=1, payload="request1")
    resp1 = Transaction(timestamp=10, payload="payload1")
    resp2 = Transaction(timestamp=20, payload="payload2")
    records.add_record(request=req, responses=[resp1, resp2])

    ms.process(records.records)

    for tag, metric in ms.get_metrics().items():
        assert tag == "ttft"
        assert metric == [9]
