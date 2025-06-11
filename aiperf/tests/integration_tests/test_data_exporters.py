# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from aiperf.common.config.endpoint_config import EndPointConfig
from aiperf.data_exporter.exporter_manager import ExporterManager
from aiperf.data_exporter.record import Record

@pytest.fixture
def endpoint_config():
    return EndPointConfig(type="llm", streaming=True)

@pytest.fixture
def records():
    return [
        Record(name="Request Latency", unit="ms", avg=120.5, min=110.0, max=130.0, p99=128.0, p90=125.0, p75=122.0),
        Record(name="Request Throughput", unit="per sec", avg=95.0, min=None, max=None, p99=None, p90=None, p75=None),
        Record(name="Time to First Token", unit="ms", avg=150.3, min=140.1, max=160.4, p99=158.5, p90=156.7, p75=152.2),
        Record(name="Inter Token Latency", unit="ms", avg=3.7, min=2.9, max=5.1, p99=4.9, p90=4.5, p75=4.0, streaming_only=True),
        Record(name="Output Token Throughput", unit="tokens/sec", avg=200.0, min=180.0, max=220.0, p99=215.0, p90=210.0, p75=205.0),
    ]

class TestIntegrationDataExporters:
    def test_console_export(self, endpoint_config, records):
        manager = ExporterManager(endpoint_config)
        manager.export(records)
