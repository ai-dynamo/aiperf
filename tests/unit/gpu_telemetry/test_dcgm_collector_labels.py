# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Label handling in the DCGM collector's Prometheus parser.

dcgm-exporter#655 lowercased ``Hostname`` to ``hostname``. Both forms are in
the wild, so the parser has to read either one.
"""

from __future__ import annotations

import pytest

from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector

METRIC = (
    "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature\n"
    "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
    'DCGM_FI_DEV_GPU_TEMP{{gpu="0",UUID="GPU-0",device="nvidia0",'
    'modelName="GPU",{label}="node-1"}} 42\n'
)


@pytest.fixture
def collector() -> DCGMTelemetryCollector:
    return DCGMTelemetryCollector("http://localhost:9400/metrics")


@pytest.mark.parametrize("label", ["hostname", "Hostname"])
def test_hostname_read_from_either_label_case(collector, label):
    """Current lowercase and legacy capitalised labels both resolve."""
    records = collector._parse_metrics_to_records(METRIC.format(label=label))

    assert records
    assert records[0].hostname == "node-1"


def test_lowercase_wins_when_both_labels_are_present(collector):
    """An exporter emitting both forms is read as the current one."""
    metric = (
        "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature\n"
        "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-0",device="nvidia0",'
        'modelName="GPU",hostname="new",Hostname="legacy"} 42\n'
    )
    records = collector._parse_metrics_to_records(metric)

    assert records
    assert records[0].hostname == "new"


def test_hostname_absent_stays_none(collector):
    """No hostname label at all is still None rather than an error."""
    metric = (
        "# HELP DCGM_FI_DEV_GPU_TEMP GPU temperature\n"
        "# TYPE DCGM_FI_DEV_GPU_TEMP gauge\n"
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-0",device="nvidia0",'
        'modelName="GPU"} 42\n'
    )
    records = collector._parse_metrics_to_records(metric)

    assert records
    assert records[0].hostname is None
