# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import (
    BaselineKind,
    ServiceCapability,
    make_result_producer_capability,
    parse_result_producer_capability,
)


def test_baseline_kind_values_lowercase():
    assert BaselineKind.START == "start"
    assert BaselineKind.END == "end"


def test_baseline_kind_case_insensitive():
    assert BaselineKind("START") == BaselineKind.START
    assert BaselineKind("End") == BaselineKind.END


def test_service_capability_baseline_collector_value():
    assert ServiceCapability.BASELINE_COLLECTOR == "baseline_collector"


def test_service_capability_result_producer_value():
    assert ServiceCapability.RESULT_PRODUCER == "result_producer"


def test_make_result_producer_capability_includes_domain():
    assert make_result_producer_capability("profile") == "result_producer:profile"


def test_parse_result_producer_capability_returns_domain():
    assert parse_result_producer_capability("result_producer:telemetry") == "telemetry"


def test_parse_result_producer_capability_ignores_other_capabilities():
    assert parse_result_producer_capability("baseline_collector") is None
    assert parse_result_producer_capability("result_producer") is None
    assert parse_result_producer_capability("result_producer:") is None
