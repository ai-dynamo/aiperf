# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.inference_wire import InferenceResultsWireMessage
from aiperf.common.messages import InferenceResultsMessage
from aiperf.common.metric_records_wire import (
    MetricRecordsBatchWireMessage,
    MetricRecordsWireMessage,
)
from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestMsgpackPipeline:
    def test_profile_uses_msgspec_raw_inference_pipeline(self, cli: AIPerfCLI) -> None:
        """A full profile run should emit msgspec raw-inference messages and valid outputs."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --osl 8 \
                --request-count 4 \
                --concurrency 2 \
                --workers-max 1 \
                --ui {defaults.ui} \
                --verbose
            """
        )

        assert result.json is not None
        assert result.inputs is not None
        assert result.jsonl is not None
        assert result.csv
        assert result.request_count == 4
        assert len(result.jsonl) == 4

        runner_result = result.runner_result
        assert isinstance(runner_result, AIPerfRunnerResultWithSharedBus)

        wire_payloads = [
            payload
            for payload in runner_result.sent_payloads
            if isinstance(payload.payload, InferenceResultsWireMessage)
        ]
        metric_payloads = [
            payload
            for payload in runner_result.sent_payloads
            if isinstance(
                payload.payload,
                (MetricRecordsWireMessage, MetricRecordsBatchWireMessage),
            )
        ]

        assert len(wire_payloads) == 4
        # Record processor batches metric records before pushing to records manager,
        # so the number of messages varies with flush timing. What matters is that
        # every inference result produced exactly one metric record delivered via
        # the msgspec wire path.
        total_metric_records = sum(
            len(p.payload.records)
            if isinstance(p.payload, MetricRecordsBatchWireMessage)
            else 1
            for p in metric_payloads
        )
        assert total_metric_records == 4
        assert not any(
            isinstance(payload.payload, InferenceResultsMessage)
            for payload in runner_result.sent_payloads
        )
