# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration test: the E2E Normalized Interactivity metric is emitted
end-to-end from a real ``aiperf profile`` run against the mock server."""

import math

import pytest

from aiperf.common.models import JsonExportData
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.utils import AIPerfCLI


def _metric_avg(json_export: JsonExportData, tag: str) -> float:
    """Read a metric's ``avg`` from the JSON export.

    ``JsonExportData`` is ``extra="allow"``, so metrics not declared on the model
    (like this one) come through as raw dicts rather than typed results.
    """
    metric = getattr(json_export, tag, None)
    assert metric is not None, f"{tag} missing from the JSON export"
    return metric["avg"] if isinstance(metric, dict) else metric.avg


@pytest.mark.component_integration
class TestE2ENormalizedInteractivityMetric:
    """Smoke test that the p90/p75 interactivity scalars appear and are finite."""

    def test_metric_emitted_and_finite(self, cli: AIPerfCLI) -> None:
        """A streaming chat run (which has TTFT/ISL/OSL/latency) emits both
        interactivity percentiles as finite, positive values, and the slow-tail
        ordering holds (p90 <= p75).

        Uses the synchronous ``run_sync``: the component_integration ``cli``
        fixture is an in-process runner and ``_single_run`` calls ``asyncio.run``
        internally, so awaiting it inside the pytest event loop would raise
        "asyncio.run() cannot be called from a running event loop"."""
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} --tokenizer {defaults.tokenizer} \
                --endpoint-type chat --streaming \
                --synthetic-input-tokens-mean 200 --output-tokens-mean 100 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.request_count == defaults.request_count

        p90 = _metric_avg(result.json, "e2e_normalized_interactivity_p90")
        p75 = _metric_avg(result.json, "e2e_normalized_interactivity_p75")

        # Finite and positive (NaN/Inf discipline; unit is tokens/sec/user).
        assert math.isfinite(p90) and p90 > 0, f"p90 not finite/positive: {p90}"
        assert math.isfinite(p75) and p75 > 0, f"p75 not finite/positive: {p75}"

        # Slow-tail convention: p90 is the pessimistic tail, so 1/p90(ratio) is
        # never larger than 1/p75(ratio). Higher percentile -> lower interactivity.
        assert p90 <= p75, f"p90 (slow tail) should be <= p75, got p90={p90}, p75={p75}"
