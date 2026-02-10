# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for GPU telemetry collection with DCGM faker."""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as Defaults,
)


@pytest.mark.component_integration
class TestGPUTelemetryBasic:
    """Basic GPU telemetry collection tests."""

    def test_dcgm_endpoints(self, cli, mock_dcgm_endpoints):
        """Test profile with multiple DCGM endpoints collects metrics from each.

        Verifies that when DCGM endpoints are specified, AIPerf:
        1. Collects telemetry from default endpoints (9400, 9401) plus any additional
        2. Each endpoint reports the expected number of GPUs
        3. All GPU metrics contain valid statistical data (avg, min, max, unit)

        Note: AIPerf always attempts default endpoints (localhost:9400, localhost:9401)
        in addition to any explicitly specified endpoints.
        """
        # Pass an additional endpoint beyond defaults to verify multi-endpoint collection
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {Defaults.model} \
                --url http://localhost:8000 \
                --gpu-telemetry http://localhost:9402/metrics \
                --benchmark-duration 2 \
                --concurrency 2 \
                --workers-max {Defaults.workers_max} \
                --ui {Defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.has_gpu_telemetry

        # Verify we collected telemetry from endpoints:
        # - Default endpoints: localhost:9400, localhost:9401
        # - Explicitly passed: localhost:9402
        endpoints = result.json.telemetry_data.endpoints
        assert endpoints is not None
        assert len(endpoints) >= 2, (
            f"Expected at least 2 endpoints (defaults + explicit), got {len(endpoints)}"
        )

        # Verify each endpoint has correct GPU data structure
        for dcgm_url, endpoint_data in endpoints.items():
            assert endpoint_data.gpus is not None
            assert len(endpoint_data.gpus) == 2, (
                f"Endpoint {dcgm_url}: expected 2 GPUs, got {len(endpoint_data.gpus)}"
            )

            # Verify each GPU has valid metrics with all required fields
            # Counter metrics only have avg (delta), not min/max
            counter_metrics = {"energy_consumption", "xid_errors", "power_violation"}
            for gpu_id, gpu_data in endpoint_data.gpus.items():
                assert gpu_data.metrics, f"GPU {gpu_id}: no metrics collected"
                for metric_name, metric_value in gpu_data.metrics.items():
                    assert metric_value.avg is not None, (
                        f"GPU {gpu_id} metric {metric_name}: missing avg"
                    )
                    assert metric_value.unit is not None, (
                        f"GPU {gpu_id} metric {metric_name}: missing unit"
                    )
                    # Gauge metrics should have min/max; counter metrics only have avg
                    if metric_name not in counter_metrics:
                        assert metric_value.min is not None, (
                            f"GPU {gpu_id} metric {metric_name}: missing min"
                        )
                        assert metric_value.max is not None, (
                            f"GPU {gpu_id} metric {metric_name}: missing max"
                        )
