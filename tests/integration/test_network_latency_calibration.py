# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for network latency calibration.

Drives a full ``aiperf profile`` run against the mock server and asserts that
enabling calibration produces the per-sample JSONL artifact and the
``network_adjusted_*`` / ``network_rtt`` keys in the JSON export, while a
baseline run without the flag is unaffected (non-regression).
"""

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer, AIPerfResults
from tests.integration.conftest import IntegrationTestDefaults as defaults

_NETWORK_LATENCY_GLOB = "**/*network_latency.jsonl"
_ADJUSTED_KEYS = (
    "network_adjusted_time_to_first_token",
    "network_rtt",
)


def _load_json_export(results: AIPerfResults) -> dict:
    """Load the raw aiperf.json export as a dict (extra metric keys preserved)."""
    path = next(results.artifacts_dir.glob("**/*aiperf.json"), None)
    assert path is not None, "profile_export_aiperf.json was not written"
    return orjson.loads(path.read_bytes())


@pytest.mark.integration
@pytest.mark.asyncio
class TestNetworkLatencyCalibration:
    """Tests for the --network-latency-automatic feature."""

    async def test_calibration_writes_jsonl_and_adjusted_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Active probing emits the JSONL artifact and adjusted metric keys."""
        results = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-32B-Instruct \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --network-latency-automatic \
                --network-latency-ping-interval 0.05 \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert results.request_count == defaults.request_count

        jsonl_path = next(results.artifacts_dir.glob(_NETWORK_LATENCY_GLOB), None)
        assert jsonl_path is not None, "network latency JSONL artifact was not written"

        lines = [line for line in jsonl_path.read_text().splitlines() if line.strip()]
        assert lines, "network latency JSONL artifact is empty"
        for line in lines:
            sample = orjson.loads(line)
            assert "success" in sample
            assert sample["target_port"] is not None

        export = _load_json_export(results)
        for key in _ADJUSTED_KEYS:
            assert key in export, f"{key} missing from JSON export"

    async def test_baseline_without_flag_has_no_network_artifacts(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Non-regression: omitting the flag emits no network latency outputs."""
        results = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-32B-Instruct \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert results.request_count == defaults.request_count

        assert next(results.artifacts_dir.glob(_NETWORK_LATENCY_GLOB), None) is None

        export = _load_json_export(results)
        assert not any(key.startswith("network_adjusted_") for key in export)
        assert "network_rtt" not in export

    async def test_rtt_override_adjusts_without_probing(
        self, cli: AIPerfCLI, mock_server_factory
    ):
        """A fixed override produces adjusted metrics with no active probing.

        Uses a controlled-latency mock (TTFT well above the 5 ms override) so the
        subtraction never clamps at 0, letting us lock in the exact-shift property.
        """
        async with mock_server_factory(ttft=80, itl=10, workers=8) as server:
            results = await cli.run(
                f"""
                aiperf profile \
                    --model Qwen/Qwen2.5-32B-Instruct \
                    --url {server.url} \
                    --endpoint-type chat \
                    --streaming \
                    --network-latency-mean 5.0 \
                    --request-count {defaults.request_count} \
                    --concurrency {defaults.concurrency} \
                    --workers-max {defaults.workers_max} \
                    --ui {defaults.ui}
                """
            )
        assert results.request_count == defaults.request_count

        export = _load_json_export(results)
        for key in _ADJUSTED_KEYS:
            assert key in export, f"{key} missing from JSON export with override"

        # A fixed 5 ms mean is deterministic, so lock in the core correctness
        # property end-to-end: each adjusted latency is shifted down by exactly the
        # RTT and its standard deviation is unchanged (display units are ms).
        assert export["network_rtt"]["avg"] == pytest.approx(5.0, abs=1e-3)
        for raw_tag, adjusted_tag in (
            ("time_to_first_token", "network_adjusted_time_to_first_token"),
            ("request_latency", "network_adjusted_request_latency"),
        ):
            raw, adjusted = export[raw_tag], export[adjusted_tag]
            assert raw["avg"] - adjusted["avg"] == pytest.approx(5.0, abs=1e-2)
            assert raw["std"] == pytest.approx(adjusted["std"], abs=1e-6)

        # No active probing: the per-sample JSONL artifact should not be written.
        assert next(results.artifacts_dir.glob(_NETWORK_LATENCY_GLOB), None) is None
