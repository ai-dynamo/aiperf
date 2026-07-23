# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for BurstGPT trace custom dataset type."""

from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.utils import create_burst_gpt_csv_file


def _sample_rows() -> list[dict]:
    """A trimmed BurstGPT-shaped CSV with sub-second timestamps."""
    return [
        {"Timestamp": 0.0, "Model": "ChatGPT", "Request tokens": 472, "Response tokens": 18, "Total tokens": 490, "Log Type": "Conversation log"},
        {"Timestamp": 0.1, "Model": "ChatGPT", "Request tokens": 1087, "Response tokens": 230, "Total tokens": 1317, "Log Type": "Conversation log"},
        {"Timestamp": 0.2, "Model": "GPT-4", "Request tokens": 417, "Response tokens": 276, "Total tokens": 693, "Log Type": "Conversation log"},
        {"Timestamp": 0.3, "Model": "ChatGPT", "Request tokens": 1360, "Response tokens": 647, "Total tokens": 2007, "Log Type": "Conversation log"},
        {"Timestamp": 0.4, "Model": "ChatGPT", "Request tokens": 185, "Response tokens": 215, "Total tokens": 400, "Log Type": "Conversation log"},
    ]  # fmt: skip


@pytest.mark.integration
@pytest.mark.asyncio
class TestBurstGPTTraceIntegration:
    """Integration tests for burst_gpt_trace dataset loader."""

    async def test_fixed_schedule_with_explicit_dataset_type(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """``--custom-dataset-type burst_gpt_trace --fixed-schedule`` runs end-to-end."""
        rows = _sample_rows()
        csv_file = create_burst_gpt_csv_file(tmp_path, rows)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {csv_file} \
                --custom-dataset-type burst_gpt_trace \
                --request-count {len(rows)} \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == len(rows)
        assert result.has_all_outputs_except_inputs
        assert result.inputs is None, "trace datasets skip inputs.json"

    async def test_fixed_schedule_auto_detected(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        tmp_path: Path,
    ) -> None:
        """No ``--custom-dataset-type`` flag — the loader's ``can_load``"""
        rows = _sample_rows()
        csv_file = create_burst_gpt_csv_file(tmp_path, rows)

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --input-file {csv_file} \
                --request-count {len(rows)} \
                --fixed-schedule \
                --fixed-schedule-auto-offset \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )

        assert result.exit_code == 0
        assert result.request_count == len(rows)
        assert result.has_all_outputs_except_inputs
        assert result.inputs is None, "trace datasets skip inputs.json"
