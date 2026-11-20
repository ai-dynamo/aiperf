# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_export_attributes_success_and_failure_to_selected_url(
    cli: AIPerfCLI,
    mock_server_factory: Callable[..., AbstractAsyncContextManager[AIPerfMockServer]],
) -> None:
    async with (
        mock_server_factory(fast=True, workers=1) as healthy,
        mock_server_factory(fast=True, workers=1, error_rate=100.0) as failing,
    ):
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {healthy.url} --url {failing.url} \
                --endpoint-type chat --streaming \
                --num-sessions 4 --concurrency 1 --workers-max 1 \
                --isl 16 --osl 8 --random-seed 42 --ui {defaults.ui}
            """,
            assert_success=False,
        )

    assert result.jsonl is not None
    assert len(result.jsonl) == 4
    healthy_records = [r for r in result.jsonl if r.metadata.target_url_index == 0]
    failing_records = [r for r in result.jsonl if r.metadata.target_url_index == 1]
    assert len(healthy_records) == len(failing_records) == 2
    assert all(r.error is None for r in healthy_records)
    assert all(r.error is not None for r in failing_records)
