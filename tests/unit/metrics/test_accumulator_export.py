# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final metric export must not race cancellation-path ingestion."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.metrics.accumulator import MetricsAccumulator


@pytest.mark.asyncio
async def test_cancelled_export_does_not_offload_live_column_store_read() -> None:
    """Ctrl+C can export while late records are still entering the accumulator."""
    accumulator = MetricsAccumulator.__new__(MetricsAccumulator)
    expected = MagicMock()
    accumulator._summarize_for_export_context = MagicMock(return_value=expected)
    context = MagicMock(cancelled=True)

    with patch(
        "aiperf.metrics.accumulator.asyncio.to_thread", new_callable=AsyncMock
    ) as to_thread:
        assert await accumulator.export_results(context) is expected

    to_thread.assert_not_awaited()
    accumulator._summarize_for_export_context.assert_called_once_with(context)
