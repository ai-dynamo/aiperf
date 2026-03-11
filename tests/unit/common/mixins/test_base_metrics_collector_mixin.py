# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.mixins.base_metrics_collector_mixin import (
    BaseMetricsCollectorMixin,
)
from aiperf.transports.http_defaults import AioHttpDefaults


class ConcreteCollector(BaseMetricsCollectorMixin[dict]):
    """Minimal concrete subclass for testing the abstract mixin."""

    async def _collect_and_process_metrics(self) -> None:
        pass


class TestTrustEnvPassedToSessions:
    """Test that trust_env is consistently passed to all aiohttp.ClientSession constructors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trust_env_value", [True, False])
    async def test_trust_env_passed_to_all_sessions(
        self,
        trust_env_value: bool,
        monkeypatch,
    ) -> None:
        """Test that TRUST_ENV is passed to both the persistent and temporary sessions."""
        monkeypatch.setattr(AioHttpDefaults, "TRUST_ENV", trust_env_value)

        collector = ConcreteCollector(
            endpoint_url="http://localhost:9400/metrics",
            collection_interval=1.0,
            reachability_timeout=5.0,
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            # First call: _initialize_http_client creates the persistent session
            mock_persistent = MagicMock()
            mock_persistent.close = AsyncMock()

            # Second call: is_url_reachable creates a temporary session (as context manager)
            mock_response = MagicMock(status=200)
            mock_response_cm = MagicMock()
            mock_response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response_cm.__aexit__ = AsyncMock(return_value=None)
            mock_temp = MagicMock()
            mock_temp.head = MagicMock(return_value=mock_response_cm)
            mock_temp_cm = MagicMock()
            mock_temp_cm.__aenter__ = AsyncMock(return_value=mock_temp)
            mock_temp_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.side_effect = [mock_persistent, mock_temp_cm]

            with patch(
                "aiperf.common.mixins.base_metrics_collector_mixin.create_tcp_connector"
            ) as mock_create:
                mock_conn = AsyncMock()
                mock_conn.close = AsyncMock()
                mock_create.return_value = mock_conn

                await collector._initialize_http_client()
                # Reset _session so is_url_reachable takes the temporary session path
                collector._session = None
                await collector.is_url_reachable()

            assert mock_session_class.call_count == 2
            for call in mock_session_class.call_args_list:
                assert call[1]["trust_env"] == trust_env_value
