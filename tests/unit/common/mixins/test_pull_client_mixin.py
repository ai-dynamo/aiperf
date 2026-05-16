# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from aiperf.common.enums import CommAddress
from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.common.mixins.pull_client_mixin import PullClientMixin


class _DummyPullService(PullClientMixin):
    pass


def test_pull_client_mixin_forwards_custom_codec() -> None:
    codec = object()

    def fake_communication_init(self, **_: object) -> None:
        self.comms = MagicMock()

    with patch.object(CommunicationMixin, "__init__", fake_communication_init):
        service = _DummyPullService(
            run=MagicMock(),
            pull_client_address=CommAddress.RAW_INFERENCE_PROXY_BACKEND,
            pull_client_codec=codec,
        )

    service.comms.create_pull_client.assert_called_once_with(
        CommAddress.RAW_INFERENCE_PROXY_BACKEND,
        bind=False,
        max_pull_concurrency=None,
        additional_bind_address=None,
        codec=codec,
    )
