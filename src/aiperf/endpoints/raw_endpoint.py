# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import RequestInfo
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.response_mixin import JMESPathResponseMixin


class RawEndpoint(JMESPathResponseMixin, BaseEndpoint):
    """Raw payload endpoint for verbatim API replay.

    Does not format payloads -- use with raw_payload or inputs_json dataset
    types.  Parses responses using auto-detection with optional JMESPath
    extraction via ``response_field`` in endpoint.extra.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_response_parser()

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        raise NotImplementedError(
            "RawEndpoint does not format payloads. "
            "Use raw_payload or inputs_json dataset types."
        )
