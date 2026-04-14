# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.models import RequestInfo
from aiperf.common.types import RequestOutputT
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.response_mixin import JMESPathResponseMixin


class RawEndpoint(JMESPathResponseMixin, BaseEndpoint):
    """Fallback endpoint for non-standard APIs.

    Does not format payloads or append a URL path.  Parses responses using
    auto-detection with optional JMESPath extraction via ``response_field``
    in endpoint.extra.  Prefer a regular endpoint type (e.g. chat) when the
    target API is supported -- raw payloads bypass formatting regardless of
    endpoint type, and regular endpoints provide structured response parsing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_response_parser()

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT:
        raise NotImplementedError(
            "RawEndpoint does not format payloads. "
            "Use raw_payload or inputs_json dataset types."
        )
