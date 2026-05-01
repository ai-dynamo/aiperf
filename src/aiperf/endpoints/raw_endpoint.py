# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import RequestInfo
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

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Return the pre-built raw payload from request turns.

        During live requests the inference client bypasses this method via the
        payload_bytes / raw_payload fast paths.  This implementation exists so
        that downstream consumers (e.g. raw-export post-processor) can
        reconstruct the payload from the serialised RequestInfo.
        """
        if request_info.turns:
            turn = request_info.turns[-1]
            if turn.raw_payload is not None:
                return turn.raw_payload
        raise NotImplementedError(
            "RawEndpoint does not construct payloads and no raw_payload "
            "found on request turns. Use raw_payload or inputs_json dataset types."
        )
